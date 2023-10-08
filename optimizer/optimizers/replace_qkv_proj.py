from typing import Callable

import torch

from ...kernels.attention_proj import qkv_proj_wrapper
from torch.fx import subgraph_rewriter

def exists(val):
    return val is not None

def linear_wrapper(x: torch.Tensor, q_weight, k_weight: torch.Tensor, v_weight):
    B, M, N = x.shape
    N = N // 64
    H = 64
    return linear_wrapper_functional(x, q_weight, k_weight, v_weight, B, M, N, H)


torch.fx.wrap("linear_wrapper")


def linear_wrapper_functional(x: torch.Tensor, q_weight, k_weight: torch.Tensor, v_weight, B, M, N, H):
    k = torch.empty((B, M, N), dtype=torch.float16, device=x.device)
    v = torch.empty((B, M, N), dtype=torch.float16, device=x.device)
    # small hack to avoid casting weights/bias at each call
    if q_weight.dtype == torch.float32:
        q_weight.data = q_weight.data.half()
    if k_weight.dtype == torch.float32:
        k_weight.data = k_weight.data.half()
    if v_weight.dtype == torch.float32:
        v_weight.data = v_weight.data.half()

    return qkv_proj_wrapper(x, q_weight, k_weight, v_weight, n_heads=N, head_dim=H, k=k, v=v)


torch.fx.wrap("linear_wrapper_functional")


def replace_linear_activation(gm: torch.fx.GraphModule):
    class Pattern(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q = torch.nn.Linear(1, 1)
            self.k = torch.nn.Linear(1, 1)
            self.v = torch.nn.Linear(1, 1)


        def forward(self, v):
            return self.q(v), self.k(v), self.v(v)

    class Replacement(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q = torch.nn.Linear(1, 1)
            self.k = torch.nn.Linear(1, 1)
            self.v = torch.nn.Linear(1, 1)
            # All activations in sdxl are silu no need to specify what activation to use

        def forward(self, v):
            return linear_wrapper(v, self.q.weight, self.k.weight, self.v.weight)

    subgraph_rewriter.replace_pattern(gm, Pattern(), Replacement())

# TODO: Needs to be changed to work with sdxl qkv proj layers
def replace_all_linear(gm: torch.fx.GraphModule):
    replace_linear_activation(gm)
    replace_linear_activation(gm)
    replace_linear_activation(gm)