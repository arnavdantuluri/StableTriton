from typing import Optional

import torch

from flashfuse.kernels.geglu import geglu_wrapper
from flashfuse.optimizers.replace_linear import replace_linear
import torch.nn as nn
from torch.fx import subgraph_rewriter
import torch.fx as fx
import math

class GEGLU(nn.Module):
    def __init__(self, in_features, out_features):
        super(GEGLU, self).__init__()
        self.proj = nn.Linear(in_features, out_features * 2, bias=True)

    def forward(self, x):
        x_proj = self.proj(x)
        x1, x2 = x_proj.chunk(2, dim=-1)
        return x1 * torch.nn.functional.gelu(x2)


def geglu_triton(
    state: torch.Tensor, gate: torch.Tensor,
) -> torch.Tensor:
    output = geglu_wrapper(state, gate)
    return output


torch.fx.wrap("geglu_triton")


def fuse_gelu(gm: torch.fx.GraphModule):
    def pattern(state, gate):
        return state * torch.nn.functional.gelu(gate)

    def replace(state, gate):
        output = geglu_triton(state.contiguous(), gate.contiguous())
        return output

    subgraph_rewriter.replace_pattern(gm, pattern, replace)

if __name__ == "__main__":
    m = GEGLU(5, 5).cuda()
    fx_model = fx.symbolic_trace(m)
    old_traced = fx.symbolic_trace(m)
    replace_linear(fx_model)
    fuse_gelu(fx_model)
    assert fx_model.code != old_traced.code, "Issue with fusion with fx graph"
    print("Fx Graph replacement was a success! Kernel Fusion works perfectly")
    print(fx_model.code)
    # Test output
    x = torch.rand(5, 5, dtype=torch.float32).cuda()
    out_old = m(x)
    out_fused = fx_model(x)
    # Some margin for triton code.
    assert ((out_fused - out_old).abs() < 1e-3).all(), "Outputs don't match"