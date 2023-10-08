from typing import Callable

import torch

from ...kernels.linear import sdxl_forward
from torch.fx import symbolic_trace, subgraph_rewriter
# TODO: Needs to be rewritten
def exists(val):
    return val is not None

def linear_wrapper(v: torch.Tensor, linear: torch.nn.Linear, activation=""):
    return linear_wrapper_functional(v, linear.weight, linear.bias, activation=activation)


torch.fx.wrap("linear_wrapper")


def linear_wrapper_functional(v: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, activation=""):
    # small hack to avoid casting weights/bias at each call
    if weight.dtype == torch.float32:
        weight.data = weight.data.half()
    if bias is not None and bias.dtype == torch.float32:
        bias.data = bias.data.half()

    return sdxl_forward(v, weight, bias, activation=activation)


torch.fx.wrap("linear_wrapper_functional")


def replace_linear_activation(gm: torch.fx.GraphModule, activation_module: Callable, activation: str):
    class Pattern(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1)
            self.activation = activation_module

        def forward(self, v):
            return self.activation(self.linear(v))

    class Replacement(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1)
            # All activations in sdxl are silu no need to specify what activation to use
            self.activation = True if exists(activation_module) else False

        def forward(self, v):
            return linear_wrapper(v, self.linear, activation=activation)

    subgraph_rewriter.replace_pattern(gm, Pattern(), Replacement())


def replace_linear(gm: torch.fx.GraphModule):
    class Pattern(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1)

        def forward(self, v):
            return self.linear(v)

    class Replacement(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1)

        def forward(self, v):
            return linear_wrapper(v, self.linear)

    subgraph_rewriter.replace_pattern(gm, Pattern(), Replacement())


def replace_linear_fn(gm: torch.fx.GraphModule):
    def pattern(v, weight, bias):
        return torch.nn.functional.linear(v, weight, bias)

    def replace(v, weight, bias):
        output = linear_wrapper_functional(v, weight, bias)
        return output

    subgraph_rewriter.replace_pattern(gm, pattern, replace)


# Todo: to be removed when we support dynamic constant match
def replace_linear_fn_constant_match(gm: torch.fx.GraphModule):
    def pattern(v, weight):
        return torch.nn.functional.linear(v, weight, None)

    def replace(v, weight):
        output = linear_wrapper_functional(v, weight, None)
        return output

    subgraph_rewriter.replace_pattern(gm, pattern, replace)

# TODO: Needs to be changed to work with sdxl linear layers
def replace_all_linear(gm: torch.fx.GraphModule):
    replace_linear_activation(gm, torch.nn.Tanh(), "tanh")
    replace_linear_activation(gm, torch.nn.ReLU(), "relu")
    replace_linear_activation(gm, torch.nn.functional.gelu, "gelu")
    replace_linear(gm)
    replace_linear_fn(gm)
    replace_linear_fn_constant_match(gm)