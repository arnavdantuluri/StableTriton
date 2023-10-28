# Works with nested modules; Refer to test in replace_layernorm.py
from typing import Callable

import torch
import torch.nn as nn
from stabletriton.kernels.linear import sdxl_forward
from stabletriton.optimizers.utils.util import replace_pattern
import torch.fx as fx

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(5, 5)
        self.lin2 = nn.Linear(5,5)
        self.lin3 = nn.Linear(5,5)
        self.nonlin = nn.SiLU()
        self.dropout = nn.Dropout(p=0.0)
    
    def forward(self, x):
        return self.dropout(self.nonlin(self.lin3(self.lin2(self.lin1(x)))))

def linear_wrapper(v: torch.Tensor, linear: torch.nn.Linear):
    return linear_wrapper_functional(v, linear.weight, linear.bias)


torch.fx.wrap("linear_wrapper")

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(5, 5)
        self.lin2 = nn.Linear(5,5)
        self.lin3 = nn.Linear(5,5)
        self.nonlin = nn.SiLU()
    
    def forward(self, x):
        return self.nonlin(self.lin3(self.lin2(self.lin1(x))))


def linear_wrapper_functional(v: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    # small hack to avoid casting weights/bias at each call
    if v.dtype == torch.float16:
        if weight.dtype == torch.float32:
            weight.data = weight.data.half()
        if bias is not None and bias.dtype == torch.float32:
            bias.data = bias.data.half()

    return sdxl_forward(v, weight, bias, activation=False)


torch.fx.wrap("linear_wrapper_functional")

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

    replace_pattern(gm, Pattern(), Replacement())

if __name__ == "__main__":
    m = M().cuda()
    fx_model = fx.symbolic_trace(m)
    replace_linear(fx_model)
    old_traced = fx.symbolic_trace(m)
    assert fx_model.code != old_traced.code, "Issue with fusion with fx graph"
    print("Fx Graph replacement was a success! Kernel Fusion works perfectly")
    # Test output
    x = torch.rand(5, 5, dtype=torch.float32).cuda()
    out_old = m(x)
    out_fused = fx_model(x)
    print(fx_model.code)
    # Some margin for triton code.
    assert ((out_fused - out_old).abs() < 1e-3).all(), "Outputs don't match"