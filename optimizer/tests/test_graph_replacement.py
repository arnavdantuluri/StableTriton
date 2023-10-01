# Needs to be done for every forward function in sdxl_original_unet
import torch
from torch.fx import symbolic_trace, subgraph_rewriter

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear((2, 2), bias=False)
        self.nonlin = torch.nn.SiLU()
        self.l2 = torch.nn.Linear((2, 2), bias=False)

    def forward(self, x):
        return self.l2(self.nonlin(self.l1(x)))

def pattern(x, weight):
    m = torch.nn.functional.linear(x, weight, None)

def replacement(x, weight):
    m = x + weight

traced_module = symbolic_trace(M())

print(traced_module.code)
subgraph_rewriter.replace_pattern(traced_module, pattern, replacement)
print(traced_module.code)
