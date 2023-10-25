import torch
from torch.fx import symbolic_trace, subgraph_rewriter

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, w1, w2):
        m1 = torch.cat([w1, w2]).sum()
        m2 = torch.cat([w1, w2]).sum()
        return x + torch.max(m1) + torch.max(m2)

def pattern(w1, w2):
    return torch.cat([w1, w2]).sum()

def replacement(w1, w2):
    return torch.stack([w1, w2])

traced_module = symbolic_trace(M())

subgraph_rewriter.replace_pattern(traced_module, pattern, replacement)