# Irrelevant now; found a workaround
import torch
import torch.nn as nn
import copy
from functools import partial
from typing import Callable

class M(nn.Module):
    def __init__():
        super().__init__()
    
    def forward(self, x):
        return torch.add(x, x)

def patch_fwd(x):
    return torch.mul(x, x)

x = torch.rand(2, 2)
m = M()
replaced_m = copy.deepcopy(m)
replaced_m.forward = patch_fwd