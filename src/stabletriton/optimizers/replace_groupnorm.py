# TODO: Need to debug
import torch

from stabletriton.kernels.groupnorm import groupnorm_wrapper
from stabletriton.optimizers.utils.util import replace_pattern
import torch.nn as nn
import torch.fx as fx

class subpattern(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.groupnorm = torch.nn.GroupNorm(32, 64, eps=1e-06, affine=True)
        self.groupnorm1 = torch.nn.GroupNorm(32, 64, eps=1e-06, affine=True)

    def forward(self, v):
        return self.groupnorm1(self.groupnorm(v))


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(64, 64)
        self.lin2 = nn.Linear(64, 64)
        self.lin3 = nn.Linear(64, 64)
        self.nonlin = nn.SiLU()
        self.norm =  nn.GroupNorm(32, 64, eps=1e-06, affine=True)
        self.norm1 = subpattern()
    
    def forward(self, x):
        return self.norm1(self.norm(self.nonlin(self.lin3(self.lin2(self.lin1(x))))))


def group_norm_wrapper(v: torch.Tensor, groupnorm: torch.nn.GroupNorm):
    return groupnorm_wrapper(v, groupnorm.num_groups, groupnorm.weight, groupnorm.bias, groupnorm.eps)

torch.fx.wrap("group_norm_wrapper")

def replace_group_norm(gm: torch.fx.GraphModule):
    class Pattern(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.groupnorm = torch.nn.GroupNorm(32, 32)

        def forward(self, v):
            return self.groupnorm(v)

    class Replacement(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.groupnorm = torch.nn.GroupNorm(32, 32)

        def forward(self, v):
            return group_norm_wrapper(v, self.groupnorm)

    replace_pattern(gm, Pattern(), Replacement())

if __name__ == "__main__":
    m = M().cuda()
    fx_model = fx.symbolic_trace(m)
    old_traced = fx.symbolic_trace(m)
    replace_group_norm(fx_model)
    assert fx_model.code != old_traced.code, "Issue with fusion with fx graph"
    print("Fx Graph replacement was a success! Kernel Fusion works perfectly")
    # Test output
    x = torch.rand(1, 64, 64, dtype=torch.float32).cuda()
    out_old = m(x)
    out_fused = fx_model(x)
    # Uncomment below if you want a better understanding of what fx replacement is doing
    # Should see 3 "__main___group_norm_wrapper" functions since there are three groupnorms being replaced
    # print(fx_model.code) 
    # Some margin for triton code.
    assert ((out_fused - out_old).abs() < 1e-3).all(), "Outputs don't match"