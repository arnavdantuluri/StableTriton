# TODO: Need to test with nested modules and see if it still works
import torch

from flashfuse.kernels.groupnorm import groupnorm_wrapper
from torch.fx import subgraph_rewriter
import torch.nn as nn
import torch.fx as fx

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(5, 5)
        self.lin2 = nn.Linear(5,5)
        self.lin3 = nn.Linear(5,5)
        self.nonlin = nn.SiLU()
        self.dropout =  nn.GroupNorm(32, 5, eps=1e-06, affine=True)
    
    def forward(self, x):
        return self.dropout(self.nonlin(self.lin3(self.lin2(self.lin1(x)))))


def layer_norm_wrapper(v: torch.Tensor, groupnorm: torch.nn.LayerNorm):
    return groupnorm_wrapper(v, groupnorm.num_groups, groupnorm.weight, groupnorm.bias, groupnorm.eps)

torch.fx.wrap("layer_norm_wrapper")
torch.fx.wrap("layer_norm_rms_wrapper")


def replace_layer_norm(gm: torch.fx.GraphModule):
    class Pattern(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.groupnorm = torch.nn.GroupNorm(32, 1)

        def forward(self, v):
            return self.groupnorm(v)

    class Replacement(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.groupnorm = torch.nn.GroupNorm(32, 1)

        def forward(self, v):
            return layer_norm_wrapper(v, self.groupnorm)

    subgraph_rewriter.replace_pattern(gm, Pattern(), Replacement())

if __name__ == "__main__":
    m = M().cuda()
    fx_model = fx.symbolic_trace(m)
    old_traced = fx.symbolic_trace(m)
    replace_layer_norm(fx_model)
    assert fx_model.code != old_traced.code, "Issue with fusion with fx graph"
    print("Fx Graph replacement was a success! Kernel Fusion works perfectly")
    # Test output
    x = torch.rand(5, 5, dtype=torch.float32).cuda()
    out_old = m(x)
    out_fused = fx_model(x)
    # Some margin for triton code.
    assert ((out_fused - out_old).abs() < 1e-3).all(), "Outputs don't match"