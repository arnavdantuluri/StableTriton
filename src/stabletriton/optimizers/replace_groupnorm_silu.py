import torch.fx as fx
import torch
import torch.nn as nn
import copy
from torch.fx.experimental.optimization import  matches_module_pattern, replace_node_module
from stabletriton.kernels.groupnorm import groupnorm_wrapper
from functools import partial

class subpattern(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.groupnorm = torch.nn.GroupNorm(32, 64, eps=1e-06, affine=True)
        self.groupnorm1 = torch.nn.SiLU()

    def forward(self, v):
        return self.groupnorm1(self.groupnorm(v))

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(64, 64)
        self.lin2 = nn.Linear(64, 64)
        self.lin3 = nn.GroupNorm(32, 64)
        self.nonlin = nn.SiLU()
        self.final = subpattern()
    
    def forward(self, x):
        return self.final(self.nonlin(self.lin3(self.lin2(self.lin1(x)))))

def fuse_groupnorm_activ(model: torch.nn.Module, inplace=False) -> torch.nn.Module:
    """
    Fuses convolution/BN layers for inference purposes. Will deepcopy your
    model by default, but can modify the model inplace as well.
    """
    patterns = [(nn.GroupNorm, nn.SiLU)]
    if not inplace:
        model = copy.deepcopy(model)
    fx_model = fx.symbolic_trace(model)
    modules = dict(fx_model.named_modules())
    new_graph = copy.deepcopy(fx_model.graph)

    for pattern in patterns:
        for node in new_graph.nodes:
            if matches_module_pattern(pattern, node, modules):
                if len(node.args[0].users) > 1:  # Output of conv is used by other nodes
                    continue
                # Silu and Linear dependent on each other
                groupnorm = modules[node.args[0].target]
                fused_groupnorm = return_patched_groupnorm(groupnorm, True)
                replace_node_module(node.args[0], modules, fused_groupnorm)
                node.replace_all_uses_with(node.args[0])
                new_graph.erase_node(node)
    return fx.GraphModule(fx_model, new_graph)

def return_patched_groupnorm(gnorm, activation):
    mod = copy.deepcopy(gnorm)
    mod.forward = partial(fused_groupnorm_forward, gnorm, activation)
    return mod

def fused_groupnorm_forward(groupnorm, activation, x):
    return groupnorm_wrapper(x, groupnorm.num_groups, groupnorm.weight, groupnorm.bias, groupnorm.eps, activation=activation)   

if __name__ == "__main__":
    m = M().cuda()
    fx_model = fuse_groupnorm_activ(m)
    old_traced = fx.symbolic_trace(m)
    assert fx_model.code != old_traced.code, "Issue with fusion with fx graph"
    print("Fx Graph replacement was a success! Kernel Fusion works perfectly")
    # Test output
    x = torch.rand(1, 64, 64, dtype=torch.float32).cuda()
    out_old = m(x)
    out_fused = fx_model(x)
    # Uncomment below if you want a better understanding of what fx replacement is doing
    # Slightly different than regular ones since there is no replace_graph happening we are just overwriting the forward and fusing gnorm and silu together
    # print(fx_model.code)
    # Some margin for triton code.
    assert ((out_fused - out_old).abs() < 1e-3).all(), "Outputs don't match"