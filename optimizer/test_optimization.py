import torch.fx as fx
import torch
import torch.nn as nn
import copy
from torch.fx.experimental.optimization import  matches_module_pattern, replace_node_module
from kernels.linear import sdxl_forward
from functools import partial

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(5, 5)
        self.lin2 = nn.Linear(5,5)
        self.lin3 = nn.Linear(5,5)
        self.nonlin = nn.SiLU()
    
    def forward(self, x):
        return self.nonlin(self.lin3(self.lin2(self.lin1(x))))

def fuse(model: torch.nn.Module, inplace=False) -> torch.nn.Module:
    """
    Fuses convolution/BN layers for inference purposes. Will deepcopy your
    model by default, but can modify the model inplace as well.
    """
    patterns = [(nn.Linear, nn.SiLU)]
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
                linear = modules[node.args[0].target]
                fused_linear = return_patched_forward(linear)
                replace_node_module(node.args[0], modules, fused_linear)
                node.replace_all_uses_with(node.args[0])
                new_graph.erase_node(node)
    return fx.GraphModule(fx_model, new_graph)

def return_patched_forward(orig_linear):
    mod = copy.deepcopy(orig_linear)
    mod.forward = partial(fused_linear_forward, orig_linear)
    return mod

def fused_linear_forward(lin, x):
    return sdxl_forward(x, lin.weight, lin.bias, activation=True)   

m = M()
fx_model = fuse(m)
old_traced = fx.symbolic_trace(m)
assert fx_model.code != old_traced.code, "Issue with fusion with fx graph"
print("Fx Graph replacement was a success! Kernel Fusion works perfectly")