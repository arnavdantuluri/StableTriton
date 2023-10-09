# TODO: Need to test with nested modules and see if it still works
import torch

from flashfuse.kernels.layer_norm import layer_norm
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
        self.dropout = nn.LayerNorm((5,5), eps=1e-05, elementwise_affine=True)
    
    def forward(self, x):
        return self.dropout(self.nonlin(self.lin3(self.lin2(self.lin1(x)))))


def layer_norm_wrapper(v: torch.Tensor, layernorm: torch.nn.LayerNorm):
    # small hack to avoid casting weights/bias at each call
    if v.dtype == torch.float16:
        if layernorm.weight.dtype == torch.float32:
            layernorm.weight.data = layernorm.weight.data.half()
        if layernorm.bias.dtype == torch.float32:
            layernorm.bias.data = layernorm.bias.data.half()

    return layer_norm(v, layernorm.weight, layernorm.bias, layernorm.eps)



torch.fx.wrap("layer_norm_wrapper")
torch.fx.wrap("layer_norm_rms_wrapper")


def replace_layer_norm(gm: torch.fx.GraphModule):
    class Pattern(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layernorm = torch.nn.LayerNorm((1, 1))

        def forward(self, v):
            return self.layernorm(v)

    class Replacement(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layernorm = torch.nn.LayerNorm((1, 1))

        def forward(self, v):
            return layer_norm_wrapper(v, self.layernorm)

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