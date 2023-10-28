import torch

from stabletriton.kernels.layer_norm import _layer_norm_fwd_fused_single_pass, layer_norm
import torch.fx as fx
from stabletriton.optimizers.utils.util import replace_pattern

class subpattern(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layernorm = torch.nn.LayerNorm((1, 1))
        self.layernorm1 = torch.nn.LayerNorm((1, 1))

    def forward(self, v):
        return self.layernorm1(self.layernorm(v))

class Pattern(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layernorm = torch.nn.LayerNorm((1, 1))
        self.layernorm1 = torch.nn.LayerNorm((1, 1))
        self.layernorm2 = subpattern()

    def forward(self, v):
        return self.layernorm2(self.layernorm1(self.layernorm(v)))

def layer_norm_wrapper(v: torch.Tensor, layernorm: torch.nn.LayerNorm):
    # small hack to avoid casting weights/bias at each call
    if layernorm.weight.dtype == torch.float32:
        layernorm.weight.data = layernorm.weight.data.half()
    if layernorm.bias.dtype == torch.float32:
        layernorm.bias.data = layernorm.bias.data.half()

    return layer_norm(v, layernorm.weight, layernorm.bias, layernorm.eps)


torch.fx.wrap("layer_norm_wrapper")


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

    replace_pattern(gm, Pattern(), Replacement())

if __name__ == "__main__":
    m = Pattern().cuda().half()
    fx_model = fx.symbolic_trace(m)
    old_traced = fx.symbolic_trace(m)
    replace_layer_norm(fx_model)
    assert fx_model.code != old_traced.code, "Issue with fusion with fx graph"
    print("Fx Graph replacement was a success! Kernel Fusion works perfectly")
    # Test output
    x = torch.rand(1, 1, dtype=torch.float16).cuda()
    out_old = m(x)
    out_fused = fx_model(x)
    # Some margin for triton code.
    print(fx_model.code)
    assert torch.allclose(out_old, out_fused, atol=1e-2, rtol=0), "Outputs don't match"