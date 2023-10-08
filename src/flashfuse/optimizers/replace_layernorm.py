import torch

from ...kernels.layer_norm import layer_norm
from torch.fx import subgraph_rewriter
# TODO: Needs to be rewritten

def layer_norm_wrapper(v: torch.Tensor, layernorm: torch.nn.LayerNorm):
    # small hack to avoid casting weights/bias at each call
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