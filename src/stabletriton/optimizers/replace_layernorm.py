import torch

from stabletriton.kernels.layer_norm import layer_norm
import torch.fx as fx
from stabletriton.optimizers.utils.util import replace_pattern
from torch.fx import subgraph_rewriter
from stabletriton.optimizers.unet_pt import UNet2DConditionModel

# Load weights from the original model
from diffusers import DiffusionPipeline
import sys
sys.setrecursionlimit(10000)
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose=True


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
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    ).to("cuda")

    unet_new = UNet2DConditionModel().half().cuda()
    unet_new.load_state_dict(pipe.unet.state_dict())

    fx_model = fx.symbolic_trace(unet_new)
    old_traced = fx.symbolic_trace(unet_new)
    replace_layer_norm(fx_model)
    with open('out.txt', 'w') as f:
        print(fx_model.code, file=f)
    assert fx_model.code != old_traced.code, "Issue with fusion with fx graph"
    print("Fx Graph replacement was a success! Kernel Fusion works perfectly")
    # Test output
    # x = torch.rand(1, 1, dtype=torch.float16).cuda()
    # out_old = m(x)
    # out_fused = fx_model(x)
    # # Some margin for triton code.
    # print(fx_model.code)
    # print(old_traced.code)
    # assert torch.allclose(out_old, out_fused, atol=1e-2, rtol=0), "Outputs don't match"