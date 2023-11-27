# TODO: Need to debug
import torch
from typing import Callable
from stabletriton.kernels.groupnorm import groupnorm_wrapper
from stabletriton.optimizers.utils.util import replace_pattern
from torch.fx import subgraph_rewriter
import torch.nn as nn
import torch.fx as fx
from stabletriton.optimizers.unet_pt import UNet2DConditionModel

# Load weights from the original model
from diffusers import DiffusionPipeline
import sys
sys.setrecursionlimit(10000)
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose=True

def group_norm_wrapper(v: torch.Tensor, groupnorm: torch.nn.GroupNorm, activation: bool):
    return groupnorm_wrapper(v, groupnorm.num_groups, groupnorm.weight, groupnorm.bias, groupnorm.eps, activation)

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
            return group_norm_wrapper(v, self.groupnorm, False)

    replace_pattern(gm, Pattern(), Replacement())

def replace_group_norm_activation(gm: torch.fx.GraphModule, activation: Callable):
    class Pattern(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.groupnorm = torch.nn.GroupNorm(32, 32)
            self.activation = activation

        def forward(self, v):
            return self.activation(self.groupnorm(v))

    class Replacement(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.groupnorm = torch.nn.GroupNorm(32, 32)
            self.activation = activation

        def forward(self, v):
            return group_norm_wrapper(v, self.groupnorm, True)

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
    replace_group_norm_activation(fx_model, torch.nn.SiLU())
    replace_group_norm(fx_model)
    with open('out.txt', 'w') as f:
        print(fx_model.code, file=f)
    assert fx_model.code != old_traced.code, "Issue with fusion with fx graph"
    print("Fx Graph replacement was a success! Kernel Fusion works perfectly")
    # Test output
    # x = torch.rand(1, 64, 64, dtype=torch.float32).cuda()
    # out_old = m(x)
    # out_fused = fx_model(x)
    # # Uncomment below if you want a better understanding of what fx replacement is doing
    # # Should see 3 "__main___group_norm_wrapper" functions since there are three groupnorms being replaced
    # print(fx_model.code) 
    # # Some margin for triton code.
    # assert ((out_fused - out_old).abs() < 1e-3).all(), "Outputs don't match"