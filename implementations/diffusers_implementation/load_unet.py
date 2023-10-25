from unet import UNet2DConditionModel
import torch
import torch.nn as nn
from torch.fx import symbolic_trace, subgraph_rewriter
unet_new = UNet2DConditionModel().half()
import sys
# Load weights from the original model
from diffusers import DiffusionPipeline


pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
).cuda()

unet_new.load_state_dict(pipe.unet.state_dict())
# use the weights
pipe.unet = unet_new
gen = torch.Generator().manual_seed(42)
pipe("a colorful, movie like photo of dog", generator=gen).images[0]