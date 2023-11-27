from unet_pt import UNet2DConditionModel as UNet2DConditionModelPT
import torch
from collections import namedtuple
from PIL import Image, ImageChops

# Load weights from the original model
from diffusers import DiffusionPipeline
from stabletriton.optimization import optimize_model
import sys
from datetime import datetime

sys.setrecursionlimit(10000)
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose=True

fuse = False
pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
).to("cuda")

unet_new = UNet2DConditionModelPT().half().cuda()
unet_new.load_state_dict(pipe.unet.state_dict())

# 'We need to include some extra unet methodologies since other methods in sdxl pipeline are dependent on it'
unet_new = optimize_model(unet_new, cuda_graph=True)
unet_new.config = namedtuple(
        "config", "in_channels addition_time_embed_dim sample_size"
        )
unet_new.config.in_channels = 4
unet_new.config.addition_time_embed_dim = 256
unet_new.config.sample_size = 128
pipe.unet = unet_new

prompt = "a photo of an astronaut riding a horse on mars"

image = pipe(prompt).images[0]
del image 

#5.94it/s with pytorch
#8.58it/s with triton and cuda graphs
#Extra call once triton kernels have been autotuned and cuda graphs captured to get accurate it/s measure

image = pipe(prompt).images[0]