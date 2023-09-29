from unet import UNet2DConditionModel
import torch
from torch.fx import symbolic_trace, subgraph_rewriter
unet_new = UNet2DConditionModel().half()

# Load weights from the original model
from diffusers import DiffusionPipeline


pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
)

unet_new.load_state_dict(pipe.unet.state_dict())

# use the weights
pipe.unet = unet_new
print(unet_new)

prompt = "a photo of an astronaut riding a horse on mars"

image = pipe(prompt).images[0]