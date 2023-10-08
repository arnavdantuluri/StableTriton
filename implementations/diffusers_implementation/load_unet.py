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
)

unet_new.load_state_dict(pipe.unet.state_dict())
# use the weights
pipe.unet = unet_new
traced_module = symbolic_trace(unet_new)

orig_stdout = sys.stdout
f = open('named_modules.txt', 'w')
sys.stdout = f

for module in traced_module.named_modules():
    print(module[1])

sys.stdout = orig_stdout
f.close()