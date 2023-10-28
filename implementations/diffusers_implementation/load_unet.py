from unet import UNet2DConditionModel
import torch

# Load weights from the original model
from diffusers import DiffusionPipeline
from stabletriton.optimization import optimize_model
import sys
from datetime import datetime

sys.setrecursionlimit(10000)
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose=True

pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
).to("cuda")

unet_new = UNet2DConditionModel().half().cuda()
unet_new.load_state_dict(pipe.unet.state_dict())

# use the weights
pipe.unet = unet_new.cuda()

prompt = "a photo of an astronaut riding a horse on mars"

image = pipe(prompt).images[0]
del image
optimize_model(unet_new)

#Run a sample input through unet to compile kernels before measuring it/s
sample = torch.rand(2, 4, 128, 128).cuda().half()
timesteps = torch.rand([]).cuda()
encoder_hidden_states = torch.rand(2, 77, 2048).cuda().half()
added_cond_kwargs = {
    'text_embeds': torch.rand(2, 1280).cuda().half(),
    'time_ids': torch.rand(2, 6).cuda().half(),
}
output = unet_new(sample, timesteps, encoder_hidden_states, added_cond_kwargs)
del output

pipe.unet = unet_new.cuda()
image = pipe(prompt).images[0]

#region Test Unet and fused unet
# sample = torch.rand(2, 4, 128, 128).cuda().half()
# timesteps = torch.rand([]).cuda()
# encoder_hidden_states = torch.rand(2, 77, 2048).cuda().half()
# added_cond_kwargs = {
#     'text_embeds': torch.rand(2, 1280).cuda().half(),
#     'time_ids': torch.rand(2, 6).cuda().half(),
# }

# startTime = datetime.now()
# output = unet_new(sample, timesteps, encoder_hidden_states, added_cond_kwargs)
# print(datetime.now() - startTime)
# del output
# torch.cuda.empty_cache()

# optimize_model(unet_new)
# output = unet_new(sample, timesteps, encoder_hidden_states, added_cond_kwargs)
# del output

# startTime = datetime.now()
# output = unet_new(sample, timesteps, encoder_hidden_states, added_cond_kwargs)
# print(datetime.now() - startTime)
#endregion