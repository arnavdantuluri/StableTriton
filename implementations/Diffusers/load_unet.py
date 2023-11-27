from implementations.Diffusers.unet_tri import UNet2DConditionModel
import torch

# Load weights from the original model
from diffusers import DiffusionPipeline
from stabletriton.optimization import optimize_model
import sys
from datetime import datetime
import copy
import triton

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


# region compile optimized model
sample = torch.rand(2, 4, 128, 128).cuda().half()
timesteps = torch.rand([]).cuda()
encoder_hidden_states = torch.rand(2, 77, 2048).cuda().half()
added_cond_kwargs = {
    'text_embeds': torch.rand(2, 1280).cuda().half(),
    'time_ids': torch.rand(2, 6).cuda().half(),
}

# startTime = datetime.now()
# output = unet_new(sample, timesteps, encoder_hidden_states, added_cond_kwargs)
# del output
# print(datetime.now() - startTime)
torch.cuda.empty_cache()

optimize_model(unet_new)
# unet_new(sample, timesteps, encoder_hidden_states, added_cond_kwargs)

startTime = datetime.now()
unet_new(sample, timesteps, encoder_hidden_states, added_cond_kwargs)
print(datetime.now() - startTime)
# endregion