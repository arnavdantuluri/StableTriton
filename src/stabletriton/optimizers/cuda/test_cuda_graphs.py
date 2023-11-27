from stabletriton.optimizers import cuda_graphs_wrapper, static_inputs_pool
from stabletriton.optimizers.unet_pt import UNet2DConditionModel
import torch
from collections import namedtuple
import torch._dynamo as torchdynamo
# Load weights from the original model
from diffusers import DiffusionPipeline
import sys

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
).to("cuda")

unet_new = UNet2DConditionModel().half().cuda()
unet_new.load_state_dict(pipe.unet.state_dict())

def compiler(gm: torch.fx.GraphModule, example_inputs):
    return cuda_graphs_wrapper(gm, example_inputs)

def optimize_model(model) -> None:
    assert torch.cuda.is_available(), "CUDA capacity is required to use Kernl"
    major, _ = torch.cuda.get_device_capability()
    if major < 8:
        raise RuntimeError("GPU compute capability 8.0 (Ampere) or higher is required to use Kernl")
    assert next(model.parameters()).device.type == "cuda", "Model must be on GPU"
    static_inputs_pool.clear()
    model.forward_original = model.forward

    @torchdynamo.optimize(compiler)
    def run(*args, **kwargs):
        return model.forward_original(*args, **kwargs)

    model.forward = run

optimize_model(unet_new)

pipe.unet = unet_new

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]
del image
image = pipe(prompt).images[0]