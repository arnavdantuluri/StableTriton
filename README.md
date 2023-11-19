# StableTriton (WIP)
A Triton inference engine built in the same grain as Kernl https://github.com/ELS-RD/kernl/ but for diffusion models, specifically for SDXL

# Usage
When done you will simply need to run model = compile(model). As long as it is a nn.Module torch.fx should overwrite native torch ops with triton ops. Will initally set it up with ComfyUI and Diffusers. Automatic1111 will be worked on soon after
