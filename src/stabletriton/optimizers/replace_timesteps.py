from typing import Optional

import torch

from flashfuse.kernels.timestep import timstep_triton
import torch.nn as nn
from torch.fx import subgraph_rewriter
import torch.fx as fx
import math

class Timesteps(nn.Module):
    def __init__(self, num_channels: int = 320):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps):
        print(timesteps.shape)
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(
            half_dim, dtype=torch.float32
        ).to(timesteps.device)
        exponent = exponent / (half_dim - 0.0)

        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]

        sin_emb = torch.sin(emb)
        cos_emb = torch.cos(emb)
        emb = torch.cat([cos_emb, sin_emb], dim=-1)

        return emb

def timestep_wrapper(
    x: torch.Tensor,
) -> torch.Tensor:
    sin, cos = timstep_triton(x)
    return sin, cos


torch.fx.wrap("timestep_wrapper")


def fuse_timesteps(gm: torch.fx.GraphModule):
    def pattern(x, exponent):
        exponent = exponent / (x.shape[2] - 0.0)

        emb = torch.exp(exponent)
        emb = x[:, None].float() * emb[None, :]

        sin_emb = torch.sin(emb)
        cos_emb = torch.cos(emb)
        return sin_emb, cos_emb

    def replace(x, exponent):
        sin, cos = timestep_wrapper(x)
        return sin, cos

    subgraph_rewriter.replace_pattern(gm, pattern, replace)

if __name__ == "__main__":
    m = Timesteps(num_channels=10).cuda()
    fx_model = fx.symbolic_trace(m)
    old_traced = fx.symbolic_trace(m)
    fuse_timesteps(fx_model)
    assert fx_model.code != old_traced.code, "Issue with fusion with fx graph"
    print("Fx Graph replacement was a success! Kernel Fusion works perfectly")
    # Test output
    x = torch.rand(5, 5, dtype=torch.float32).cuda()
    out_old = m(x)
    out_fused = fx_model(x)
    # Some margin for triton code.
    assert ((out_fused - out_old).abs() < 1e-3).all(), "Outputs don't match"