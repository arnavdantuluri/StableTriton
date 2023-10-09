import triton
import triton.language as tl
import torch
import math
from einops import rearrange
from timestep import call

def torch_ref(num_channels, timesteps):
    half_dim = num_channels // 2
    exponent = -math.log(10000) * torch.arange(
        half_dim, dtype=torch.float32
    ).to(timesteps.device)
    exponent = exponent / (half_dim - 0.0)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    sin_emb = torch.sin(emb)
    cos_emb = torch.cos(emb)

    return sin_emb, cos_emb

if __name__ == "__main__":
    x = torch.rand((1, 256, 256), dtype=torch.float32).cuda()
    torch_sin, torch_cos = torch_ref(512, x)
    # Prove that emb = timesteps[:, None].float() * emb[None, :] just expands dim by 1
    # torch_sin, torch_cos = rearrange(torch_sin, "(b e) h w -> b e h w", e=1), rearrange(torch_cos, "(b e) h w -> b e h w", e=1)
    
    triton_sin, triton_cos = call(x)
    triton_sin, triton_cos = rearrange(triton_sin, "(b e) h w -> b e h w", e=1), rearrange(triton_cos, "(b e) h w -> b e h w", e=1)

    assert (torch_sin - triton_sin).all() < 1e-8, "Assertion does not hold, some issue in the triton kernel"
    assert (torch_cos - triton_cos).all() < 1e-8, "Assertion does not hold, some issue in the triton kernel"
    #benchmrk 
    @triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[
            i for i in range(64, 1024, 128)
        ],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'torch'],  # Possible values for `line_arg`.
        line_names=['Triton', 'Torch'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='vector-add-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    )
    )
    def benchmark(size, provider):
        x = torch.rand((1, 10, size, size), device='cuda', dtype=torch.float32)
        print(x.shape)
        num_channels = size * 2
        quantiles = [0.5, 0.2, 0.8]
        if provider == 'torch':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_ref(num_channels, x), quantiles=quantiles)
        if provider == 'triton':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: call(x), quantiles=quantiles)
        gbps = lambda ms: 12 * size / ms * 1e-6
        return min_ms

    benchmark.run(print_data=True, show_plots=True)