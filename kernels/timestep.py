#Triton kernels are slower than vanilla torch ~5x slower at [1, 1024, 1024]
import triton
import triton.language as tl
import torch
import math
from einops import rearrange

# Torch Inductor Kernel for inspiration
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel: tl.constexpr, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % XBLOCK
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = x0
    tmp2 = tmp1.to(tl.float64)
    tmp3 = 1.0
    tmp4 = tmp2 * tmp3
    tmp5 = 0.0
    tmp6 = tmp4 + tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp8 = -9.210340371976184
    tmp9 = tmp7 * tmp8
    tmp10 = XBLOCK
    tmp11 = tmp9 / tmp10
    tmp12 = tl.exp(tmp11)
    tmp13 = tmp0 * tmp12
    tmp14 = tl.sin(tmp13)
    tmp15 = tl.cos(tmp13)
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp14, xmask)
    tl.store(out_ptr1 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp15, xmask)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32,  'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
    ],
    key=['m_shape', 'k_shape'],
)
@triton.jit
def sinusoidal_kernel(x_ptrs, sin, cos,
                      b_shape: tl.constexpr, m_shape, k_shape: tl.constexpr, 
                      b_stride, m_stride, k_stride, 
                      half_dim: tl.constexpr, base: tl.constexpr, 
                      e: tl.constexpr,
                      BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    
    sin_emb = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    cos_emb = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    exponent = tl.zeros((BLOCK_SIZE_M, ), dtype=tl.float32)
    exponent += base * tl.arange(0, BLOCK_SIZE_M)
    exponent = exponent / BLOCK_SIZE_M

    emb = tl.math.pow(e, exponent)

    x = tl.load(x_ptrs + offs_m)
    emb = x * emb

    sin_emb += tl.sin(emb)
    cos_emb += tl.cos(emb)

    tl.store(sin + (offs_m), sin_emb)
    tl.store(cos + (offs_m), cos_emb)

    # Seems returning has some issues within triton need to store in output tensor which is annoying
    # return sin_emb, cos_emb


def triton_wrapper(num_channels, timesteps):
    b_shape, m_shape, k_shape = timesteps.shape
    b_stride, m_stride, k_stride = timesteps.stride(0), timesteps.stride(1), timesteps.stride(2)
    m_block_size, k_block_size = m_shape, k_shape
    half_dim = num_channels // 2
    base = 10000
    e = 2.71828182846
    dtype = timesteps.dtype
    assert m_shape % m_block_size == 0, "M Shape needs to be multiple of M Block Size"
    assert k_shape % k_block_size == 0, "K Shape needs to be multiple of K Block Size"
    sin_emb = torch.empty(b_shape, m_shape, k_shape, dtype=timesteps.dtype).to(timesteps.device)
    cos_emb = torch.empty(b_shape, m_shape, k_shape, dtype=timesteps.dtype).to(timesteps.device)
    grid = lambda META: (
        timesteps.numel() // 128, 
        )
    triton_[grid](
        timesteps, sin_emb, cos_emb,
        timesteps.numel(), 128
        )
    return sin_emb, cos_emb

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
    
    triton_sin, triton_cos = triton_wrapper(512, x)
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
        x = torch.rand((1, size, size), device='cuda', dtype=torch.float32)
        print(x.shape)
        num_channels = size * 2
        quantiles = [0.5, 0.2, 0.8]
        if provider == 'torch':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_ref(num_channels, x), quantiles=quantiles)
        if provider == 'triton':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_wrapper(num_channels, x), quantiles=quantiles)
        gbps = lambda ms: 12 * size / ms * 1e-6
        return min_ms

    benchmark.run(print_data=True, show_plots=True)