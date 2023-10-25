# Let linear ffn in geglu be replaced with triton impl and this replaces state * gelu(gate)
import triton 
import triton.language as tl
from triton import jit

import torch
import math

sqrt2 = math.sqrt(2.0)

@triton.jit
def gelu(x):
    """Gaussian Error Linear Unit (GELU)"""
    return x * 0.5 * (1.0 + tl.math.erf(x / sqrt2))

# returns o = state_ptr * gelu(gate_ptr)
@jit
def geglu_kernel(state_ptr, gate_ptr, output_ptr, xnumel, xblock: tl.constexpr):
    pid = tl.program_id(0)
    xidx = xblock * pid
    offsets = xidx + tl.arange(0, xblock)
    xmask = offsets < xnumel
    state = tl.load(state_ptr + offsets, xmask)
    gate = tl.load(gate_ptr + offsets, xmask)
    output = state * gelu(gate)
    tl.store(output_ptr + offsets, output, xmask)

def geglu_wrapper(state, gate):
    assert state.is_contiguous()
    assert gate.is_contiguous()
    output = torch.empty_like(state)
    n_elements = state.numel() # state and gate elements should be the same
    grid = lambda meta: (triton.cdiv(n_elements, meta['xblock']),)
    geglu_kernel[grid](state, gate, output, n_elements, xblock=1024) # 1024 set arbitrarily to be same as dropout op since both are elementwise ops
    return output

if __name__ == "__main__":
    # Test correctness of output
    state = torch.rand(5, 5).cuda()
    gate = torch.rand(5, 5).cuda()
    output_pt = state * torch.nn.functional.gelu(gate)
    output_tri = geglu_wrapper(state, gate)
    assert ((output_pt - output_tri).abs() < 1e-3).all(), "Outputs don't match"
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['size'],  # Argument names to use as an x-axis for the plot.
            x_vals=[
                2 ** i for i in range(12, 17, 1)
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
        x = torch.rand(size, device='cuda', dtype=torch.float32)
        y = torch.rand(size, device='cuda', dtype=torch.float32)
        quantiles = [0.5, 0.2, 0.8]
        if provider == 'torch':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: x * torch.nn.functional.gelu(y), quantiles=quantiles)
            print("Torch:", min_ms)
        if provider == 'triton':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: geglu_wrapper(x, y), quantiles=quantiles)
            print("Triton:", min_ms)
        gbps = lambda ms: 12 * size / ms * 1e-6
        return gbps(ms), gbps(max_ms), gbps(min_ms)

    benchmark.run(print_data=True, show_plots=True)