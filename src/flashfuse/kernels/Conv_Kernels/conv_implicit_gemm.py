import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'TILE_SIZE_M': 64, 'TILE_SIZE_N': 32, 'TILE_SIZE_K': 64, 'GROUP_SIZE_M': 1}, num_stages=5, num_warps=4),
    ],
    key=['N', 'C', 'H', 'W', 'R', 'S', 'K'],
)
@triton.jit
def implicit_gemm_fprop_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Problem Size 
    N, C, H, W, R, S, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    # A: [n,h,w,c]
    # B: [k,r,s,c]
    # C: [n,h,w,c]
    stride_An, stride_Ah, stride_Aw, stride_Ac,
    stride_Bk, stride_Br, stride_Bs, stride_Bc,
    stride_Cn, stride_Cp, stride_Cq, stride_Ck,
    # Meta-parameters
    TILE_SIZE_M: tl.constexpr, TILE_SIZE_N: tl.constexpr, TILE_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # will it affect if these are not constants?
    Pad_H = 1
    Pad_W = 1
    Stride_H = 1
    Stride_W = 1
    Dilation_H = 1
    Dilation_W = 1

    # logical problem size of the mapped GEMM
    P = ((H + Pad_H * 2 - R * Dilation_H) // Stride_H) + 1;
    Q = ((W + Pad_W * 2 - S * Dilation_W) // Stride_W) + 1;
    GEMM_M = N * P * Q
    GEMM_N = K
    GEMM_K = C * R * S

    # TODO: Regular cta swizzling for GEMM is reserved here.
    #       Need further study if we need other cta swizzling for 
    #       implicit_gemm_fprop
    #
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse
    # See above `L2 Cache Optimizations` section for details
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(GEMM_M, TILE_SIZE_M)
    num_pid_n = tl.cdiv(GEMM_N, TILE_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # a_ptrs is a block of [TILE_SIZE_M, TILE_SIZE_K] pointers
    # b_ptrs is a block of [TILE_SIZE_K, TILE_SIZE_n] pointers

    # range of implicit rows
    pq = pid_m * TILE_SIZE_M + tl.arange(0, TILE_SIZE_M)
    q = pq % Q
    p = pq // Q

    # range of GEMM_K
    k = pid_n * TILE_SIZE_N + tl.arange(0, TILE_SIZE_N)

    # range for reduction
    crs = tl.arange(0, TILE_SIZE_K)
    s = crs % S
    c = (crs // S) // R
    r = (crs // S) % R

    a_ptrs = a_ptr + q[:, None] * stride_Aw + \
                     p[:, None] * stride_Ah + \
                     r[None, :] * stride_Ah + \
                     s[None, :] * stride_Aw + \
                     c[None, :] * stride_Ac

    b_ptrs = b_ptr + r[:, None] * stride_Br + \
                     s[:, None] * stride_Bs + \
                     c[:, None] * stride_Bc + \
                     k[None, :] * stride_Bk

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix
    # We accumulate into a `[TILE_SIZE_M, TILE_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop
    accumulator = tl.zeros((TILE_SIZE_M, TILE_SIZE_N), dtype=tl.float32)
    for gemm_k in range(0, GEMM_K, TILE_SIZE_K):
        # Note that for simplicity, we don't apply a mask here.
        # This means that if K is not a multiple of BLOCK_SIZE_K,
        # this will access out-of-bounds memory and produce an
        # error or (worse!) incorrect results.
        # TODO: check if need to fix this for productive purpose
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        # We accumulate along the K dimension
        accumulator += tl.dot(a, b)

        # Advance the ptrs to the next K block
        crs += TILE_SIZE_K
        s = crs % S
        c = (crs // S) // R
        r = (crs // S) % R

        a_ptrs = a_ptr + q[:, None] * stride_Aw + \
                         p[:, None] * stride_Ah + \
                         r[None, :] * stride_Ah + \
                         s[None, :] * stride_Aw + \
                         c[None, :] * stride_Ac

        b_ptrs = b_ptr + r[:, None] * stride_Br + \
                         s[:, None] * stride_Bs + \
                         c[:, None] * stride_Bc + \
                         k[None, :] * stride_Bk

    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C
    offs_cm = pid_m * TILE_SIZE_M + tl.arange(0, TILE_SIZE_M)
    offs_cn = pid_n * TILE_SIZE_N + tl.arange(0, TILE_SIZE_N)
    # layout of C: NPQK
    c_ptrs = c_ptr + stride_Cq * offs_cm[:, None] + stride_Ck * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < GEMM_M) & (offs_cn[None, :] < GEMM_N)
    tl.store(c_ptrs, c, mask=c_mask)

# %%
# We can now create a convenience wrapper function that only takes two input tensors
# and (1) checks any shape constraint; (2) allocates the output; (3) launches the above kernel
def implicit_gemm_fprop(a, b, activation=None):
    # checks constraints
    # layout: a: nhwc
    #         b: krsc
    #         c: nhwc
    assert a.shape[3] == b.shape[3], "incompatible dimensions"
    assert a.is_contiguous(), "matrix A must be contiguous"
    assert b.is_contiguous(), "matrix B must be contiguous"
    N, H, W, C = a.shape
    K, R, S, C = b.shape
    assert (
        (C * R * S) % 32 == 0
    ), "We don't check memory-out-of-bounds with GEMM_K so GEMM_K must be divisible by BLOCK_SIZE_K"

    # will it affect if these are not constants?
    Pad_H = 1
    Pad_W = 1
    Stride_H = 1
    Stride_W = 1
    Dilation_H = 1
    Dilation_W = 1
    P = ((H + Pad_H * 2 - R * Dilation_H) // Stride_H) + 1;
    Q = ((W + Pad_W * 2 - S * Dilation_W) // Stride_W) + 1;

    # allocates output
    c = torch.empty((N, P, Q, K), device=a.device, dtype=a.dtype)
    # 1D launch kernel where each block gets its own program.
    GEMM_M = N * P * Q
    GEMM_N = K
    grid = lambda META: (
        triton.cdiv(GEMM_M, META['TILE_SIZE_M']) * triton.cdiv(GEMM_N, META['TILE_SIZE_N']),
    )
    implicit_gemm_fprop_kernel[grid](
        a, b, c,
        N, C, H, W, R, S, K,
        a.stride(0), a.stride(1), a.stride(2), a.stride(3),
        b.stride(0), b.stride(1), b.stride(2), b.stride(3),
        c.stride(0), c.stride(1), c.stride(2), c.stride(3),
    )
    return c


# %%
# Unit Test
# -----------
#
# We can test our custom matrix multiplication operation against a native torch implementation (i.e., cuBLAS)

torch.manual_seed(0)
UT_N = 16
UT_H = 28
UT_W = 28
UT_C = 128
UT_R = 3
UT_S = 3
UT_K = 128
a = torch.randn((UT_N, UT_H, UT_W, UT_C), device='cuda', dtype=torch.float16)
b = torch.randn((UT_K, UT_R, UT_S, UT_C), device='cuda', dtype=torch.float16)
triton_output = implicit_gemm_fprop(a, b)
conv = torch.nn.Conv2d(UT_C, UT_K, (UT_R, UT_S))
conv.weight.data = b
torch_output = conv(a, b)
print(f"triton_output={triton_output}")
print(f"torch_output={torch_output}")
if triton.testing.allclose(triton_output, torch_output):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")

# %%
# Benchmark
# --------------
#
# Square Matrix Performance
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# We can now compare the performance of our kernel against that of cuBLAS. Here we focus on square matrices, but feel free to arrange this script as you wish to benchmark any other matrix shape.


@triton.testing.perf_report(
    triton.testing.Benchmark(
        #x_names=['M', 'N', 'K'],  # argument names to use as an x-axis for the plot
        #x_vals=[
        #    128 * i for i in range(2, 33)
        #],  # different possible values for `x_name`
        x_names=['N',],
        x_vals=[16,],
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        # possible values for `line_arg``
        # line_vals=['cublas', 'cublas + relu', 'triton', 'triton + relu'],
        line_vals=['triton',],
        # label name for the lines
        #line_names=["cuBLAS", "cuBLAS (+ torch.nn.LeakyReLU)", "Triton", "Triton (+ LeakyReLU)"],
        line_names=["Triton",],
        # line styles
        #styles=[('green', '-'), ('green', '--'), ('blue', '-'), ('blue', '--')],
        styles=[('green', '-'),],
        ylabel="TFLOPS",  # label name for the y-axis
        plot_name="matmul-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={'H':28, 'W':28, 'C':128, 'K':128},
    )
)
def benchmark(N, H, W, C, K, provider):
    a = torch.randn((N, H, W, C), device='cuda', dtype=torch.float16)
    b = torch.randn((K, 3, 3, C), device='cuda', dtype=torch.float16)
    conv = torch.nn.Conv2d(C, K, (3, 3), bias=False)
    a_torch = torch.randn((N, C, H, W), device='cuda', dtype=torch.float16)
    conv.weight.data = b
    if provider == 'cublas':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: conv(a_torch))
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: implicit_gemm_fprop(a, b))
    print("N: {}, H: {}, W: {}, C: {}, K: {} latency: {}/{}/{}".format(N, H, W, C, K, ms, min_ms, max_ms))
    perf = lambda ms: 2 * N * H * W * K * C * 3 * 3 * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


benchmark.run(show_plots=True, print_data=True)
