# Linear, SiLU, Linear
import torch

import triton
import triton.language as tl

@triton.jit
def SiLU(input):
    return input * tl.sigmoid(input)

@triton.jit
def linear_sdxl(
    a_ptr, w1_ptr, w3_ptr, out_ptr,
    bias_1, bias_2,
    M, N, K,
    stride_am, stride_ak,
    stride_w1k, stride_w1n,
    stride_w3k, stride_w3n,
    stride_outm, stride_outn,
    USE_FP8: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    """
    w1 and w3 are weights (linear layers)
    F.silu(w1(x)) * w3(x)
    """
    pid = tl.program_id(axis=0)
    pid_m = pid // tl.cdiv(N, BLOCK_SIZE_N)
    pid_n = pid % tl.cdiv(N, BLOCK_SIZE_N)

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    w1_ptrs = w1_ptr + (offs_k[:, None] * stride_w1k + offs_bn[None, :] * stride_w1n)
    w3_ptrs = w3_ptr + (offs_k[:, None] * stride_w3k + offs_bn[None, :] * stride_w3n)
    acc1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    a_sum = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    for _ in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs)
        a_sum += tl.math.pow(a.to(tl.float32), 2)
        b = tl.load(w1_ptrs)
        acc1 += tl.dot(a, b)
        c = tl.load(w3_ptrs)
        acc2 += tl.dot(a, c)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        w1_ptrs += BLOCK_SIZE_K * stride_w1k
        w3_ptrs += BLOCK_SIZE_K * stride_w3k

    a_mean = tl.sum(a_sum, axis=1) / K + EPS
    a_norm = tl.math.rsqrt(a_mean)
    acc1 = acc1 * a_norm[:, None]
    acc2 = acc2 * a_norm[:, None]
    accumulator = (acc1 * tl.sigmoid(acc1)) * acc2

    offs_outm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_outn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    out_ptrs = out_ptr + (stride_outm * offs_outm[:, None] + stride_outn * offs_outn[None, :])
    out_mask = (offs_outm[:, None] < M) & (offs_outn[None, :] < N)
    tl.store(out_ptrs, accumulator, mask=out_mask)


def timestep_embedding(x: torch.Tensor, w1: torch.Tensor, w3: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.float16
    assert w1.dtype == w3.dtype
    assert w1.dtype in [torch.int8, torch.float16]
    assert w1.shape == w3.shape

    w1_t = w1.t()
    w3_t = w3.t()

    batch, seq_len, dim = x.shape
    M, K = batch * seq_len, dim

    N = w1_t.shape[1]
    assert K == w1_t.shape[0]
    assert w1_t.shape == w3_t.shape
    x_reshape = x.reshape(M, K)
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    grid = lambda META: (triton.cdiv(META["M"], META["BLOCK_SIZE_M"]) * triton.cdiv(META["N"], META["BLOCK_SIZE_N"]),)
    linear_sdxl[grid](
        x_reshape, w1_t, w3_t, out,
        M, N, K,
        *x_reshape.stride(),
        *w1_t.stride(),
        *w3_t.stride(),
        *out.stride(),
        USE_FP8=w1_t.dtype != torch.float16,
        EPS=1e-6,
        BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=64,
        num_stages=2, num_warps=4
    )
    out = out.view(batch, seq_len, -1)
    return out

if __name__ == "__main__":
    x = torch.randn([1, 16, 4096], dtype=torch.float16, device="cuda")
    # weights tends to be very small values
    w1_w = torch.randn([11008, 4096], dtype=torch.float16, device="cuda") * 0.2
    w3_w = torch.randn([11008, 4096], dtype=torch.float16, device="cuda") * 0.2


    w1_p = x @ w1_w.t()
    w1_silu_p = torch.nn.functional.silu(w1_p)
    w3_p = x @ w3_w.t()


    def ff_pytorch(x: torch.Tensor, w1: torch.Tensor, w3: torch.Tensor) -> torch.Tensor:
        a = torch.nn.functional.silu(torch.matmul(x, w1.t()))
        b = torch.matmul(x, w3.t())
        return a * b


    output_triton = timestep_embedding(x=x, w1=w1_w, w3=w3_w)
    output_pytorch = ff_pytorch(x=x, w1=w1_w, w3=w3_w)

    assert torch.allclose(output_triton, w1_silu_p * w3_p, atol=1e-1), f"max diff: {torch.max(torch.abs(output_triton - w1_silu_p * w3_p))}"
    assert torch.allclose(output_triton, output_pytorch, atol=1e-1), f"max diff: {torch.max(torch.abs(output_triton - output_pytorch))}"

    print("rms matmul silu mul triton", triton.testing.do_bench(lambda: timestep_embedding(x=x, w1=w1_w, w3=w3_w)))
    print("rms matmul silu mul pytorch", triton.testing.do_bench(lambda: ff_pytorch(x=x, w1=w1_w, w3=w3_w)))
    