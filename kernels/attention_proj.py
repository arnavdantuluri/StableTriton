# Q, K, V projection
import torch

import triton
import triton.language as tl

torch.manual_seed(1234)

@triton.jit
def rms_matmul_rbe(
        x_ptr, w_ptr, out_ptr,
        M, N, K,
        stride_x_batch, stride_x_m, stride_x_k,
        stride_w_k, stride_w_n,
        stride_out_batch, stride_out_m, stride_out_n,
        USE_FP8: tl.constexpr,
        EPS: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_batch = tl.program_id(axis=0)
    pid = tl.program_id(axis=1)
    pid_m = pid // tl.cdiv(N, BLOCK_SIZE_N)
    pid_n = pid % tl.cdiv(N, BLOCK_SIZE_N)

    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (pid_batch * stride_x_batch + offs_m[:, None] * stride_x_m + offs_k[None, :] * stride_x_k)
    w_ptrs = w_ptr + (offs_k[:, None] * stride_w_k + offs_n[None, :] * stride_w_n)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    x_sum = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        x = tl.load(x_ptrs)
        x_sum += tl.math.pow(x.to(tl.float32), 2)
        w = tl.load(w_ptrs)  # TODO add an assert that w is a multiple of BLOCK SIZE K
        if USE_FP8:
            w = w.to(tl.float8e5, bitcast=True)
            w = w.to(tl.float32)
            w = w.to(tl.float16)
        accumulator += tl.dot(x, w)
        x_ptrs += BLOCK_SIZE_K * stride_x_k
        w_ptrs += BLOCK_SIZE_K * stride_w_k

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    out_ptrs = out_ptr + (
                pid_batch * stride_out_batch + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n)
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    tl.store(out_ptrs, accumulator, mask=out_mask)

@triton.jit
def rms_matmul_rbe_qkv(x_ptr,
                       q_weight_ptr, k_weight_ptr, v_weight_ptr,
                       q_ptr, k_ptr, v_ptr,
                       M, N, K,
                       stride_x_batch, stride_x_m, stride_x_k,
                       stride_q_w_k, stride_q_w_n,
                       stride_k_w_k, stride_k_w_n,
                       stride_v_w_k, stride_v_w_n,
                       stride_q_batch, stride_q_m, stride_q_n,
                       stride_k_batch, stride_k_m, stride_k_n,
                       stride_v_batch, stride_v_m, stride_v_n,
                       USE_FP8: tl.constexpr,
                       EPS: tl.constexpr,
                       BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    # q
    rms_matmul_rbe(
        x_ptr=x_ptr,
        w_ptr=q_weight_ptr, out_ptr=q_ptr,
        M=M, N=N, K=K,
        stride_x_batch=stride_x_batch, stride_x_m=stride_x_m, stride_x_k=stride_x_k,
        stride_w_k=stride_q_w_k, stride_w_n=stride_q_w_n,
        stride_out_batch=stride_q_batch, stride_out_m=stride_q_m, stride_out_n=stride_q_n,
        USE_FP8=USE_FP8,
        EPS=EPS,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    # k
    rms_matmul_rbe(
        x_ptr=x_ptr,
        w_ptr=k_weight_ptr, out_ptr=k_ptr,
        M=M, N=N, K=K,
        stride_x_batch=stride_x_batch, stride_x_m=stride_x_m, stride_x_k=stride_x_k,
        stride_w_k=stride_k_w_k, stride_w_n=stride_k_w_n,
        stride_out_batch=stride_k_batch, stride_out_m=stride_k_m, stride_out_n=stride_k_n,
        USE_FP8=USE_FP8,
        EPS=EPS,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    # v
    rms_matmul_rbe(
        x_ptr=x_ptr,
        w_ptr=v_weight_ptr, out_ptr=v_ptr,
        M=M, N=N, K=K,
        stride_x_batch=stride_x_batch, stride_x_m=stride_x_m, stride_x_k=stride_x_k,
        stride_w_k=stride_v_w_k, stride_w_n=stride_v_w_n,
        stride_out_batch=stride_v_batch, stride_out_m=stride_v_m, stride_out_n=stride_v_n,
        USE_FP8=USE_FP8,
        EPS=EPS,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
    )


def rms_matmul_rbe_qkv_wrapper(x: torch.Tensor,
                               q_weight: torch.Tensor, k_weight: torch.Tensor, v_weight: torch.Tensor,
                               n_heads: int, head_dim: int,
                               k: torch.Tensor,
                               v: torch.Tensor,
                               eps: float = 1e-6, theta=10000.):
    assert q_weight.shape == k_weight.shape == v_weight.shape
    assert q_weight.dtype == k_weight.dtype == v_weight.dtype
    assert q_weight.dtype in [torch.float16, torch.int8]
    batch, M, K = x.shape

    q_weight_t = q_weight.t()
    k_weight_t = k_weight.t()
    v_weight_t = v_weight.t()
    K_W, N = q_weight_t.shape
    assert K == K_W
    q = torch.empty((batch, M, N), dtype=torch.float16, device=q_weight_t.device)

    k = k.view((batch, M, N))
    v = v.view((batch, M, N))
    assert k.dtype == k_weight.dtype
    assert v.dtype == v_weight.dtype

    q_ptr = triton.reinterpret(q, tl.float16)
    k_ptr = triton.reinterpret(k, tl.float8e5 if k.dtype == torch.int8 else tl.float16)
    v_ptr = triton.reinterpret(v, tl.float8e5 if v.dtype == torch.int8 else tl.float16)

    grid = lambda META: (
    batch, triton.cdiv(META["M"], META["BLOCK_SIZE_M"]) * triton.cdiv(META["N"], META["BLOCK_SIZE_N"]))

    rms_matmul_rbe_qkv[grid](
        x_ptr=x,
        q_weight_ptr=q_weight_t, k_weight_ptr=k_weight_t, v_weight_ptr=v_weight_t,
        q_ptr=q_ptr, k_ptr=k_ptr, v_ptr=v_ptr,
        M=M, N=N, K=K,
        stride_x_batch=x.stride(0), stride_x_m=x.stride(1), stride_x_k=x.stride(2),
        stride_q_w_k=q_weight_t.stride(0), stride_q_w_n=q_weight_t.stride(1),
        stride_k_w_k=k_weight_t.stride(0), stride_k_w_n=k_weight_t.stride(1),
        stride_v_w_k=v_weight_t.stride(0), stride_v_w_n=v_weight_t.stride(1),
        stride_q_batch=q.stride(0), stride_q_m=q.stride(1), stride_q_n=q.stride(2),
        stride_k_batch=k.stride(0), stride_k_m=k.stride(1), stride_k_n=k.stride(2),
        stride_v_batch=v.stride(0), stride_v_m=v.stride(1), stride_v_n=v.stride(2),
        USE_FP8=q_weight.dtype == torch.int8,
        THETA=theta,
        EPS=eps,
        BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64,
        num_stages=4, num_warps=4
    )
    q = q.view(batch, M, n_heads, head_dim)
    k = k.view(batch, M, n_heads, head_dim)
    v = v.view(batch, M, n_heads, head_dim)
    return q, k, v

