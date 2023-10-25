import pytest
import torch

import torch
import triton.ops as ops

def test_op(Z, H, N_CTX, D_HEAD):
    dtype = torch.float16
    torch.manual_seed(20)
    q = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    k = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    v = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    sm_scale = 0.5
    # reference implementation
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    p = torch.softmax(p.float(), dim=-1).to(dtype)
    ref_out = torch.matmul(p, v)
    # triton implementation
    tri_out = ops.attention(q, k, v, False, sm_scale)
    # compare
    assert ((tri_out - ref_out).abs() < 1e-3).all(), "Outputs don't match"

test_op(1, 1, 16, 64)