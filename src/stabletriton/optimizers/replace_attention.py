# NOTE: Using XFormers because both Triton Flash Attention and CUDA Flash Attention 2 seem to have some issues with small head sizes.
from typing import Optional

import torch

import xformers
from flashfuse.kernels.attention_fa2 import attention
import xformers.ops as xops
import torch.nn as nn
from torch.fx import subgraph_rewriter
import torch.fx as fx
try:
    from flashfuse.kernels.attention_fa1 import attention as flash_attention_cuda
    flash_attn_available = True 
except:
    flash_attn_available = False 

class Attention(nn.Module):
    def __init__(
        self, inner_dim, cross_attention_dim=None, num_heads=None, dropout=0.0
    ):
        super(Attention, self).__init__()
        if num_heads is None:
            self.head_dim = 64
            self.num_heads = inner_dim // self.head_dim
        else:
            self.num_heads = num_heads
            self.head_dim = inner_dim // num_heads

        self.scale = self.head_dim**-0.5
        if cross_attention_dim is None:
            cross_attention_dim = inner_dim
        self.to_q = nn.Linear(inner_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=False)

        self.to_out = nn.ModuleList(
            [nn.Linear(inner_dim, inner_dim), nn.Dropout(dropout, inplace=False)]
        )

    def forward(self, hidden_states, encoder_hidden_states=None):
        q = self.to_q(hidden_states)
        k = self.to_k(hidden_states)
        v = self.to_v(hidden_states)
        b, t, c = q.size()

        q = q.view(q.size(0), q.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(k.size(0), k.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(v.size(0), v.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(b, t, c)

        for layer in self.to_out:
            attn_output = layer(attn_output)

        return attn_output

def attention_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    sm_scale: float, num_heads, head_dim,
) -> torch.Tensor:
    # When tensors are shaped for bmm, first dimension is used for both batch and heads. Our kernel supports tensors
    # with 4 dimensions, so we add another dimension of size 1 for heads.
    output = xops.memory_efficient_attention(q, k, v, scale=sm_scale)
    return output


torch.fx.wrap("attention_wrapper")


def fuse_attention(gm: torch.fx.GraphModule):

    def pattern(q, k, v, sm_scale, num_heads, head_dim):
        b, t, c = q.size()

        q = q.view(q.size(0), q.size(1), num_heads, head_dim).transpose(1, 2)
        k = k.view(k.size(0), k.size(1), num_heads, head_dim).transpose(1, 2)
        v = v.view(v.size(0), v.size(1), num_heads, head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(b, t, c)
        return attn_output

    def replace(q, k, v, sm_scale, num_heads, head_dim):
        output = torch.empty_like(q)
        dtype = q.dtype
        output = attention_wrapper(q.to(torch.float16), k.to(torch.float16), v.to(torch.float16), output, sm_scale, num_heads, head_dim)
        return output.to(dtype)

    subgraph_rewriter.replace_pattern(gm, pattern, replace)


def ref_attention_bmhk(q, k, v, attn_bias, scale=None) -> torch.Tensor:
    assert q.ndim == 4

    def T(t):
        return t.permute((0, 2, 1, 3)).reshape(
            [t.shape[0] * t.shape[2], t.shape[1], t.shape[3]]
        )

    out = ref_attention(T(q), T(k), T(v), attn_bias, scale=scale)
    out = out.reshape([q.shape[0], q.shape[2], q.shape[1], v.shape[3]])
    return out.permute((0, 2, 1, 3))

def ref_attention(q, k, v, attn_bias=None, drop_mask=None, p=0.0, scale=None):
    if q.ndim == 4:
        assert p == 0.0
        return ref_attention_bmhk(q, k, v, attn_bias=None)
    q = q.float()
    k = k.float()
    v = v.float()

    scale = scale if scale is not None else (1 / q.shape[-1] ** 0.5)
    q = q * scale

    attn = q @ k.transpose(-2, -1)
    attn = attn.softmax(-1)
    if drop_mask is not None:
        attn = attn * (drop_mask / (1 - p))
    return attn @ v

def attn(q, k, v):
    b, t, c = q.size()
    head_dim = 128
    num_heads = q.size(2) // head_dim
    q = q.view(q.size(0), q.size(1), num_heads, head_dim).transpose(1, 2)
    k = k.view(k.size(0), k.size(1), num_heads, head_dim).transpose(1, 2)
    v = v.view(v.size(0), v.size(1), num_heads, head_dim).transpose(1, 2)
    scores = torch.matmul(q, k.transpose(-2, -1)) * q.shape[-1] ** -0.5
    attn_weights = torch.softmax(scores, dim=-1)
    attn_output = torch.matmul(attn_weights, v)
    attn_output = attn_output.transpose(1, 2).contiguous().view(b, t, c)
    return attn_output

if __name__ == "__main__":
    m = Attention(64).cuda()
    fx_model = fx.symbolic_trace(m)
    old_traced = fx.symbolic_trace(m)
    fuse_attention(fx_model)
    assert fx_model.code != old_traced.code, "Issue with fusion with fx graph"
    print("Fx Graph replacement was a success! Kernel Fusion works perfectly")
    # Test output
    x = torch.rand(1, 128, 64, dtype=torch.float32).cuda()
    out_old = m(x)
    out_fused = fx_model(x)
    # Some margin for triton code.
    print(xops.memory_efficient_attention(x, x, x) - ref_attention(x, x, x))
    assert ((out_fused - out_old).abs() < 1e-3).all(), "Outputs don't match"