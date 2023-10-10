# TODO: Needs to be tested
from typing import Optional

import torch

from flashfuse.kernels.attention_fa2 import attention
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
    sm_scale: float,
) -> torch.Tensor:
    # When tensors are shaped for bmm, first dimension is used for both batch and heads. Our kernel supports tensors
    # with 4 dimensions, so we add another dimension of size 1 for heads.

    extend_head = q.dim() == 3
    if extend_head:
        q = q.unsqueeze(dim=1)
        k = k.unsqueeze(dim=1)
        v = v.unsqueeze(dim=1)
        output = output.unsqueeze(dim=1)

    # When there is a large difference between those dimensions, our kernel become inefficient
    # (almost no parallelization), so we use pytorch instead
    capability = torch.cuda.get_device_capability()
    if capability[0] < 8:
        assert flash_attn_available == True, "Attempting to use flash attention but it is not installed. Please install using pip install flash-attn==1.0.5 --no-deps --no-dependencies and attempt again. If you did not mean to use flash attention, double check and make sure the fa=False flag is set in compile_model()"
        output = flash_attention_cuda(q, k, v, output, sm_scale)
    else:    
        attention(q, k, v, output, sm_scale)

    if extend_head:
        output = output.squeeze(dim=1)
    return output


torch.fx.wrap("attention_wrapper")


def fuse_attention(gm: torch.fx.GraphModule):
    def pattern(q, k, v):
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * (q.shape[2] ** -0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output

    def replace(q, k, v):
        output = torch.empty_like(q)
        output = attention_wrapper(q, k, v, output, q.shape[2] ** -0.5)
        return output

    subgraph_rewriter.replace_pattern(gm, pattern, replace)

if __name__ == "__main__":
    m = Attention().cuda(5)
    fx_model = fx.symbolic_trace(m)
    old_traced = fx.symbolic_trace(m)
    fuse_attention(fx_model)
    assert fx_model.code != old_traced.code, "Issue with fusion with fx graph"
    print("Fx Graph replacement was a success! Kernel Fusion works perfectly")
    # Test output
    x = torch.rand(5, 5, dtype=torch.float32).cuda()
    out_old = m(x)
    out_fused = fx_model(x)
    # Some margin for triton code.
    assert ((out_fused - out_old).abs() < 1e-3).all(), "Outputs don't match"