'''
Potentially outdated, need to test but I believe subgraph_rewriter does not require you to specify exactly what nodes to replace.
I'm fairly certain it does so with whatever nodes follow the pattern listed
Pattern-matching is done based on use-def relationships, not node names; I was right!
'''
# TODO: Needs to be rewritten
from typing import Optional

import torch

from ...kernels.attention_fa2 import attention
from torch.fx import symbolic_trace, subgraph_rewriter
try:
    from ...kernels.attention_fa1 import attention as flash_attention_cuda
    flash_attn_available = True 
except:
    flash_attn_available = False 


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
        del output
        output = flash_attention_cuda(q, k, v, output, sm_scale)
    else:    
        attention(q, k, v, output, sm_scale)

    if extend_head:
        output = output.squeeze(dim=1)
    return output


torch.fx.wrap("attention_wrapper")


def fuse_attention_pattern_1(gm: torch.fx.GraphModule):
    def pattern(q, k, attention_mask, v):
        transpose_10 = k.transpose(-1, -2)
        matmul_20 = torch.matmul(q, transpose_10)
        truediv_10 = matmul_20 / 8.0
        add_30 = truediv_10 + attention_mask
        softmax_10 = torch.nn.functional.softmax(add_30, dim=-1)
        matmul_21 = torch.matmul(softmax_10, v)
        return matmul_21

    def replace(q, k, v):
        output = torch.empty_like(q)
        output = attention_wrapper(q, k, v, output, 1 / 8.0)
        return output

    subgraph_rewriter.replace_pattern(gm, pattern, replace)


def fuse_attention_pattern_2(gm: torch.fx.GraphModule):
    def pattern(q, k, v):
        transpose_3 = k.transpose(3, 2)
        matmul = torch.matmul(q, transpose_3)
        softmax = torch.nn.functional.softmax(matmul, dim=-1)
        matmul_1 = torch.matmul(softmax, v)
        return matmul_1

    def replace(q, k, v):
        output = torch.empty_like(q)
        output = attention_wrapper(q, k, v, output, 1.0)
        return output

    subgraph_rewriter.replace_pattern(gm, pattern, replace)


def fuse_attention_pattern_3(gm: torch.fx.GraphModule):
    def pattern(q, k, v):
        transpose_46 = k.transpose(1, 2)
        bmm_22 = torch.bmm(q, transpose_46)
        softmax_11 = torch.nn.functional.softmax(bmm_22, dim=-1)
        bmm_23 = torch.bmm(softmax_11, v)

        return bmm_23

    def replace(q, k, v):
        output = torch.empty_like(q)
        output = attention_wrapper(q, k, v, output, 1.0)
        return output

    subgraph_rewriter.replace_pattern(gm, pattern, replace)