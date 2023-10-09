import torch

import triton
import triton.language as tl
from torch._inductor.triton_ops.autotune import grid
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream
from torch._inductor.utils import print_performance

import triton
import triton.language as tl
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 921600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 960
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
    tmp10 = 960.0
    tmp11 = tmp9 / tmp10
    tmp12 = tl.exp(tmp11)
    tmp13 = tmp0 * tmp12
    tmp14 = tl.sin(tmp13)
    tmp15 = tl.cos(tmp13)
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp14, xmask)
    tl.store(out_ptr1 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp15, xmask)

def call(x):
    xnumel = x.numel()
    buf0 = torch.empty_like(x)
    buf1 = torch.empty_like(x)
    stream0 = get_cuda_stream(0)
    triton_[grid(xnumel)](x, buf0, buf1, xnumel, stream=stream0)
    del x
    return (buf0, buf1, )


if __name__ == "__main__":
    x = torch.rand((1, 1, 960, 960), device='cuda', dtype=torch.float32)
    # call([arg0_1])
    print_performance(lambda: call(x))