# Copyright 2023 ⓒ Kakao Brain Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import triton
import triton.language as tl
import torch

@triton.jit
def silu(input):
    return 1 / (1 + tl.math.fast_expf(-input.to(tl.float32)))

@triton.jit
def group_norm(
    output_ptr: tl.tensor,
    input_ptr: tl.tensor,
    rstd_ptr: tl.tensor,
    mean_ptr: tl.tensor,
    y_size: tl.int32,
    x_size: tl.int32,
    num_groups: tl.int32,
    weight_ptr: tl.tensor,
    bias_ptr: tl.tensor,
    eps: tl.float32,
    dtype: tl.constexpr,
    activation: tl.constexpr,
    y_block_size: tl.constexpr,
    x_block_size: tl.constexpr,
):
    group_size = y_size // num_groups
    pid = tl.program_id(0)
    batch = pid // num_groups
    group = pid % num_groups
    batch_offset = batch * y_size * x_size
    num_elements = group_size * x_size
    group_offset = batch_offset + group * num_elements
    output_block_ptr = tl.make_block_ptr(
        output_ptr + group_offset,
        shape=(group_size, x_size),
        strides=(x_size, 1),
        offsets=(0, 0),
        block_shape=(y_block_size, x_block_size),
        order=(1, 0),
    )
    input_block_ptr = tl.make_block_ptr(
        input_ptr + group_offset,
        shape=(group_size, x_size),
        strides=(x_size, 1),
        offsets=(0, 0),
        block_shape=(y_block_size, x_block_size),
        order=(1, 0),
    )
    rstd_block_ptr = tl.make_block_ptr(
        rstd_ptr + batch * num_groups,
        shape=(group_size,),
        strides=(1,),
        offsets=(group,),
        block_shape=(1,),
        order=(0,),
    )
    mean_block_ptr = tl.make_block_ptr(
        mean_ptr + batch * num_groups,
        shape=(group_size,),
        strides=(1,),
        offsets=(group,),
        block_shape=(1,),
        order=(0,),
    )
    input = tl.load(input_block_ptr, boundary_check=(0, 1), padding_option="zero")
    mean = tl.sum(tl.view(input / num_elements, (1, y_block_size * x_block_size)), 1)
    y_condition = tl.arange(0, y_block_size) < group_size
    x_condition = tl.arange(0, x_block_size) < x_size
    condition = y_condition[:, None] & x_condition[None, :]
    centered_mean = tl.where(condition, input - mean, 0)
    var = tl.sum(tl.view(centered_mean * centered_mean / num_elements, (1, y_block_size * x_block_size)), 1)
    rstd = tl.math.rsqrt(var + eps)
    output = centered_mean * rstd

    if weight_ptr is not None:
        weight_block_ptr = tl.make_block_ptr(
            weight_ptr,
            shape=(y_size, 1),
            strides=(1, y_size),
            offsets=(group * group_size, 0),
            block_shape=(y_block_size, 1),
            order=(0, 1),
        )
        weight = tl.load(weight_block_ptr, boundary_check=(0,))
        output *= weight

    if bias_ptr is not None:
        bias_block_ptr = tl.make_block_ptr(
            bias_ptr,
            shape=(y_size, 1),
            strides=(1, y_size),
            offsets=(group * group_size, 0),
            block_shape=(y_block_size, 1),
            order=(0, 1),
        )
        bias = tl.load(bias_block_ptr, boundary_check=(0,))
        output += bias

    output = silu(output)
    tl.store(output_block_ptr, output.to(dtype), boundary_check=(0, 1))
    tl.store(rstd_block_ptr, rstd.to(dtype))
    tl.store(mean_block_ptr, mean.to(dtype))

def groupnorm_wrapper(
        input: torch.Tensor, num_groups: torch.int, weight: torch.Tensor, bias: torch.Tensor, eps: torch.float
    ):
        factory_kwargs = {"device": input.device, "dtype": input.dtype}
        num_batches, y_size, x_size = input.shape
        output = torch.zeros_like(input)
        rstd = torch.empty((num_batches, num_groups), **factory_kwargs)
        mean = torch.empty((num_batches, num_groups), **factory_kwargs)

        def grid(meta):
            return (num_batches * num_groups,)

        group_norm[grid](
            output,
            input,
            rstd,
            mean,
            y_size // num_groups,
            y_size,
            x_size,
            num_groups,
            weight,
            bias,
            eps,
            input.dtype,
            triton.next_power_of_2(y_size // num_groups),
            triton.next_power_of_2(x_size),
        )

        return output, rstd, mean