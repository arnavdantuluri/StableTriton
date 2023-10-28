import triton
import triton.language as tl
import torch
import torch.nn as nn

def dtype(input):
    if input == torch.float32:
        return tl.float32
    elif input == torch.float16:
        return tl.float16
    elif input == torch.bfloat16:
        return tl.bfloat16
    elif input == torch.int64:
        return tl.int64
    else:
        raise ValueError(f"Unable to convert the given input: '{input}'.")


@triton.jit
def silu(input):
    return input * tl.sigmoid(input)


@triton.jit
def forward(
    output_ptr: tl.tensor,
    input_ptr: tl.tensor,
    rstd_ptr: tl.tensor,
    mean_ptr: tl.tensor,
    group_size,
    y_size,
    x_size,
    num_groups,
    weight_ptr: tl.tensor,
    bias_ptr: tl.tensor,
    eps,
    dtype: tl.constexpr,
    group_block_size: tl.constexpr,
    x_block_size: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    pid = tl.program_id(0)
    batch = pid // num_groups
    group = pid % num_groups
    num_elements = group_size * x_size
    batch_offset = batch * num_groups * num_elements
    group_offset = batch_offset + group * num_elements
    output_block_ptr = tl.make_block_ptr(
        output_ptr + group_offset,
        shape=(group_size, x_size),
        strides=(x_size, 1),
        offsets=(0, 0),
        block_shape=(group_block_size, x_block_size),
        order=(1, 0),
    )
    input_block_ptr = tl.make_block_ptr(
        input_ptr + group_offset,
        shape=(group_size, x_size),
        strides=(x_size, 1),
        offsets=(0, 0),
        block_shape=(group_block_size, x_block_size),
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


    input = tl.load(input_block_ptr)
    mean = tl.sum(tl.view(input / num_elements, (1, group_block_size * x_block_size)), 1)
    centered_mean = input - mean

    var = tl.sum(tl.view(centered_mean * centered_mean / num_elements, (1, group_block_size * x_block_size)), 1)
    rstd = tl.math.rsqrt(var + eps)
    output = centered_mean * rstd

    if weight_ptr is not None:
        weight_block_ptr = tl.make_block_ptr(
            weight_ptr,
            shape=(y_size, 1),
            strides=(1, y_size),
            offsets=(group * group_size, 0),
            block_shape=(group_block_size, 1),
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
            block_shape=(group_block_size, 1),
            order=(0, 1),
        )
        bias = tl.load(bias_block_ptr, boundary_check=(0,))
        output += bias
    if ACTIVATION:
        output = silu(output)
    
    tl.store(output_block_ptr, output.to(dtype))

    tl.store(rstd_block_ptr, rstd.to(dtype))
    tl.store(mean_block_ptr, mean.to(dtype))

map_dtype = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.int32: tl.int32,
    torch.int16: tl.int16,
}

def groupnorm_wrapper(
        input: torch.Tensor, num_groups: torch.int, weight: torch.Tensor, bias: torch.Tensor, eps: torch.float, activation: bool = False,
    ):
        factory_kwargs = {"device": input.device, "dtype": input.dtype}
        num_batches, _, y_size, x_size = input.shape
        output = torch.zeros_like(input)
        rstd = torch.empty((num_batches, num_groups), **factory_kwargs)
        mean = torch.empty((num_batches, num_groups), **factory_kwargs)

        def grid(meta):
            return (num_batches * num_groups,)

        forward[grid](
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
            map_dtype[input.dtype],
            triton.next_power_of_2(y_size // num_groups),
            triton.next_power_of_2(x_size),
            ACTIVATION=activation,
        )

        return output

if __name__ == "__main__":
    input = torch.randn(1, 128, 32).cuda()
    # Separate 6 channels into 3 groups
    m = nn.GroupNorm(32, 128).cuda()
    output_triton = groupnorm_wrapper(input, 32, m.weight, m.bias, m.eps)
    output = m(input)
    assert ((output_triton - output).abs() < 1e-3).all(), "Outputs don't match"