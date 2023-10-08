import triton
import triton.language as tl
from typing import Callable, List
import torch
from linear import kernel_fma

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
def gelu(input: tl.tensor):
        return 0.5 * input * (1 + tl.math.tanh(input * 0.7978845608028654 * (1 + 0.044715 * input * input)))

def num_warps_and_stages_for_geglu(size):
    if size >= 2**15:
        num_warps = 8
        num_stages = 3
    elif size >= 2**14:
        num_warps = 4
        num_stages = 4
    else:
        num_warps = 2
        num_stages = 5
    return num_warps, num_stages


def geglu_configs():
    configs = []
    for k_block_size in [32, 64]:
        for m_block_size in [16, 64, 128]:
            for x_block_size in [32, 64, 128]:
                num_warps, num_stages = num_warps_and_stages_for_geglu(m_block_size * x_block_size)
                config = triton.Config(
                    {
                        "m_block_size": m_block_size,
                        "k_block_size": k_block_size,
                        "x_block_size": x_block_size,
                    },
                    num_warps=num_warps,
                    num_stages=num_stages,
                )
                configs.append(config)
    return configs

def autotune(
    configs: List[triton.Config],
    key: List[str],
    prune_configs_by: Callable = None,
    reset_to_zero: List[str] = None,
    warmup: int = 25,
    rep: int = 100,
):
    return triton.autotune(
        configs if True else [configs[0]], key, prune_configs_by, reset_to_zero, warmup, rep
    )



@autotune(geglu_configs(), ["m_size", "k_size", "x_size"])
@triton.jit
def geglu(
    output_ptr: tl.tensor,
    state_ptr: tl.tensor,
    gate_ptr: tl.tensor,
    input_ptr: tl.tensor,
    weight_ptr: tl.tensor,
    bias_ptr: tl.tensor,
    m_size: tl.int32,
    n_size: tl.int32,
    k_size: tl.int32,
    x_size: tl.int32,
    input_batch_stride: tl.int32,
    input_m_stride: tl.int32,
    input_k_stride: tl.int32,
    weight_n_stride: tl.int32,
    weight_k_stride: tl.int32,
    use_accelerator: tl.constexpr,
    dtype: tl.constexpr,
    m_block_size: tl.constexpr,
    k_block_size: tl.constexpr,
    x_block_size: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m_blocks = tl.cdiv(m_size, m_block_size)
    num_x_blocks = tl.cdiv(x_size, x_block_size)
    num_blocks = num_m_blocks * num_x_blocks
    batch = pid // num_blocks
    block = pid % num_blocks
    m_block = block // num_x_blocks
    x_block = block % num_x_blocks
    m_offset = m_block * m_block_size
    x_offset = x_block * x_block_size
    output_block_ptr = tl.make_block_ptr(
            output_ptr + batch * m_size * x_size,
            shape=(m_size, x_size),
            strides=(x_size, 1),
            offsets=(m_offset, x_offset),
            block_shape=(m_block_size, x_block_size),
            order=(1, 0),
        )
    # Gate Projection
    kernel_fma(
        gate_ptr,
        input_ptr,
        weight_ptr,  # data ptrs
        bias_ptr,  # auto skip bias if not present
        dtype,
        m_size,  # shapes
        n_size,
        k_size,
        m_size // 32,  # key for triton cache (limit number of compilations)
        n_size // 32,
        k_size // 32,
        output_m_stride=gate_ptr.stride(0),  # strides
        output_n_stride=gate_ptr.stride(1),
        a_m_stride=input_m_stride,
        a_k_stride=input_k_stride,
        b_n_stride=weight_n_stride,
        b_k_stride=weight_k_stride,
        HAS_BIAS=bias_ptr is not None,  # optional fused bias
        ACTIVATION=False,
        BLOCK_M=m_block_size,
        BLOCK_N=k_block_size,
        BLOCK_K=x_block_size,
        GROUP_M=8,  # speed optimization: group the programs
    ) 
    # State Projection
    kernel_fma(
        state_ptr,
        input_ptr,
        weight_ptr,  # data ptrs
        bias_ptr,  # auto skip bias if not present
        dtype,
        m_size,  # shapes
        n_size,
        k_size,
        m_size // 32,  # key for triton cache (limit number of compilations)
        n_size // 32,
        k_size // 32,
        output_m_stride=state_ptr.stride(0),  # strides
        output_n_stride=state_ptr.stride(1),
        a_m_stride=input_m_stride,
        a_k_stride=input_k_stride,
        b_n_stride=weight_n_stride,
        b_k_stride=weight_k_stride,
        HAS_BIAS=bias_ptr is not None,  # optional fused bias
        ACTIVATION=False,
        BLOCK_M=m_block_size,
        BLOCK_N=k_block_size,
        BLOCK_K=x_block_size,
        GROUP_M=8,
    )
    output = state_ptr * gelu(gate_ptr)
    
    tl.store(output_block_ptr, output.to(dtype))

def geglu_wrapper(input, weight, bias, use_accelerator):
    factory_kwargs = {"device": input.device, "dtype": input.dtype}
    num_batches, m_size, k_size = input.shape
    n_size, _ = weight.shape
    x_size = n_size // 2
    output = torch.empty(num_batches, m_size, x_size, **factory_kwargs)
    state_gate = torch.empty(num_batches, m_size, n_size, **factory_kwargs)

    def grid(meta):
        num_m_blocks = triton.cdiv(m_size, meta["m_block_size"])
        num_x_blocks = triton.cdiv(x_size, meta["x_block_size"])
        return (num_batches * num_m_blocks * num_x_blocks,)

    geglu[grid](
        output,
        state_gate,
        input,
        weight,
        bias,
        m_size,
        n_size,
        k_size,
        x_size,
        input.stride(0),
        input.stride(1),
        input.stride(2),
        weight.stride(0),
        weight.stride(1),
        use_accelerator,
        dtype(input.dtype),
    )