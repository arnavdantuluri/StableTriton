import torch
import torch._dynamo as torchdynamo

from stabletriton.optimizers import remove_dropout, fuse_attention, fuse_geglu, \
                                    replace_group_norm,replace_layer_norm, \
                                    replace_linear, fuse_timesteps, \
                                    replace_linear_activ, replace_group_norm_activation, \
                                    make_dynamic_graphed_callable, cuda_graphs_wrapper, static_inputs_pool

def replace_backend(gm: torch.fx.GraphModule):
    remove_dropout(gm)
    fuse_attention(gm)
    fuse_geglu(gm)
    replace_linear_activ(gm, torch.nn.SiLU())
    replace_group_norm_activation(gm, torch.nn.SiLU())
    replace_group_norm(gm)
    replace_layer_norm(gm)
    # Replacing Linear makes it slower??
    # Drops from 8.38 to 6.61 (Still faster than vanilla pytorch tho :) )
    # replace_linear(gm)
    fuse_timesteps(gm)
    return gm

def run_compiler(gm: torch.fx.GraphModule):
    return replace_backend(gm)

def optimize_model(model: torch.nn.Module, cuda_graph: bool) -> None:
    # Assertions to make sure Stable Triton is compatible
    assert torch.cuda.is_available(), "CUDA capacity is required to use Stable Triton"
    major, _ = torch.cuda.get_device_capability()
    if major < 8:
        raise RuntimeError("GPU compute capability 8.0 (Ampere) or higher is required to use Stable Triton")
    assert next(model.parameters()).device.type == "cuda", "Model must be on GPU"

    model = replace_backend(torch.fx.symbolic_trace(model))
    if cuda_graph:
        model.forward = make_dynamic_graphed_callable(model.forward)
    return model