from typing import List

import torch
import torch._dynamo as torchdynamo
from transformers import PreTrainedModel

from torch.cuda import make_graphed_callables
from stabletriton.optimizers.cuda.cuda_graph import cuda_graphs_wrapper, static_inputs_pool
from stabletriton.optimizers.cuda.graphs import simple_make_graphed_callable
from stabletriton.optimizers import remove_dropout, fuse_attention, fuse_geglu, \
                                                            fuse_groupnorm_activ, replace_group_norm,\
                                                            replace_layer_norm, fuse_linear_activ, replace_linear, fuse_timesteps

# Test first withouth groupnorm and linear fused kernels
def replace_backend(gm: torch.fx.GraphModule):
    remove_dropout(gm)
    fuse_attention(gm)
    fuse_geglu(gm)
    fuse_groupnorm_activ(gm)
    fuse_linear_activ(gm)
    replace_group_norm(gm)
    replace_layer_norm(gm)
    replace_linear(gm)
    fuse_timesteps(gm)

def _compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    replace_backend(gm)
    return gm
    # return simple_make_graphed_callable(gm, example_inputs)


def optimize_model(model) -> None:
    assert torch.cuda.is_available(), "CUDA is needed to use StableTriton"
    major, _ = torch.cuda.get_device_capability()
    if major < 8:
        raise RuntimeError("GPU compute capability 8.0 (Ampere) or higher is required (for now)")
    assert next(model.parameters()).device.type == "cuda", "Model must be on GPU"
    static_inputs_pool.clear()
    model.forward_original = model.forward

    @torchdynamo.optimize(_compiler)
    def run(*args, **kwargs):
        return model.forward_original(*args, **kwargs)

    model.forward = run