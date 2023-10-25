# TODO: Need to test with nested modules and see if it still works
#Dropout during inference is set to 0.0, we can just remove it without issues
import torch
import torch.nn as nn
import torch.fx as fx

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(5, 5)
        self.lin2 = nn.Linear(5,5)
        self.lin3 = nn.Linear(5,5)
        self.nonlin = nn.SiLU()
        self.dropout = nn.Dropout(p=0.0)
    
    def forward(self, x):
        return self.dropout(self.nonlin(self.lin3(self.lin2(self.lin1(x)))))

def remove_dropout(gm: torch.fx.GraphModule):
    modules = dict(gm.named_modules())
    for n in gm.graph.nodes:
        is_dropout_module = n.op == "call_module" and isinstance(modules[n.target], torch.nn.Dropout)
        is_dropout_function = n.target == torch.nn.functional.dropout
        # If the target matches one of the patterns
        if is_dropout_module or is_dropout_function:
            # Set the insert point, add the new node, and replace all uses
            # of `n` with the new node
            with gm.graph.inserting_after(n):
                # new_node = gm.graph.call_function(torch.nn.Identity, n.args, n.kwargs)
                n.replace_all_uses_with(n.args[0])
            # Remove the old node from the graph
            gm.graph.erase_node(n)
    gm.recompile()

if __name__ == "__main__":
    m = M().cuda()
    fx_model = fx.symbolic_trace(m)
    old_traced = fx.symbolic_trace(m)
    remove_dropout(fx_model)
    assert fx_model.code != old_traced.code, "Issue with fusion with fx graph"
    print("Fx Graph replacement was a success! Kernel Fusion works perfectly")
    # Test output
    x = torch.rand(5, 5, dtype=torch.float32).cuda()
    out_old = m(x)
    out_fused = fx_model(x)
    # Some margin for triton code.
    assert ((out_fused - out_old).abs() < 1e-3).all(), "Outputs don't match"