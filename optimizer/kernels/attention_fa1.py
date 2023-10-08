#If gpu is detected to be pre-Ampere, FA1 is utilized. Requires flash-attn==1.0.5 to be installed.
#If you want to install w/o installing dependencies you can do it as such
#pip install flash-attn==1.0.5 --no-deps --no-dependencies
#You can try different versions if you want but 1.0.5 works best for my setup so that's what i'm using to test. Might not work with a diff version.

#Very thin wrapper for flash attention 1 cuda implementation. This should support some pre-ampere architectures such as Tesla, Volta etc.
import torch
try:
    from flash_attn.modules.mha import FlashSelfAttention
    flash_attn_available = True 
except:
    flash_attn_available = False 

def attention(q, k, v, sm_scale, sequence_parallel=False):
    assert flash_attn_available == True, "Flash Attention did not install properly. Please check whether this is a version issue or an installation issue on your end."
    flash_self_attention = FlashSelfAttention(causal = False, softmax_scale=sm_scale)
    qkv = torch.concat([q.unsqueeze(2), k.unsqueeze(2), v.unsqueeze(2)], dim = 2).half()
    attn_output = flash_self_attention(qkv)
    return attn_output