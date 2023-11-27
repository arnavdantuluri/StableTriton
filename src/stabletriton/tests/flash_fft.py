''' FN = P(IN2 ⊗ FN1 )DP^{-1}(IN1 ⊗FN2 )P
'''
import torch
import math

def fft_matrix(N):
    n = torch.arange(N)
    k = n.view(-1, 1)
    M = torch.exp(-2j * torch.pi * n * k / N)
    return M

def compute_twiddle_factors_fft(n, m):
    """Compute the twiddle factors of size n x m"""
    n_a = torch.arange(n).view(-1, 1)
    m_a = torch.arange(m)
    N = n * m
    M = torch.exp(-2j * torch.pi * n_a * m_a / N)
    return M

def ifft_matrix(N):
    n = torch.arange(N)
    k = n.view(-1, 1)
    M = torch.exp(2j * torch.pi * n * k / N)
    return M

def compute_twiddle_factors_ifft(n, m):
    """Compute the twiddle factors of size n x m"""
    n_a = torch.arange(n).view(-1, 1)
    m_a = torch.arange(m)
    N = n * m
    M = torch.exp(2j * torch.pi * n_a * m_a / N)
    return M

'''X ← ((F.T @ X) * t)F ▷ FFT, decomposed into two steps
X ← X * Kf.T ▷ Elementwise multiply with kf
Y ← ((XF^-1).T * tinv )F^-1 ▷ Inverse FFT, decomposed into two steps'''
def compute_fft(input, kernel, fft, fft_inv, twiddle, twiddle_inv):
    xs = torch.chunk(input, 32, dim=2)
    kernels = torch.chunk(kernel, 32, dim=1)
    outputs = []
    # To be parallelized across SMs with triton?
    for i, x in enumerate(xs):
        output = torch.matmul(torch.matmul(fft.T, x) * twiddle, fft)
        output = output * kernels[i]
        output = torch.matmul(torch.matmul(fft_inv.T, x) * twiddle_inv, fft_inv)
        outputs.append(output)
    return torch.stack(outputs, dim=1)

seqlen = 1024
dtype = torch.float16

N = seqlen
sqrt_N = int(math.sqrt(seqlen))

input = torch.rand(2, 32, seqlen).to(dtype).cuda()
B, H, L = input.shape
k_f = torch.rand((H, N), dtype=torch.cfloat).cuda()

f_sqrt_N_fft = torch.rand(sqrt_N, sqrt_N, 2).cuda().half()#torch.view_as_real(fft_matrix(sqrt_N)).to(dtype).cuda()
f_sqrt_N_ifft = torch.rand(sqrt_N, sqrt_N, 2).cuda().half()#torch.view_as_real(ifft_matrix(sqrt_N)).to(dtype).cuda()

twiddle_factors_fft = torch.rand(sqrt_N).cuda().half()#torch.view_as_real(compute_twiddle_factors_fft(sqrt_N, sqrt_N) / N).to(dtype).cuda()
twiddle_factors_ifft = torch.rand(sqrt_N).cuda().half()#torch.view_as_real(compute_twiddle_factors_ifft(sqrt_N, sqrt_N)).to(dtype).cuda()

k_f_permuted = torch.view_as_real(k_f.reshape(H, sqrt_N, sqrt_N).transpose(-1, -2).reshape(H, N).to(dtype).contiguous())

output = compute_fft(input, k_f_permuted, f_sqrt_N_fft, f_sqrt_N_ifft, twiddle_factors_fft, twiddle_factors_ifft)