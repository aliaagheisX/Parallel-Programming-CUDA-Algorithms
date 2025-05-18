import torch
from torch.utils.cpp_extension import load



my_extension = load(
    name='my_extension',
    sources=['coarsing.cu'],
    verbose=True,
    with_cuda=True,
    extra_cuda_cflags=["-O2"],
)

N = M = K = 32

A = torch.rand((N, K)).cuda()
B = torch.rand((K, M)).cuda()

C = my_extension.matrix_mult_coarsing(A, B)

corr = A @ B

print(corr.allclose(C))

print(torch.abs(C - corr).max())