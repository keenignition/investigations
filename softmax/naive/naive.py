import torch
torch.manual_seed(0)

x = torch.rand(4096, 4096, device='cuda')


# https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html
def naive_softmax(x):
    x_max = x.max(dim=1)[0]  # (M, N) -> (M, )
    z = x - x_max[:, None]  # (M, N) + (M, ) ->  (M, N)
    num = torch.exp(z)  # (M, N) -> (M, N)
    den = num.sum(dim=1)  # (M,N) -> (M, )
    ret = num / den[:, None]  # (M, N) + (M, ) -> (M, N)

    # 5 (M, N) + 2 (M, ) reads
    # 3 (M, N) + 2 (M, ) writes
    return ret


torch.cuda.synchronize()
for _ in range(1):
    y = naive_softmax(x)
torch.cuda.synchronize()
