import sys
import torch
import softmax_kernel
M, N = int(sys.argv[1]), int(sys.argv[2])
x = torch.randn((M, N), device="cuda")
# warmup
for _ in range(5): softmax_kernel.softmax_online_v2(x)
torch.cuda.synchronize()
# timed
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(20): softmax_kernel.softmax_online_v2(x)
end.record()
torch.cuda.synchronize()
ms = start.elapsed_time(end) / 20
gbps = 2 * x.numel() * x.element_size() / (ms * 1e-3) / 1e9
print(gbps)
