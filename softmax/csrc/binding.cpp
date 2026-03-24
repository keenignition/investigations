#include <torch/extension.h>

torch::Tensor softmax_naive(torch::Tensor x);
torch::Tensor softmax_wr(torch::Tensor x);
torch::Tensor softmax_fused_warp(torch::Tensor x);
torch::Tensor softmax_fused_block(torch::Tensor x);
torch::Tensor softmax_online(torch::Tensor x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("softmax_naive", &softmax_naive,
        "Naive kernel (one thread per element)");
  m.def("softmax_wr", &softmax_wr, "Warp reduction kernel");
  m.def("softmax_fused_warp", &softmax_fused_warp,
        "Warp-fused kernel: N in [1024,4096], row in registers");
  m.def("softmax_fused_block", &softmax_fused_block,
        "Block-fused kernel: N in [4096,65536], row in registers");
  m.def("softmax_online", &softmax_online,
        "Online softmax: N in [4096,65536], fused (max,sum) reduction (2 syncs vs 4)");
}
