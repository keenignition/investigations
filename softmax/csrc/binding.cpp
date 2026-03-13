#include <torch/extension.h>

torch::Tensor softmax_naive(torch::Tensor x);
torch::Tensor softmax_wr(torch::Tensor x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("softmax_naive", &softmax_naive, "Softmax naive kernel");
  m.def("softmax_wr", &softmax_wr, "Softmax warp reduction");
}
