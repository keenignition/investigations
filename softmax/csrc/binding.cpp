#include <torch/extension.h>

torch::Tensor softmax_naive(torch::Tensor x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("softmax_naive", &softmax_naive, "Softmax naive kernel (one block per row)");
}
