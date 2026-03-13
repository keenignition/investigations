#include <torch/extension.h>
#include "utils.h"

static void launch_softmax_naive(const float* in, float* out, int M, int N) {
  const int threads = 256;
  const size_t smem = threads * sizeof(float);
  softmax_kernel<<<M, threads, smem>>>(in, out, M, N);
}

torch::Tensor softmax_naive(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
  TORCH_CHECK(x.dim() == 2, "x must be 2D (M, N)");

  auto out = torch::empty_like(x);
  const int M = x.size(0);
  const int N = x.size(1);
  launch_softmax_naive(x.data_ptr<float>(), out.data_ptr<float>(), M, N);
  return out;
}

// ---

static void launch_softmax_wr(const float* in, float* out, int M, int N) {
  const int threads = 256;
  const size_t smem = (threads / 32) * sizeof(float);
  softmax_kernel_wr<<<M, threads, smem>>>(in, out, M, N);
}

torch::Tensor softmax_wr(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
  TORCH_CHECK(x.dim() == 2, "x must be 2D (M, N)");

  auto out = torch::empty_like(x);
  const int M = x.size(0);
  const int N = x.size(1);
  launch_softmax_wr(x.data_ptr<float>(), out.data_ptr<float>(), M, N);
  return out;
}

// ---
