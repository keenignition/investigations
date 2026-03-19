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

// Register-based fused kernel is only fast for N <= 4096 (avoids register spill).
// For larger N we use the block-reduction kernel (shared-mem or no-cache) for better performance.
constexpr int FUSED_MAX_N_REGISTER_KERNEL = 4096;

static void launch_softmax_fused_binding(const float* in, float* out, int M, int N) {
  TORCH_CHECK(N >= 1024 && (N & (N - 1)) == 0,
              "softmax_fused requires N a power of 2 >= 1024, got N=", N);
  // if (N > FUSED_MAX_N_REGISTER_KERNEL) {
  //   launch_softmax_block(in, out, M, N);
  // } else {
  //   launch_softmax_fused(in, out, M, N);
  // }
  launch_softmax_fused(in, out, M, N);
}

// static void launch_softmax_block_binding(const float* in, float* out, int M, int N) {
//   launch_softmax_block(in, out, M, N);
// }

torch::Tensor softmax_fused(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
  TORCH_CHECK(x.dim() == 2, "x must be 2D (M, N)");

  auto out = torch::empty_like(x);
  const int M = x.size(0);
  const int N = x.size(1);
  launch_softmax_fused_binding(x.data_ptr<float>(), out.data_ptr<float>(), M, N);
  return out;
}

// ---
