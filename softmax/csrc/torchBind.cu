#include "utils.h"
#include <torch/extension.h>

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

// Warp-fused: N in [FUSED_WARP_MIN_N, FUSED_WARP_MAX_N], 32 threads/row.
torch::Tensor softmax_fused_warp(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
  TORCH_CHECK(x.dim() == 2, "x must be 2D (M, N)");
  const int N = x.size(1);
  TORCH_CHECK(N >= FUSED_WARP_MIN_N && N <= FUSED_WARP_MAX_N && (N & (N - 1)) == 0,
              "softmax_fused_warp: N must be pow2 in [", FUSED_WARP_MIN_N, ",",
              FUSED_WARP_MAX_N, "], got N=", N);
  auto out = torch::empty_like(x);
  launch_softmax_fused_warp(x.data_ptr<float>(), out.data_ptr<float>(), x.size(0), N);
  return out;
}

// Block-fused: N in [FUSED_BLOCK_MIN_N, FUSED_BLOCK_MAX_N], 1024 threads/row.
torch::Tensor softmax_fused_block(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
  TORCH_CHECK(x.dim() == 2, "x must be 2D (M, N)");
  const int N = x.size(1);
  TORCH_CHECK(N >= FUSED_BLOCK_MIN_N && N <= FUSED_BLOCK_MAX_N && (N & (N - 1)) == 0,
              "softmax_fused_block: N must be pow2 in [", FUSED_BLOCK_MIN_N, ",",
              FUSED_BLOCK_MAX_N, "], got N=", N);
  auto out = torch::empty_like(x);
  launch_softmax_fused_block(x.data_ptr<float>(), out.data_ptr<float>(), x.size(0), N);
  return out;
}

// ---

// Online softmax: N in [ONLINE_MIN_N, ONLINE_MAX_N], fused (max,sum) reduction.
torch::Tensor softmax_online(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
  TORCH_CHECK(x.dim() == 2, "x must be 2D (M, N)");
  const int N = x.size(1);
  TORCH_CHECK(N >= ONLINE_MIN_N && N <= ONLINE_MAX_N && (N & (N - 1)) == 0,
              "softmax_online: N must be pow2 in [", ONLINE_MIN_N, ",",
              ONLINE_MAX_N, "], got N=", N);
  auto out = torch::empty_like(x);
  launch_softmax_online(x.data_ptr<float>(), out.data_ptr<float>(), x.size(0), N);
  return out;
}

// Online softmax V2: multi-block per row for large N, 1-pass for small N.
torch::Tensor softmax_online_v2(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
  TORCH_CHECK(x.dim() == 2, "x must be 2D (M, N)");
  const int N = x.size(1);
  TORCH_CHECK(N >= ONLINE_V2_MIN_N && N <= ONLINE_V2_MAX_N && (N & (N - 1)) == 0,
              "softmax_online_v2: N must be pow2 in [", ONLINE_V2_MIN_N, ",",
              ONLINE_V2_MAX_N, "], got N=", N);
  auto out = torch::empty_like(x);
  launch_softmax_online_v2(x.data_ptr<float>(), out.data_ptr<float>(), x.size(0), N);
  return out;
}
