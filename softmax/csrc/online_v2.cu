#include "utils.h"
#include <cooperative_groups.h>
#include <math.h>

namespace cg = cooperative_groups;

__device__ __forceinline__ void onlineMergeV2(float &m1, float &s1, float m2,
                                              float s2) {
  float M = fmaxf(m1, m2);
  s1 = s1 * __expf(m1 - M) + s2 * __expf(m2 - M);
  m1 = M;
}

__device__ __forceinline__ void warpReduceOnlineV2(float &max_val,
                                                   float &sum_val) {
  auto warp = cg::tiled_partition<32>(cg::this_thread_block());
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
    float other_max = warp.shfl_xor(max_val, mask);
    float other_sum = warp.shfl_xor(sum_val, mask);
    onlineMergeV2(max_val, sum_val, other_max, other_sum);
  }
}

template <int BS>
__device__ __forceinline__ void
blockReduceOnlineV2(float &max_val, float &sum_val, float *scratch_max,
                    float *scratch_sum) {
  constexpr int n_warps = BS / WARP_SIZE;
  const int lane = threadIdx.x % WARP_SIZE;
  const int wid = threadIdx.x / WARP_SIZE;

  auto block = cg::this_thread_block();

  warpReduceOnlineV2(max_val, sum_val);
  if (lane == 0) {
    scratch_max[wid] = max_val;
    scratch_sum[wid] = sum_val;
  }
  block.sync();

  if (threadIdx.x < n_warps) {
    max_val = scratch_max[threadIdx.x];
    sum_val = scratch_sum[threadIdx.x];
  } else {
    max_val = -INFINITY;
    sum_val = 0.0f;
  }
  if (wid == 0)
    warpReduceOnlineV2(max_val, sum_val);
  if (threadIdx.x == 0) {
    scratch_max[0] = max_val;
    scratch_sum[0] = sum_val;
  }
  block.sync();
  max_val = scratch_max[0];
  sum_val = scratch_sum[0];
}

template <int NP, int BS>
__global__ __launch_bounds__(BS, 1) void softmax_v2_single_kernel(
    const float *__restrict__ in, float *__restrict__ out, int M) {
  constexpr int N = NP * BS * 4;
  constexpr int n_warps = BS / WARP_SIZE;

  extern __shared__ float smem[];
  float *smem_max = smem;
  float *smem_sum = smem + n_warps;

  const int tid = threadIdx.x;
  const int row = blockIdx.x;
  if (row >= M)
    return;

  const float4 *row_in = reinterpret_cast<const float4 *>(in + row * N);
  float4 *row_out = reinterpret_cast<float4 *>(out + row * N);

  float4 buf[NP];
  float local_max = -INFINITY;
  float local_sum = 0.0f;

#pragma unroll
  for (int i = 0; i < NP; i++) {
    buf[i] = row_in[tid + i * BS];
    float vals[4] = {buf[i].x, buf[i].y, buf[i].z, buf[i].w};
#pragma unroll
    for (int j = 0; j < 4; j++) {
      float new_max = fmaxf(local_max, vals[j]);
      local_sum =
          local_sum * __expf(local_max - new_max) + __expf(vals[j] - new_max);
      local_max = new_max;
    }
  }

  blockReduceOnlineV2<BS>(local_max, local_sum, smem_max, smem_sum);
  float row_max = local_max;

  float invSum = __frcp_rn(local_sum);

#pragma unroll
  for (int i = 0; i < NP; i++) {
    buf[i].x = __expf(buf[i].x - row_max) * invSum;
    buf[i].y = __expf(buf[i].y - row_max) * invSum;
    buf[i].z = __expf(buf[i].z - row_max) * invSum;
    buf[i].w = __expf(buf[i].w - row_max) * invSum;
    row_out[tid + i * BS] = buf[i];
  }
}

// get max and sum per tile
template <int NP, int BS>
__global__ __launch_bounds__(BS) void softmax_v2_stats_kernel(
    const float *__restrict__ in, float *__restrict__ partial_max,
    float *__restrict__ partial_sum, int M, int N, int splits) {
  constexpr int n_warps = BS / WARP_SIZE;
  const int chunk_f4 = NP * BS;

  extern __shared__ float smem[];
  float *smem_max = smem;
  float *smem_sum = smem + n_warps;

  const int tid = threadIdx.x;
  const int row = blockIdx.x / splits;
  const int split_id = blockIdx.x % splits;
  if (row >= M)
    return;

  const int f4_offset = split_id * chunk_f4;
  const float4 *chunk_in =
      reinterpret_cast<const float4 *>(in + row * N) + f4_offset;

  float local_max = -INFINITY;
  float local_sum = 0.0f;
#pragma unroll
  for (int i = 0; i < NP; i++) {
    float4 v = chunk_in[tid + i * BS];
    float vals[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int j = 0; j < 4; j++) {
      float new_max = fmaxf(local_max, vals[j]);
      local_sum =
          local_sum * __expf(local_max - new_max) + __expf(vals[j] - new_max);
      local_max = new_max;
    }
  }

  blockReduceOnlineV2<BS>(local_max, local_sum, smem_max, smem_sum);

  if (threadIdx.x == 0) {
    int idx = row * splits + split_id;
    partial_max[idx] = local_max;
    partial_sum[idx] = local_sum;
  }
}

template <int NP, int BS>
__global__ __launch_bounds__(BS) void softmax_v2_normalize_kernel(
    const float *__restrict__ in, float *__restrict__ out,
    const float *__restrict__ partial_max,
    const float *__restrict__ partial_sum, int M, int N, int splits) {
  const int row = blockIdx.x / splits;
  const int split_id = blockIdx.x % splits;
  if (row >= M)
    return;
  const int tid = threadIdx.x;

  float global_max = -INFINITY;
  float global_sum = 0.0f;
  for (int s = 0; s < splits; s++) {
    int idx = row * splits + s;
    onlineMergeV2(global_max, global_sum, partial_max[idx], partial_sum[idx]);
  }
  float invSum = __frcp_rn(global_sum);

  const int chunk_f4 = NP * BS;
  const int f4_offset = split_id * chunk_f4;
  const float4 *chunk_in =
      reinterpret_cast<const float4 *>(in + row * N) + f4_offset;
  float4 *chunk_out = reinterpret_cast<float4 *>(out + row * N) + f4_offset;

#pragma unroll
  for (int i = 0; i < NP; i++) {
    float4 v = __ldg(chunk_in + tid + i * BS);
    v.x = __expf(v.x - global_max) * invSum;
    v.y = __expf(v.y - global_max) * invSum;
    v.z = __expf(v.z - global_max) * invSum;
    v.w = __expf(v.w - global_max) * invSum;
    chunk_out[tid + i * BS] = v;
  }
}

void launch_softmax_online_v2(const float *d_in, float *d_out, int M, int N) {
#define LAUNCH_SINGLE(n_, bs_)                                                 \
  case n_: {                                                                   \
    constexpr int np_ = (n_) / ((bs_) * 4);                                    \
    const size_t smem_ = 2 * ((bs_) / WARP_SIZE) * sizeof(float);              \
    softmax_v2_single_kernel<np_, bs_><<<M, bs_, smem_>>>(d_in, d_out, M);     \
    break;                                                                     \
  }

  if (N <= ONLINE_V2_SINGLE_MAX_N) {
    switch (N) {
      ONLINE_V2_SINGLE_SIZES(LAUNCH_SINGLE)
    default:
      fprintf(stderr, "softmax_online_v2: N=%d not supported\n", N);
      abort();
    }
#undef LAUNCH_SINGLE
    return;
  }

  // Multi-block path
  constexpr int BS = V2_MULTI_BS;
  const int target_chunk = V2_TARGET_NP * BS * 4;
  int splits = (N + target_chunk - 1) / target_chunk;
  if (splits < 2)
    splits = 2;
  while (N % (splits * BS * 4) != 0 && splits < 64)
    splits++;

  const int chunk_elems = N / splits;
  const int np = chunk_elems / (BS * 4);

  float *partial_max = nullptr;
  float *partial_sum = nullptr;
  cudaMalloc(&partial_max, (size_t)M * splits * sizeof(float));
  cudaMalloc(&partial_sum, (size_t)M * splits * sizeof(float));
  //   For production code, these should be pre-allocated in a memory pool
  //   to avoid the ~10 μs overhead of cudaMalloc per kernel invocation.

  constexpr int n_warps = BS / WARP_SIZE;
  const size_t smem = 2 * n_warps * sizeof(float);

#define LAUNCH_STATS(np_val)                                                   \
  case np_val:                                                                 \
    softmax_v2_stats_kernel<np_val, BS><<<M * splits, BS, smem>>>(             \
        d_in, partial_max, partial_sum, M, N, splits);                         \
    break;

#define LAUNCH_NORMALIZE(np_val)                                               \
  case np_val:                                                                 \
    softmax_v2_normalize_kernel<np_val, BS><<<M * splits, BS, 0>>>(            \
        d_in, d_out, partial_max, partial_sum, M, N, splits);                  \
    break;

  switch (np) {
    LAUNCH_STATS(1)
    LAUNCH_STATS(2)
    LAUNCH_STATS(4)
    LAUNCH_STATS(8)
    LAUNCH_STATS(12)
    LAUNCH_STATS(16)
  default:
    fprintf(stderr,
            "softmax_online_v2: np=%d not supported (N=%d, splits=%d)\n", np, N,
            splits);
    cudaFree(partial_max);
    cudaFree(partial_sum);
    abort();
  }
#undef LAUNCH_STATS

  switch (np) {
    LAUNCH_NORMALIZE(1)
    LAUNCH_NORMALIZE(2)
    LAUNCH_NORMALIZE(4)
    LAUNCH_NORMALIZE(8)
    LAUNCH_NORMALIZE(12)
    LAUNCH_NORMALIZE(16)
  default:
    fprintf(stderr,
            "softmax_online_v2: np=%d not supported for normalize (N=%d, "
            "splits=%d)\n",
            np, N, splits);
    cudaFree(partial_max);
    cudaFree(partial_sum);
    abort();
  }
#undef LAUNCH_NORMALIZE

  cudaFree(partial_max);
  cudaFree(partial_sum);
}

#ifdef SOFTMAX_STANDALONE

int main() {
  const int M = BENCH_DEFAULT_M;
  const int N = 131072;
  float *h_in = nullptr, *h_out = nullptr;
  float *d_in = nullptr, *d_out = nullptr;

  bench_alloc(&h_in, &h_out, &d_in, &d_out, M, N);

  for (int i = 0; i < 5; i++) {
    launch_softmax_online_v2(d_in, d_out, M, N);
  }
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  float ms;
  bench_timing_begin(&start, &stop);
  launch_softmax_online_v2(d_in, d_out, M, N);
  bench_timing_end(start, stop, &ms);

  bench_copy_back(h_out, d_out, M, N);
  printf("kernel time: %f ms\n", ms);
  bench_free(h_in, h_out, d_in, d_out);
  return 0;
}

#endif /* SOFTMAX_STANDALONE */
