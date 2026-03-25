#include "utils.h"
#include <math.h>

// Implementation from
// https://arxiv.org/abs/1805.02867
// https://github.com/NVIDIA/online-softmax?tab=readme-ov-file

// Merge two partial online-softmax accumulators
__device__ __forceinline__ void onlineMerge(float &m1, float &s1, float m2,
                                            float s2) {
  float M = fmaxf(m1, m2);
  s1 = s1 * __expf(m1 - M) + s2 * __expf(m2 - M);
  m1 = M;
}

// Warp-level online reduction via butterfly shuffle
__device__ __forceinline__ void warpReduceOnline(float &max_val,
                                                 float &sum_val) {
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
    float other_max = __shfl_xor_sync(0xffffffff, max_val, mask);
    float other_sum = __shfl_xor_sync(0xffffffff, sum_val, mask);
    onlineMerge(max_val, sum_val, other_max, other_sum);
  }
}

template <int BS>
__device__ __forceinline__ void
blockReduceOnline(float &max_val, float &sum_val, float *scratch_max,
                  float *scratch_sum) {
  constexpr int n_warps = BS / WARP_SIZE;
  const int lane = threadIdx.x & (WARP_SIZE - 1);
  const int wid = threadIdx.x / WARP_SIZE;

  warpReduceOnline(max_val, sum_val); // intra-warp, register-only
  if (lane == 0) {
    scratch_max[wid] = max_val; // n_warps writes
    scratch_sum[wid] = sum_val;
  }
  __syncthreads();

  if (threadIdx.x < n_warps) {
    max_val = scratch_max[threadIdx.x];
    sum_val = scratch_sum[threadIdx.x];
  } else {
    max_val = -INFINITY;
    sum_val = 0.0f;
  }
  if (wid == 0)
    warpReduceOnline(max_val, sum_val);
  if (threadIdx.x == 0) {
    scratch_max[0] = max_val;
    scratch_sum[0] = sum_val;
  }
  __syncthreads();
  max_val = scratch_max[0];
  sum_val = scratch_sum[0];
}

// Template on NP (float4s per thread) and BS (block size).
// BS varies so small N (1024, 2048) can be served without a 1024-thread block
// that would have NP=0 and spill everything.
template <int NP, int BS>
__global__ __launch_bounds__(BS) void softmax_online_kernel(
    const float *__restrict__ in, float *__restrict__ out, int M) {
  constexpr int N =
      NP * BS * 4; // recovered at compile time for pointer arithmetic
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
    // Online softmax: update (max, sum) for each of the 4 elements.
    // When max increases, the old partial sum is rescaled by
    // exp(old_max - new_max).
    float vals[4] = {buf[i].x, buf[i].y, buf[i].z, buf[i].w};
#pragma unroll
    for (int j = 0; j < 4; j++) {
      float new_max = fmaxf(local_max, vals[j]);
      local_sum =
          local_sum * __expf(local_max - new_max) + __expf(vals[j] - new_max);
      local_max = new_max;
    }
  }

  // Single combined reduction: (max, sum) reduced together → 2 syncs total
  blockReduceOnline<BS>(local_max, local_sum, smem_max, smem_sum);
  float row_max = local_max;
  float invSum = __frcp_rn(local_sum);

  // Phase 2: Normalize + write back
#pragma unroll
  for (int i = 0; i < NP; i++) {
    buf[i].x = __expf(buf[i].x - row_max) * invSum;
    buf[i].y = __expf(buf[i].y - row_max) * invSum;
    buf[i].z = __expf(buf[i].z - row_max) * invSum;
    buf[i].w = __expf(buf[i].w - row_max) * invSum;
    row_out[tid + i * BS] = buf[i];
  }
}

#define ONLINE_2PASS_MIN_NP 16 

template <int NP, int BS>
__global__ __launch_bounds__(BS) void softmax_online_2pass_kernel(
    const float *__restrict__ in, float *__restrict__ out, int M) {
  constexpr int N       = NP * BS * 4;
  constexpr int n_warps = BS / WARP_SIZE;

  extern __shared__ float smem[];
  float *smem_max = smem;
  float *smem_sum = smem + n_warps;

  const int tid = threadIdx.x;
  const int row = blockIdx.x;
  if (row >= M) return;

  const float4 *row_in  = reinterpret_cast<const float4 *>(in  + row * N);
  float4       *row_out = reinterpret_cast<float4       *>(out + row * N);

  // Pass 1: accumulate (max, sum) — no buf[], O(1) register footprint
  float local_max = -INFINITY, local_sum = 0.0f;
#pragma unroll
  for (int i = 0; i < NP; i++) {
    float4 v = row_in[tid + i * BS];
    float vs[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int j = 0; j < 4; j++) {
      float new_max = fmaxf(local_max, vs[j]);
      local_sum = local_sum * __expf(local_max - new_max) + __expf(vs[j] - new_max);
      local_max = new_max;
    }
  }

  blockReduceOnline<BS>(local_max, local_sum, smem_max, smem_sum);
  const float row_max = local_max;
  const float invSum  = __frcp_rn(local_sum);

  // Pass 2: reload via __ldg (read-only cache) → exp-normalize → write
#pragma unroll
  for (int i = 0; i < NP; i++) {
    float4 v = __ldg(row_in + tid + i * BS);
    v.x = __expf(v.x - row_max) * invSum;
    v.y = __expf(v.y - row_max) * invSum;
    v.z = __expf(v.z - row_max) * invSum;
    v.w = __expf(v.w - row_max) * invSum;
    row_out[tid + i * BS] = v;
  }
}

// Supported sizes defined by ONLINE_SIZES in utils.h.
// NP = N / (BS * 4) — computed by the macro so the kernel template is correct.
// Routes to 2-pass for NP >= ONLINE_2PASS_MIN_NP to avoid register spilling.
void launch_softmax_online(const float *d_in, float *d_out, int M, int N) {
#define LAUNCH_CASE(n_, bs_)                                                   \
  case n_: {                                                                   \
    constexpr int np_   = (n_) / ((bs_) * 4);                                  \
    const size_t  smem_ = 2 * ((bs_) / WARP_SIZE) * sizeof(float);             \
    if constexpr (np_ >= ONLINE_2PASS_MIN_NP)                                  \
      softmax_online_2pass_kernel<np_, bs_><<<M, bs_, smem_>>>(d_in, d_out, M); \
    else                                                                       \
      softmax_online_kernel<np_, bs_><<<M, bs_, smem_>>>(d_in, d_out, M);      \
    break;                                                                     \
  }
  switch (N) {
    ONLINE_SIZES(LAUNCH_CASE)
  default:
    fprintf(stderr, "softmax_online: N=%d not supported (%d..%d pow2)\n", N,
            ONLINE_MIN_N, ONLINE_MAX_N);
    abort();
  }
#undef LAUNCH_CASE
}

#ifdef SOFTMAX_STANDALONE

int main() {
  const int M = BENCH_DEFAULT_M;
  const int N = 4096; // change to any supported value: 4096..65536 (pow2)
  float *d_in = nullptr;
  float *d_out = nullptr;

  bench_alloc(&d_in, &d_out, M, N);
  bench_init_curand(d_in, M, N);

  for (int i = 0; i < 5; i++) {
    launch_softmax_online(d_in, d_out, M, N);
  }
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  float ms;
  bench_timing_begin(&start, &stop);
  launch_softmax_online(d_in, d_out, M, N);
  bench_timing_end(start, stop, &ms);

  printf("kernel time: %f ms\n", ms);
  bench_free(d_in, d_out);
  return 0;
}

#endif /* SOFTMAX_STANDALONE */

// Standalone: nvcc -O3 -DSOFTMAX_STANDALONE -lcurand fusedBlock.cu -o
// fusedBlock ncu --kernel-name softmax_online_kernel --launch-skip 5
// --launch-count 1 --set full -o fusedBlock.ncu-rep ./fusedBlock
