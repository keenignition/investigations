#include "utils.h"
#include <math.h>

// inspired by
// https://github.com/facebookincubator/AITemplate/wiki/How-to-write-a-fast-Softmax-CUDA-kernel%3F#warp-reduction---when-k-is-small
// We template N so the compiler knows the exact size at compile time.
// For N > 4096 we exceed that and the compiler spills to local memory ->
// catastrophic slowdown. use a different kernel for N > 4096.

// (32,4) block = 128 threads; tell the compiler so it can tune register usage
constexpr int THREADBLOCK_SIZE = 128;

template <int N>
__global__ __launch_bounds__(THREADBLOCK_SIZE) void softmax_fused_warp_kernel(
    const float *__restrict__ in, float *__restrict__ out, int M) {

  constexpr int NP = N / 128;
  // (32,4) block: tid.x = lane (0..31), tid.y = warp (0..3)
  int lid = threadIdx.x;
  int wid = threadIdx.y;

  int row = blockIdx.x * blockDim.y + wid;
  if (row >= M)
    return;

  const float4 *v_row_in = reinterpret_cast<const float4 *>(row * N + in);
  float4 *v_row_out = reinterpret_cast<float4 *>(row * N + out);

  float4 buf[NP];
  float localMax = -INFINITY;
  // 1. for all packs
  //  read vram once
  //    - store elem in register
  //    - store to register localMax
  // sync max across all warps
#pragma unroll
  for (int i = 0; i < NP; i++) {
    buf[i] = v_row_in[lid + i * WARP_SIZE];
    float max = fmaxf(fmaxf(buf[i].x, buf[i].y), fmaxf(buf[i].z, buf[i].w));
    localMax = fmaxf(localMax, max);
  }
  for (int mask = 16; mask > 0; mask >>= 1) {
    localMax = fmaxf(localMax, __shfl_xor_sync(0xffffffff, localMax, mask));
  }

  float localSum = 0.0f;
  // 2. [register] for all packs
  // - num math (expf(i-max))
  // - localSum
  // sync localSum across all warps
#pragma unroll
  for (int i = 0; i < NP; i++) {
    buf[i].x = __expf(buf[i].x - localMax);
    buf[i].y = __expf(buf[i].y - localMax);
    buf[i].z = __expf(buf[i].z - localMax);
    buf[i].w = __expf(buf[i].w - localMax);
    localSum += (buf[i].x + buf[i].y + buf[i].z + buf[i].w);
  }
  for (int mask = 16; mask > 0; mask >>= 1) {
    localSum += __shfl_xor_sync(0xffffffff, localSum, mask);
  }

  // 3. [register] for all packs
  // - multiply by reciprocal (one division vs NP*4 divisions)
  // - log back to vram
  float invSum = __frcp_rn(localSum);

#pragma unroll
  for (int i = 0; i < NP; i++) {
    buf[i].x *= invSum;
    buf[i].y *= invSum;
    buf[i].z *= invSum;
    buf[i].w *= invSum;

    int col = i * WARP_SIZE + lid;
    v_row_out[col] = buf[i];
  }
}

// N must be a multiple of 128; supported sizes defined by FUSED_WARP_SIZES.
void launch_softmax_fused_warp(const float *d_in, float *d_out, int M, int N) {
  constexpr int WARPS_PER_BLOCK = THREADBLOCK_SIZE / WARP_SIZE; // 4
  dim3 blockSize(WARP_SIZE, WARPS_PER_BLOCK);
  dim3 gridSize((M + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
#define LAUNCH_CASE(n_)                                                        \
  case n_:                                                                     \
    softmax_fused_warp_kernel<n_><<<gridSize, blockSize>>>(d_in, d_out, M);    \
    break;
  switch (N) {
    FUSED_WARP_SIZES(LAUNCH_CASE)
  default:
    fprintf(stderr, "softmax_fused_warp: N=%d not supported (%d..%d pow2)\n", N,
            FUSED_WARP_MIN_N, FUSED_WARP_MAX_N);
    abort();
  }
#undef LAUNCH_CASE
}

#ifdef SOFTMAX_STANDALONE

int main() {
  const int M = BENCH_DEFAULT_M;
  const int N = 1024;
  float *d_in = nullptr;
  float *d_out = nullptr;

  bench_alloc(&d_in, &d_out, M, N);
  bench_init_curand(d_in, M, N);

  for (int i = 0; i < 5; i++) {
    launch_softmax_fused_warp(d_in, d_out, M, N);
  }
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  float ms;
  bench_timing_begin(&start, &stop);
  launch_softmax_fused_warp(d_in, d_out, M, N);
  bench_timing_end(start, stop, &ms);

  printf("kernel time: %f ms\n", ms);
  bench_free(d_in, d_out);
  return 0;
}

#endif /* SOFTMAX_STANDALONE */
