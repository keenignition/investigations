#include "utils.h"
#include <math.h>

__device__ __forceinline__ float warpReduceMaxFB(float val) {
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1)
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
  return val;
}

__device__ __forceinline__ float warpReduceSumFB(float val) {
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(0xffffffff, val, mask);
  return val;
}

// gathers max from all warps in the block and gives us the max in the block
__device__ __forceinline__ float blockReduceMax(float val, float *scratch) {
  constexpr int n_warps = FUSED_BLOCK_SIZE / WARP_SIZE;
  const int wid = threadIdx.x / WARP_SIZE;
  const int lane = threadIdx.x % WARP_SIZE;

  // max across warps -> 1 entry to smem per warp
  val = warpReduceMaxFB(val);
  if (lane == 0)
    scratch[wid] = val;
  __syncthreads();

  // max across smem scross the warp entries to find final max in row
  val = (threadIdx.x < n_warps) ? scratch[threadIdx.x] : -INFINITY;
  if (wid == 0)
    val = warpReduceMaxFB(val);
  if (threadIdx.x == 0)
    scratch[0] = val;
  __syncthreads();
  return scratch[0];
}

__device__ __forceinline__ float blockReduceSum(float val, float *scratch) {
  constexpr int n_warps = FUSED_BLOCK_SIZE / WARP_SIZE;
  const int wid = threadIdx.x / WARP_SIZE;
  const int lane = threadIdx.x % WARP_SIZE;

  // reduce across warps -> 1 entry to smem per warp
  val = warpReduceSumFB(val);
  if (lane == 0)
    scratch[wid] = val;
  __syncthreads();

  // reduce across smem scross the warp entries to find final reduction
  val = (threadIdx.x < n_warps) ? scratch[threadIdx.x] : 0.0f;
  if (wid == 0)
    val = warpReduceSumFB(val);
  if (threadIdx.x == 0)
    scratch[0] = val;
  __syncthreads();
  return scratch[0];
}

template <int N>
__global__ __launch_bounds__(FUSED_BLOCK_SIZE) void softmax_fused_block_kernel(
    const float *__restrict__ in, float *__restrict__ out, int M) {
  constexpr int NP = N / (FUSED_BLOCK_SIZE * 4); // float4s per thread
  constexpr int n_warps = FUSED_BLOCK_SIZE / WARP_SIZE;

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

  // 1. for all packs
  // read vram once
  //    - Load row into registers
  //    - compute local max
  // sync max reduce across the block
#pragma unroll
  for (int i = 0; i < NP; i++) {
    buf[i] = row_in[tid + i * FUSED_BLOCK_SIZE]; // coalesced
    local_max = fmaxf(
        local_max, fmaxf(fmaxf(buf[i].x, buf[i].y), fmaxf(buf[i].z, buf[i].w)));
  }
  float row_max = blockReduceMax(local_max, smem_max);

  // 2. [register] all all packs
  // - exp on register data
  // - accumulate local sum
  // sync reduce across block and multiply by reciprocal
  float local_sum = 0.0f;
#pragma unroll
  for (int i = 0; i < NP; i++) {
    buf[i].x = __expf(buf[i].x - row_max);
    buf[i].y = __expf(buf[i].y - row_max);
    buf[i].z = __expf(buf[i].z - row_max);
    buf[i].w = __expf(buf[i].w - row_max);
    local_sum += buf[i].x + buf[i].y + buf[i].z + buf[i].w;
  }
  float invSum = __frcp_rn(blockReduceSum(local_sum, smem_sum));

  // 3. Normalize register data, write back
#pragma unroll
  for (int i = 0; i < NP; i++) {
    buf[i].x *= invSum;
    buf[i].y *= invSum;
    buf[i].z *= invSum;
    buf[i].w *= invSum;
    row_out[tid + i * FUSED_BLOCK_SIZE] = buf[i];
  }
}

// Minimum N: FUSED_BLOCK_SIZE * 4 = 4096 (NP must be >= 1).
// Supported sizes defined by FUSED_BLOCK_SIZES in utils.h.
void launch_softmax_fused_block(const float *d_in, float *d_out, int M, int N) {
  constexpr int n_warps = FUSED_BLOCK_SIZE / WARP_SIZE;
  const size_t smem = 2 * n_warps * sizeof(float);
#define LAUNCH_CASE(n_)                                                        \
  case n_:                                                                     \
    softmax_fused_block_kernel<n_>                                             \
        <<<M, FUSED_BLOCK_SIZE, smem>>>(d_in, d_out, M);                       \
    break;
  switch (N) {
    FUSED_BLOCK_SIZES(LAUNCH_CASE)
  default:
    fprintf(stderr, "softmax_fused_block: N=%d not supported (%d..%d pow2)\n",
            N, FUSED_BLOCK_MIN_N, FUSED_BLOCK_MAX_N);
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
    launch_softmax_fused_block(d_in, d_out, M, N);
  }
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  float ms;
  bench_timing_begin(&start, &stop);
  launch_softmax_fused_block(d_in, d_out, M, N);
  bench_timing_end(start, stop, &ms);

  printf("kernel time: %f ms\n", ms);
  bench_free(d_in, d_out);
  return 0;
}

#endif /* SOFTMAX_STANDALONE */

// Standalone: nvcc -O3 -DSOFTMAX_STANDALONE -lcurand fusedBlock.cu -o
// fusedBlock ncu --kernel-name softmax_fused_block_kernel --launch-skip 5
// --launch-count 1 --set full -o fusedBlock.ncu-rep ./fusedBlock
