#include "utils.h"
#include <math.h>

#define WARP_SIZE 32

/* --------------------------------------------------------------------------- */
/* Warp-level reduction                                                        */
/* --------------------------------------------------------------------------- */

__device__ __forceinline__ float warpReduceMax(float val) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    // 0xffffffff -> keep all 32 threads active
    // __shfl_down_sync: lane 0 accumulates the correct result
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return val;
}

__device__ __forceinline__ float warpReduceSum(float val) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xffffffff, val, offset);
  return val;
}

__global__ void softmax_kernel_wr(const float *__restrict__ in,
                                  float *__restrict__ out, int M, int N) {
  extern __shared__ float smem[]; // we'll need smem to hold one value per warp
                                  // ---> 256 threads / 32 = 8 warps = 8 floats
  int row = blockIdx.x;
  int tid = threadIdx.x;
  int laneId = tid % 32;
  int warpId = tid / 32;

  if (row < M) {

    // ==========================================
    // 1. COMPUTE ROW MAX
    // ==========================================

    // Thread-local reduction
    float localMax = -INFINITY;
    for (int i = tid; i < N; i += blockDim.x) {
      float v = in[row * N + i];
      localMax = fmaxf(localMax, v);
    }

    // Warp-level reduction: all warps reduce independently
    localMax = warpReduceMax(localMax);

    // Write warp max to shared memory
    // localMax are stuck in the private registers of Threads 0, 32, 64, 96,
    // 128, 160, 192, and 224
    if (laneId == 0) {
      smem[warpId] = localMax;
    }
    __syncthreads();

    localMax = (tid < (blockDim.x / 32)) ? smem[laneId] : -INFINITY;
    if (warpId == 0) {
      localMax = warpReduceMax(localMax);
      if (tid == 0) {
        smem[0] = localMax;
      }
    }
    __syncthreads();
    float rowMax = smem[0]; // broadcast

    // ==========================================
    // 2. NORMALIZE AND COMPUTE ROW SUM
    // ==========================================
    float localSum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
      float v = in[row * N + i];
      float e = expf(v - rowMax);
      out[row * N + i] = e;
      localSum += e;
    }

    // Warp-level reduction for the sum: all warps reduce independently
    localSum = warpReduceSum(localSum);
    if (laneId == 0) {
      smem[warpId] = localSum;
    }
    __syncthreads();
    localSum = (tid < (blockDim.x / 32)) ? smem[laneId] : 0.0f;
    if (warpId == 0) {
      localSum = warpReduceSum(localSum);
      if (tid == 0) {
        smem[0] = localSum;
      }
    }
    __syncthreads();
    float rowSum = smem[0];

    // ==========================================
    // 3. FINAL DIVISION
    // ==========================================
    for (int i = tid; i < N; i += blockDim.x) {
      out[row * N + i] /= rowSum;
    }
  }
}

#ifdef SOFTMAX_STANDALONE

int main() {
  int M = BENCH_DEFAULT_M;
  int N = BENCH_DEFAULT_N;
  floatX *in, *out;

  bench_alloc(&in, &out, M, N);
  bench_init_curand(in, M, N);

  const int threads = BENCH_THREADS;
  const int blocks = M;
  const size_t smem = (threads / 32) * sizeof(float);

  for (int i = 0; i < 5; i++) {
    softmax_kernel_wr<<<blocks, threads, smem>>>(in, out, M, N);
  }
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  float ms;
  bench_timing_begin(&start, &stop);
  softmax_kernel_wr<<<blocks, threads, smem>>>(in, out, M, N);
  bench_timing_end(start, stop, &ms);

  printf("kernel time: %f ms\n", ms);
  bench_free(in, out);
  return 0;
}

#endif /* SOFTMAX_STANDALONE */

// Standalone: nvcc -O3 -DSOFTMAX_STANDALONE -lcurand wr.cu -o wr
// ncu --kernel-name softmax_kernel_wr --launch-skip 5 --launch-count 1 --set
// full -o wr.ncu-rep ./wr
