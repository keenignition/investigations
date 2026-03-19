#include "utils.h"
#include <math.h>

__global__ void softmax_kernel(const float *__restrict__ in,
                               float *__restrict__ out, int M, int N) {
  extern __shared__ float smem[];
  int row = blockIdx.x;
  int tid = threadIdx.x;

  if (row < M) {
    // --- get max of each row

    // iterate columns using the threads
    // so 256 threads will hold the max per thread
    // stride of 25, ie., 1024/256 = 4 values
    // each thread will iterate over 4 values and capture the max among them
    float localMax = -INFINITY;
    for (int i = tid; i < N; i += blockDim.x) {
      float v = in[row * N + i];
      localMax = fmaxf(localMax, v);
    }

    // add all 256 partial max to smem
    smem[tid] = localMax;
    __syncthreads();

    // tree reduce smem
    // width wise half folds to get the max
    // i >>= 1 --- bitshift divide by 2 ---> i = i/ 2
    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
      if (tid < i) {
        smem[tid] = fmaxf(smem[tid], smem[tid + i]);
      }
      __syncthreads(); // wait as every mem needs to be written for true max
    }

    // smme[0] will end up with the max val
    float rowMax = smem[0];

    // --- normalize and compute numerator and denominator

    // iterate over col, 256 threads
    // get exp(i - max)
    // store num in out tensor
    // accumulate sum of num
    float localSum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
      float v = in[row * N + i];
      float e = expf(v - rowMax);
      out[row * N + i] = e;
      localSum += e;
    }

    // all 256 sums to smem
    smem[tid] = localSum;
    __syncthreads();

    // add 256 smems per row to get final row sum
    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
      if (tid < i) {
        smem[tid] += smem[tid + i];
      }
      __syncthreads();
    }
    float rowSum = smem[0];

    // iterate over col — each thread divides its elements by row sum
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
  const size_t smem = threads * sizeof(float);

  for (int i = 0; i < 5; i++) {
    softmax_kernel<<<blocks, threads, smem>>>(in, out, M, N);
  }
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  float ms;
  bench_timing_begin(&start, &stop);
  softmax_kernel<<<blocks, threads, smem>>>(in, out, M, N);
  bench_timing_end(start, stop, &ms);

  printf("kernel time: %f ms\n", ms);
  bench_free(in, out);
  return 0;
}

#endif /* SOFTMAX_STANDALONE */

// Standalone: nvcc -O3 -DSOFTMAX_STANDALONE -lcurand naive.cu -o naive
// ncu --kernel-name softmax --launch-skip 5 --launch-count 1 --set full -o
// naive.ncu-rep ./naive
