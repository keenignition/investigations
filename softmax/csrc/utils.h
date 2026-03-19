#ifndef SOFTMAX_UTILS_H
#define SOFTMAX_UTILS_H

#include <stdio.h>
#include <cuda_runtime.h>

/* --------------------------------------------------------------------------- */
/* Common types and macros                                                     */
/* --------------------------------------------------------------------------- */

typedef float floatX;

#define CUDA_CHECK(err)                                                         \
  do {                                                                          \
    if ((err) != cudaSuccess) {                                                  \
      printf("CUDA error: %s\n", cudaGetErrorString(err));                       \
      exit(1);                                                                  \
    }                                                                           \
  } while (0)

/* --------------------------------------------------------------------------- */
/* Kernel declarations (for use from torchBind.cu and standalone)              */
/* --------------------------------------------------------------------------- */

#ifdef __CUDACC__
extern __global__ void softmax_kernel(const float* __restrict__ in,
                                      float* __restrict__ out,
                                      int M, int N);

extern __global__ void softmax_kernel_wr(const float* __restrict__ in,
                                         float* __restrict__ out,
                                         int M, int N);

template <int N>
extern __global__ void softmax_fused_kernel(const float* __restrict__ in,
                                           float* __restrict__ out,
                                           int M);

void launch_softmax_fused(const float* in, float* out, int M, int N);

#endif

/* --------------------------------------------------------------------------- */
/* Shared benchmark helpers (standalone builds only)                           */
/* Define SOFTMAX_STANDALONE when building with nvcc for a standalone binary. */
/* --------------------------------------------------------------------------- */

#ifdef SOFTMAX_STANDALONE

#include <curand.h>

#define BENCH_DEFAULT_M 4096
#define BENCH_DEFAULT_N 4096
#define BENCH_THREADS  256

static inline void bench_alloc(floatX** in, floatX** out, int M, int N) {
  CUDA_CHECK(cudaMalloc(in, (size_t)M * N * sizeof(floatX)));
  CUDA_CHECK(cudaMalloc(out, (size_t)M * N * sizeof(floatX)));
}

static inline void bench_init_curand(floatX* in, int M, int N) {
#ifdef __CUDACC__
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, 42);
  curandGenerateUniform(gen, in, (size_t)M * N);
  curandDestroyGenerator(gen);
#endif
}

static inline void bench_timing_begin(cudaEvent_t* start, cudaEvent_t* stop) {
  cudaEventCreate(start);
  cudaEventCreate(stop);
  cudaEventRecord(*start);
}

static inline void bench_timing_end(cudaEvent_t start, cudaEvent_t stop, float* ms) {
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

static inline void bench_free(floatX* in, floatX* out) {
  cudaFree(in);
  cudaFree(out);
}

#endif /* SOFTMAX_STANDALONE */

#endif /* SOFTMAX_UTILS_H */
