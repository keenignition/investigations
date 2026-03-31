#ifndef SOFTMAX_UTILS_H
#define SOFTMAX_UTILS_H

#include <cuda_runtime.h>
#include <stdio.h>

// All architecture-specific constants (block sizes, N ranges, X-macros, V2
// tuning knobs) live in kernel_config.h, which is auto-generated from the
// per-arch YAML in configs/archs/<sm_XX>.yml.
// Re-generate: python scripts/gen_kernel_config.py
#include "kernel_config.h"

typedef float floatX;

#define WARP_SIZE 32

// ---------------------------------------------------------------------------
// X-macros overview (defined in kernel_config.h):
//
//   FUSED_WARP_SIZES(F)       — F(N)       softmax_fused_warp_kernel<N>
//   FUSED_BLOCK_SIZES(F)      — F(N)       softmax_fused_block_kernel<N>
//   ONLINE_SIZES(F)           — F(N, BS)   softmax_online_kernel<NP, BS>
//   ONLINE_V2_SINGLE_SIZES(F) — F(N, BS)   softmax_v2_single_kernel<NP, BS>
// ---------------------------------------------------------------------------

#define CUDA_CHECK(err)                                                        \
  do {                                                                         \
    if ((err) != cudaSuccess) {                                                \
      printf("CUDA error: %s\n", cudaGetErrorString(err));                     \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#ifdef __CUDACC__
void launch_softmax_naive(const float *in, float *out, int M, int N);
void launch_softmax_wr(const float *in, float *out, int M, int N);
void launch_softmax_fused_warp(const float *in, float *out, int M, int N);
void launch_softmax_fused_block(const float *in, float *out, int M, int N);
void launch_softmax_online(const float *in, float *out, int M, int N);
void launch_softmax_online_v2(const float *in, float *out, int M, int N);
#endif

#ifdef SOFTMAX_STANDALONE

#include <curand.h>
#include <stdlib.h>

#define BENCH_DEFAULT_M 4096
#define BENCH_DEFAULT_N 4096
#define BENCH_THREADS 1024

// Full CPU→GPU lifecycle:
//   1. malloc host buffers
//   2. fill h_in with deterministic random data on CPU
//   3. cudaMalloc device buffers
//   4. cudaMemcpy h_in → d_in  (H2D)
static inline void bench_alloc(floatX **h_in, floatX **h_out, floatX **d_in,
                               floatX **d_out, int M, int N) {
  size_t bytes = (size_t)M * N * sizeof(floatX);

  *h_in = (floatX *)malloc(bytes);
  *h_out = (floatX *)calloc((size_t)M * N, sizeof(floatX));

  srand(42);
  for (int i = 0; i < M * N; i++)
    (*h_in)[i] = (floatX)rand() / RAND_MAX * 2.0f - 1.0f; // uniform [-1, 1]

  CUDA_CHECK(cudaMalloc(d_in, bytes));
  CUDA_CHECK(cudaMalloc(d_out, bytes));
  CUDA_CHECK(cudaMemcpy(*d_in, *h_in, bytes, cudaMemcpyHostToDevice));
}

// D2H copy — call after kernel to bring results back for verification
static inline void bench_copy_back(floatX *h_out, floatX *d_out, int M, int N) {
  CUDA_CHECK(cudaMemcpy(h_out, d_out, (size_t)M * N * sizeof(floatX),
                        cudaMemcpyDeviceToHost));
}

// GPU-only init via cuRAND (alternative to bench_alloc's CPU fill + H2D)
static inline void bench_init_curand(floatX *d_in, int M, int N) {
#ifdef __CUDACC__
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, 69);
  curandGenerateUniform(gen, d_in, (size_t)M * N);
  curandDestroyGenerator(gen);
#endif
}

static inline void bench_timing_begin(cudaEvent_t *start, cudaEvent_t *stop) {
  cudaEventCreate(start);
  cudaEventCreate(stop);
  cudaEventRecord(*start);
}

static inline void bench_timing_end(cudaEvent_t start, cudaEvent_t stop,
                                    float *ms) {
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

static inline void bench_free(floatX *h_in, floatX *h_out, floatX *d_in,
                              floatX *d_out) {
  free(h_in);
  free(h_out);
  cudaFree(d_in);
  cudaFree(d_out);
}

#endif /* SOFTMAX_STANDALONE */

#endif /* SOFTMAX_UTILS_H */
