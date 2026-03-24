#ifndef SOFTMAX_UTILS_H
#define SOFTMAX_UTILS_H

#include <cuda_runtime.h>
#include <stdio.h>

typedef float floatX;

#define WARP_SIZE 32
#define FUSED_BLOCK_SIZE 1024 // threads per block for block/online kernels

// ---------------------------------------------------------------------------
// Supported N ranges
// ---------------------------------------------------------------------------
#define FUSED_WARP_MIN_N 1024
#define FUSED_WARP_MAX_N 4096
#define FUSED_BLOCK_MIN_N 4096
#define FUSED_BLOCK_MAX_N 65536
#define ONLINE_MIN_N 1024
#define ONLINE_MAX_N 262144

// ---------------------------------------------------------------------------
// X-macros: expand F once per supported N.  Add a new row here to support a
// new size — the launcher switch case expands automatically.
//
// FUSED_WARP_SIZES(F)  — F(N)       kernel: softmax_fused_warp_kernel<N>
// FUSED_BLOCK_SIZES(F) — F(N)       kernel: softmax_fused_block_kernel<N>
// ONLINE_SIZES(F)      — F(N, BS)   kernel: softmax_online_kernel<NP, BS>
//                         BS varies so small N fits without spilling registers
// ---------------------------------------------------------------------------
#define FUSED_WARP_SIZES(F)                                                    \
  F(1024)                                                                      \
  F(2048)                                                                      \
  F(4096)

#define FUSED_BLOCK_SIZES(F)                                                   \
  F(4096)                                                                      \
  F(8192)                                                                      \
  F(16384)                                                                     \
  F(32768)                                                                     \
  F(65536)

// online: BS=256 for N<4096 so NP≥1 without using 1024 threads on a tiny row.
// N=131072 → NP=32 (128 regs for buf alone, likely spills under
// -maxrregcount=128). N=262144 → NP=64 (256 regs for buf, heavy spill —
// included to show the wall).
#define ONLINE_SIZES(F)                                                        \
  F(1024, 256)                                                                 \
  F(2048, 256)                                                                 \
  F(4096, 1024)                                                                \
  F(8192, 1024)                                                                \
  F(16384, 1024)                                                               \
  F(32768, 1024)                                                               \
  F(65536, 1024)                                                               \
  F(131072, 1024)                                                              \
  F(262144, 1024)

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
#endif

#ifdef SOFTMAX_STANDALONE

#include <curand.h>

#define BENCH_DEFAULT_M 4096
#define BENCH_DEFAULT_N 4096
#define BENCH_THREADS 256

static inline void bench_alloc(floatX **in, floatX **out, int M, int N) {
  CUDA_CHECK(cudaMalloc(in, (size_t)M * N * sizeof(floatX)));
  CUDA_CHECK(cudaMalloc(out, (size_t)M * N * sizeof(floatX)));
}

static inline void bench_init_curand(floatX *in, int M, int N) {
#ifdef __CUDACC__
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, 69);
  curandGenerateUniform(gen, in, (size_t)M * N);
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

static inline void bench_free(floatX *in, floatX *out) {
  cudaFree(in);
  cudaFree(out);
}

#endif /* SOFTMAX_STANDALONE */

#endif /* SOFTMAX_UTILS_H */
