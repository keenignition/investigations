#include "utils.h"
#include <math.h>

// inspired by
// https://github.com/facebookincubator/AITemplate/wiki/How-to-write-a-fast-Softmax-CUDA-kernel%3F#warp-reduction---when-k-is-small
// We template N so the compiler knows the exact size at compile time.
// For N > 4096 we exceed that and the compiler spills to local memory ->
// catastrophic slowdown. use a different kernel for N > 4096.
//
//
// * Per-thread register budget:
// *   Each thread holds NUM_PACKS = N / (32 threads × 4 floats) float4s.
// *   For N=1024: 8 float4s = 32 floats = 32 registers just for data.
// *   For N=4096: 32 float4s = 128 registers. Still fits comfortably.
// *   For N>4096: registers overflow → compiler spills to VRAM →
// *   performance cliff.
// *
template <int N>
__global__ void softmax_fused_kernel(const float *__restrict__ in,
                                     float *__restrict__ out, int M) {

  // Calculate how many float4 "packs" each thread needs to hold.
  // If N=1024, N is 256 float4s. Divided by 32 threads in a warp = 8 packs per
  // thread.
  // IMPORTANT: This kernel keeps the full row in REGISTERS (float4
  // buf[NUM_PACKS]). NUM_PACKS = N/128, so per-thread register use scales with
  // N.
  //
  //  * Why 128 threads / 4 warps per block?
  // *   Each SM has 4 sub-cores, each with its own warp scheduler.
  // *   128 threads = 4 warps → one warp per sub-core → perfect fit.
  // *   Also: 128 threads × 255 max regs = 32,640 regs, which is half
  // *   the SM's 65,536 register file, so 2 blocks can co-reside.
  constexpr int NUM_PACKS = N / 128;

  // allocated purely in ultra-fast hardware registers.
  //
  // * Why float4?
  // *   A single global memory transaction can move 128 bits (16 bytes)
  // *   = 4 floats.  Using float4 means each thread fetches the maximum
  // *   payload per instruction.
  float4 buf[NUM_PACKS];
  float localMax = -INFINITY;

  // X-dimension (32): warp. Y-dimension (4): 4 warps per block.
  // why 4 warps per block? each SM, the hardware is divided into 4
  // independent processing blocks (sub-cores). Each sub-core has its own Warp
  // Scheduler. Since 1 Warp = 32 threads, a block of 128 threads equals exactly
  // 4 Warps. in a 128-thread block, the SM flawlessly hands exactly one Warp to
  // each of its four sub-cores.
  int warpId = threadIdx.y;
  int laneId = threadIdx.x;

  int row = blockIdx.x * blockDim.y + warpId;
  if (row >= M)
    return;

  // Cast our global pointers to float4 so we can fetch 16 bytes at a time
  const float4 *row_in = reinterpret_cast<const float4 *>(in + row * N);
  float4 *row_out = reinterpret_cast<float4 *>(out + row * N);

// ==========================================
// PASS 1: READ FROM VRAM ONCE
// ==========================================
#pragma unroll
  for (int i = 0; i < NUM_PACKS; i++) {
    int col = i * 32 + laneId;

    // Read 4 floats from VRAM and stash them in our private pocket
    buf[i] = row_in[col];

    // Find the max among the 4 floats we just loaded
    float max_val = fmaxf(fmaxf(buf[i].x, buf[i].y), fmaxf(buf[i].z, buf[i].w));
    localMax = fmaxf(localMax, max_val);
  }
  // GLOBAL VRAM READS ARE NOW 100% FINISHED!

  // ==========================================
  // PASS 2: WARP MAX (Butterfly)
  // ==========================================
  // We use __shfl_xor_sync here instead of down_sync
  // XOR is a "butterfly reduction". Instead of funnelling the max only to
  // Thread 0, it perfectly swaps data so that at the end, EVERY thread holds
  // the final rowMax.
  for (int mask = 16; mask > 0; mask /= 2) {
    //   * Key hardware primitive — __shfl_xor_sync (butterfly):
    // *   Unlike __shfl_down (which funnels values to thread 0 only),
    // *   XOR-based shuffles swap values symmetrically.  After 5 rounds,
    // *   EVERY thread in the warp holds the final reduced value.
    // *   No need for a broadcast step.
    localMax = fmaxf(localMax, __shfl_xor_sync(0xffffffff, localMax, mask));
  }

  // ==========================================
  // PASS 3: MATH IN THE REGISTERS
  // ==========================================
  float localSum = 0.0f;
#pragma unroll
  for (int i = 0; i < NUM_PACKS; i++) {
    // Subtract max and exponentiate directly inside the registers
    buf[i].x = expf(buf[i].x - localMax);
    buf[i].y = expf(buf[i].y - localMax);
    buf[i].z = expf(buf[i].z - localMax);
    buf[i].w = expf(buf[i].w - localMax);

    // Accumulate local sum
    localSum += (buf[i].x + buf[i].y + buf[i].z + buf[i].w);
  }

  // ==========================================
  // PASS 4: WARP TOURNAMENT FOR SUM
  // ==========================================
  for (int mask = 16; mask > 0; mask /= 2) {
    localSum += __shfl_xor_sync(0xffffffff, localSum, mask);
  }
// Now EVERY thread knows the total rowSum.

// ==========================================
// PASS 5: DIVIDE AND WRITE TO VRAM ONCE
// ==========================================
#pragma unroll
  for (int i = 0; i < NUM_PACKS; i++) {
    // Final normalization inside registers
    buf[i].x /= localSum;
    buf[i].y /= localSum;
    buf[i].z /= localSum;
    buf[i].w /= localSum;

    // Ship the finalized 4 floats back to VRAM
    int col = i * 32 + laneId;
    row_out[col] = buf[i];
  }
}

// N must be a multiple of 128; we instantiate for benchmark sizes 1024..65536.
void launch_softmax_fused(const float *d_in, float *d_out, int M, int N) {
  dim3 blockSize(32, 4);
  dim3 gridSize((M + 3) / 4);
  switch (N) {
  case 1024:
    softmax_fused_kernel<1024><<<gridSize, blockSize>>>(d_in, d_out, M);
    break;
  case 2048:
    softmax_fused_kernel<2048><<<gridSize, blockSize>>>(d_in, d_out, M);
    break;
  case 4096:
    softmax_fused_kernel<4096><<<gridSize, blockSize>>>(d_in, d_out, M);
    break;
  // anything beyond this is kind of pointless, but adding for the saake of
  // benchmarking
  case 8192:
    softmax_fused_kernel<8192><<<gridSize, blockSize>>>(d_in, d_out, M);
    break;
  case 16384:
    softmax_fused_kernel<16384><<<gridSize, blockSize>>>(d_in, d_out, M);
    break;
  case 32768:
    softmax_fused_kernel<32768><<<gridSize, blockSize>>>(d_in, d_out, M);
    break;
  case 65536:
    softmax_fused_kernel<65536><<<gridSize, blockSize>>>(d_in, d_out, M);
    break;
  case 131072:
    softmax_fused_kernel<131072><<<gridSize, blockSize>>>(d_in, d_out, M);
    break;
  case 262144:
    softmax_fused_kernel<262144><<<gridSize, blockSize>>>(d_in, d_out, M);
    break;
  default:
    fprintf(stderr,
            "softmax_fused: N=%d not supported (use 1024,2048,...,65536)\n", N);
    abort();
  }
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
    launch_softmax_fused(d_in, d_out, M, N);
  }
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  float ms;
  bench_timing_begin(&start, &stop);
  launch_softmax_fused(d_in, d_out, M, N);
  bench_timing_end(start, stop, &ms);

  printf("kernel time: %f ms\n", ms);
  bench_free(d_in, d_out);
  return 0;
}

#endif /* SOFTMAX_STANDALONE */
