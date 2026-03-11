#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <curand.h>

__global__ void softmax(const float* in, float* out, int M, int N) {
  extern __shared__ float smem[];
  int row = blockIdx.x;
  int tid = threadIdx.x;

  // --- get max of each row

  // iterate columns using the threads
  // so 256 threads will hold the max per thread 
  // stride of 25, ie., 1024/256 = 4 values
  // each thread will iterate over 4 values and capture the max among them
  float localMax = -INFINITY;
  for (int i = tid; i < N; i+=blockDim.x) {
    float v = in[row*N + i];
    localMax = fmaxf(localMax, v);
  }

  // add all 256 partial max to smem
  smem[tid] = localMax;
  __syncthreads();

  // tree reduce smem
  // width wise half folds to get the max
  // i >>= 1 --- bitshift divide by 2 ---> i = i/ 2
  for (int i = blockDim.x / 2; i > 0; i >>= 1) {
    if(tid < i) {
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
  for (int i = tid; i < N; i += blockDim.x)
  {
    float v = in[row * N + i];
    float e = expf(v - rowMax);
    out[row * N + i] = e;
    localSum += e;
  }

  // all 256 sums to smem
  smem[tid] = localSum;
  __syncthreads();

  // add 256 smems per row to get final row sum
  for (int i = blockDim.x / 2; i>0; i>>=1) {
    if(tid < i) {
      smem[tid] += smem[tid + i];
    }
    __syncthreads();
  }
  float rowSum = smem[0];

  // iterate over col
    // out /= row sum
  for (int i = 0; i < N; i+= blockDim.x) {
    out[row * N + i] /= rowSum;
  }
}

#define cudaCheck(err) \
  if(err != cudaSuccess){ \
    printf("CUDA error: %s\n", cudaGetErrorString(err)); \
    exit(1); \
  }

typedef float floatX;

int main() {
  // set size
  int M = 4096;
  int N = 4096;

  floatX *in, *out;

  // device alloc
  cudaCheck(cudaMalloc(&in, (size_t)M * N * sizeof(floatX)));
  cudaCheck(cudaMalloc(&out, (size_t)M * N * sizeof(floatX)));

  // init curand
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, 42);
  curandGenerateUniform(gen, in, (size_t)M * N);
  curandDestroyGenerator(gen);

  // kernel launch
  int threads = 256;
  int blocks = M;
  size_t smem = (threads) * sizeof(float);

  // warmup
  for(int i = 0; i < 5; i++) {
    softmax<<<blocks, threads, smem>>>(in, out, M, N);
  }
  cudaDeviceSynchronize();

  // -- 
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  softmax<<<blocks, threads, smem>>>(in, out, M, N);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms;
  cudaEventElapsedTime(&ms, start, stop);

  printf("kernel time: %f ms\n", ms);

  // free
  cudaFree(in);
  cudaFree(out);

  return 0;
}

// nvcc -O3 -g -G -lcurand naive.cu -o naive
// ncu --kernel-name softmax --launch-skip 5 --launch-count 1 --set full -o naive.ncu-rep ./naive
