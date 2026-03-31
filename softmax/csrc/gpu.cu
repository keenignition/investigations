/* gpu_bw.cu */

#include <stdio.h>
#include <cuda_runtime.h>

int main() {
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	int memClockKHz = 0, coreClockKHz = 0;
	cudaDeviceGetAttribute(&memClockKHz,  cudaDevAttrMemoryClockRate, 0);
	cudaDeviceGetAttribute(&coreClockKHz, cudaDevAttrClockRate,       0);

	printf("=== Device ===\n");
	printf("GPU:                                %s\n", prop.name);
	printf("Compute Capability:                 %d.%d\n", prop.major, prop.minor);
	printf("Total Global Memory:                %.2f GB\n", prop.totalGlobalMem / (1024.0*1024.0*1024.0));
	printf("L2 Cache Size:                      %d bytes\n", prop.l2CacheSize);

	printf("\n=== Streaming Multiprocessors ===\n");
	printf("Number of SMs:                      %d\n", prop.multiProcessorCount);
	printf("Max Threads per SM:                 %d\n", prop.maxThreadsPerMultiProcessor);
	printf("Total Registers per SM:             %d\n", prop.regsPerMultiprocessor);
	printf("Shared Memory per SM:               %zu bytes\n", prop.sharedMemPerMultiprocessor);

	printf("\n=== Block Limits ===\n");
	printf("Max Threads per Block:              %d\n", prop.maxThreadsPerBlock);
	printf("Total Registers per Block:          %d\n", prop.regsPerBlock);
	printf("Shared Memory per Block:            %zu bytes\n", prop.sharedMemPerBlock);
	printf("Warp Size:                          %d\n", prop.warpSize);
	printf("Max Block Dims:                     [%d, %d, %d]\n",
		prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);

	printf("\n=== Grid Limits ===\n");
	printf("Max Grid Dims:                      [%d, %d, %d]\n",
		prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

	printf("\n=== Memory ===\n");
	printf("Memory Clock Rate:                  %.0f MHz\n", memClockKHz / 1e3);
	printf("Memory Bus Width:                   %d bits\n", prop.memoryBusWidth);
	double bw_gbps = 2.0 * memClockKHz * 1e3 * (prop.memoryBusWidth / 8) / 1e9;
	printf("Peak Memory Bandwidth:              %.2f GB/s\n", bw_gbps);

	printf("\n=== Clocks ===\n");
	printf("GPU Clock Rate:                     %.0f MHz\n", coreClockKHz / 1e3);

	return 0;
}