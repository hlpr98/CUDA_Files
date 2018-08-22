#include <cuda.h>
#include "cuda_runtime.h"
#include "wb.h"
#include <bits/stdc++.h>
//#include <cstdio>
//#include <cstdlib>

#define NUM_BINS 4096
#define BIN_CAP 127

#define CUDA_CHECK(ans)                                                   \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
	bool abort = true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),
			file, line);
		if (abort)
			exit(code);
	}
}

#define THREADS 8 

__global__ void histogram(unsigned int* output, unsigned int* input, int inputLength) {
	
	__shared__ unsigned int value[NUM_BINS];
	__shared__ int done[NUM_BINS];
	unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
	if(x < inputLength){
		unsigned int idx = input[x];
		value[idx] = 0;
		done[idx] = 0;
		__syncthreads();

		atomicAdd(value + idx, (unsigned int)1);
		__syncthreads();
		atomicMin(value + idx, BIN_CAP);
		__syncthreads();
		
		if(!atomicAdd(done + idx, 1)){
			atomicAdd(output + idx, value[idx]);
			atomicMin(output + idx, BIN_CAP);
		}
	}

}


int main(int argc, char *argv[]) {

	int inputLength;
	unsigned int *hostInput;
	unsigned int *hostBins;
	unsigned int *deviceInput;
	unsigned int *deviceBins;

	/* Read input arguments here */
	wbArg_t args = wbArg_read(argc, argv);

	wbTime_start(Generic, "Importing data and creating memory on host");
	hostInput = (unsigned int *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
	hostBins = (unsigned int *)calloc(NUM_BINS, sizeof(unsigned int));
	wbTime_stop(Generic, "Importing data and creating memory on host");

	wbLog(TRACE, "The input length is ", inputLength);
	wbLog(TRACE, "The number of bins is ", NUM_BINS);

	wbTime_start(GPU, "Allocating GPU memory.");
	//@@ Allocate GPU memory here
	cudaMalloc((void **)&deviceInput, inputLength * sizeof(unsigned int));
	cudaMalloc((void **)&deviceBins, NUM_BINS * sizeof(unsigned int));
	cudaMemset(deviceBins, 0, NUM_BINS);
	CUDA_CHECK(cudaDeviceSynchronize());
	wbTime_stop(GPU, "Allocating GPU memory.");

	wbTime_start(GPU, "Copying input memory to the GPU.");
	//@@ Copy memory to the GPU here
	cudaMemcpy(deviceInput, hostInput, inputLength * sizeof(unsigned int), cudaMemcpyHostToDevice);
	CUDA_CHECK(cudaDeviceSynchronize());
	wbTime_stop(GPU, "Copying input memory to the GPU.");

	// Launch kernel
	// ----------------------------------------------------------
	wbLog(TRACE, "Launching kernel");
	wbTime_start(Compute, "Performing CUDA computation");
	//@@ Perform kernel computation here

	dim3 block(THREADS, 1, 1);
	dim3 grid(ceil(inputLength / block.x), 1, 1);

	histogram << <grid, block >> > (deviceBins, deviceInput, inputLength);

	wbTime_stop(Compute, "Performing CUDA computation");

	wbTime_start(Copy, "Copying output memory to the CPU");
	//@@ Copy the GPU memory back to the CPU here
	cudaMemcpy(hostBins, deviceBins, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	CUDA_CHECK(cudaDeviceSynchronize());
	wbTime_stop(Copy, "Copying output memory to the CPU");

	wbTime_start(GPU, "Freeing GPU Memory");
	//@@ Free the GPU memory here
	cudaFree(deviceInput);
	cudaFree(deviceBins);
	wbTime_stop(GPU, "Freeing GPU Memory");

	// Verify correctness
	// -----------------------------------------------------
	wbSolution(args, hostBins, NUM_BINS);

	free(hostBins);
	free(hostInput);
	return 0;
}

