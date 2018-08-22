#include <cuda.h>
#include "cuda_runtime.h"
#include "wb.h"
#include <iostream>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define MAX_VAL 255
#define Max(a, b) ((a) < (b) ? (b) : (a))
#define Min(a, b) ((a) > (b) ? (b) : (a))
#define Clamp(a, start, end) Max(Min(a, end), start)
#define value(arry, i, j, k) arry[((i)*width + (j)) * depth + (k)]
#define THREADS 8 


__global__ void stencil(float *output, float *input, int width, int height,
	int depth) {
	//@@ INSERT CODE HERE

	/*__shared__ float arr[blockDim.x * blockDim.y * blockDim.z];*/
	__shared__ float arr[THREADS * THREADS * THREADS];

	
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z + blockIdx.z * blockDim.z;

	//arr[threadIdx.x][threadIdx.y][threadIdx.z] = 0.0;
	value(arr, threadIdx.y, threadIdx.x, threadIdx.z) = 0.0;
	__syncthreads();

	//arr[threadIdx.x][threadIdx.y][threadIdx.z] = value(input, x, y, z);
	value(arr, threadIdx.y, threadIdx.x, threadIdx.z) = value(input, y, x, z);
	__syncthreads();

	#define BORDER 0

	/*
	float center = arr[threadIdx.x][threadIdx.y][threadIdx.z];
	float up = (threadIdx.z < (blockDim.z - 1)) ? arr[threadIdx.x][threadIdx.y][threadIdx.z + 1] : BORDER;
	float down = (threadIdx.z > 0) ? arr[threadIdx.x][threadIdx.y][threadIdx.z - 1] : BORDER;
	float west = (threadIdx.x > 0) ? arr[threadIdx.x - 1][threadIdx.y][threadIdx.z] : BORDER;
	float east = (threadIdx.x < (blockDim.x - 1)) ? arr[threadIdx.x + 1][threadIdx.y][threadIdx.z] : BORDER;
	float south = (threadIdx.y > 0) ? arr[threadIdx.x][threadIdx.y - 1][threadIdx.z] : BORDER;
	float north = (threadIdx.y < (blockDim.y - 1)) ? arr[threadIdx.x][threadIdx.y + 1][threadIdx.z] : BORDER;
	*/

	#define out(i, j, k) value(output, i, j, k)
	#define in(i, j, k) value(arr, i, j, k)


	int i = threadIdx.x;
	int j = threadIdx.y;
	int k = threadIdx.z;

	float res = in(i, j, k + 1) + in(i, j, k - 1) + in(i, j + 1, k) + in(i, j - 1, k) + in(i + 1, j, k) + in(i - 1, j, k) - 6 * in(i, j, k);
	res = Clamp(res, 0, MAX_VAL);
	out(y, x, z) = res;

}

static void launch_stencil(float *deviceOutputData, float *deviceInputData,
	int width, int height, int depth) {
	//@@ INSERT CODE HERE

	dim3 block(THREADS, THREADS, THREADS);
	dim3 grid(ceil(width/block.x), ceil(height/block.y), ceil(depth/block.z));

	stencil <<<grid, block >>> (deviceOutputData, deviceInputData, width, height, depth);
}

int main(int argc, char *argv[]) {
	wbArg_t arg;
	int width;
	int height;
	int depth;
	char *inputFile;
	wbImage_t input;
	wbImage_t output;
	float *hostInputData;
	float *hostOutputData;
	float *deviceInputData;
	float *deviceOutputData;

	arg = wbArg_read(argc, argv);

	inputFile = wbArg_getInputFile(arg, 0);

	input = wbImport(inputFile);

	width = wbImage_getWidth(input);
	height = wbImage_getHeight(input);
	depth = wbImage_getChannels(input);

	output = wbImage_new(width, height, depth);

	hostInputData = wbImage_getData(input);
	hostOutputData = wbImage_getData(output);

	wbTime_start(GPU, "Doing GPU memory allocation");
	cudaMalloc((void **)&deviceInputData, width * height * depth * sizeof(float));
	cudaMalloc((void **)&deviceOutputData, width * height * depth * sizeof(float));
	wbTime_stop(GPU, "Doing GPU memory allocation");

	wbTime_start(Copy, "Copying data to the GPU");
	cudaMemcpy(deviceInputData, hostInputData, width * height * depth * sizeof(float), cudaMemcpyHostToDevice);
	wbTime_stop(Copy, "Copying data to the GPU");

	wbTime_start(Compute, "Doing the computation on the GPU");
	launch_stencil(deviceOutputData, deviceInputData, width, height, depth);
	wbTime_stop(Compute, "Doing the computation on the GPU");

	wbTime_start(Copy, "Copying data from the GPU");
	cudaMemcpy(hostOutputData, deviceOutputData, width * height * depth * sizeof(float), cudaMemcpyDeviceToHost);
	wbTime_stop(Copy, "Copying data from the GPU");

	wbSolution(arg, output);

	cudaFree(deviceInputData);
	cudaFree(deviceOutputData);

	wbImage_delete(output);
	wbImage_delete(input);

	return 0;
}
