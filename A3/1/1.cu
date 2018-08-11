#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include <iostream>
//#include <bits/stdc++.h>
#include <stdio.h>
//#include <iostream>
#include "wb.h"

#define THREADS 3000


__global__ void RGB2GRAY(float *inImage, float *outImage, int width, int height) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	
//	std::cout<<"Outside\n";
//	if (x < width && y < height) {
//		std::cout<<"Inside\n";
		int idx = x * width + y;

		float r = inImage[3 * idx];
		float g = inImage[3 * idx + 1];
		float b = inImage[3 * idx + 2];

		outImage[idx] = (0.21*r + 0.71*g + 0.07*b);
		//outImage[idx+1] = 0.0;
		//outImage[idx+2] = 0.0;
//	}

}

//@@ define error checking macro here.

#define errCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      printErrorLog(ERROR, "Failed to run stmt ", #stmt);                         \
      printErrorLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ INSERT CODE HERE

int main(int argc, char *argv[]) {

	int imageChannels;
	int imageWidth;
	int imageHeight;
	char *inputImageFile;
	wbImage_t inputImage;
	wbImage_t outputImage;
	float *hostInputImageData;
	float *hostOutputImageData;
	float *deviceInputImageData;
	float *deviceOutputImageData;

	/* parse the input arguments */

	wbArg_t args = wbArg_read(argc, argv);

	inputImageFile = wbArg_getInputFile(args, 0);

	inputImage = wbImport(inputImageFile);

	imageWidth = wbImage_getWidth(inputImage);
	imageHeight = wbImage_getHeight(inputImage);
	// For this lab the value is always 3
	imageChannels = wbImage_getChannels(inputImage);

	// Since the image is monochromatic, it only contains one channel
	outputImage = wbImage_new(imageWidth, imageHeight, 1);

	hostInputImageData = wbImage_getData(inputImage);
	hostOutputImageData = wbImage_getData(outputImage);

	wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

	wbTime_start(GPU, "Doing GPU memory allocation");
	cudaMalloc((void **)&deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
	cudaMalloc((void **)&deviceOutputImageData, imageWidth * imageHeight * sizeof(float));
	wbTime_stop(GPU, "Doing GPU memory allocation");

	wbTime_start(Copy, "Copying data to the GPU");
	cudaMemcpy(deviceInputImageData, hostInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice);
	wbTime_stop(Copy, "Copying data to the GPU");

	///////////////////////////////////////////////////////
	wbTime_start(Compute, "Doing the computation on the GPU");
	
	dim3 blockSize(THREADS, THREADS, 1);
	dim3 gridSize(ceil((float)imageWidth/ (float)THREADS), ceil((float)imageHeight / (float)THREADS), 1);

	RGB2GRAY << <blockSize, gridSize >> > (deviceInputImageData, deviceOutputImageData, imageWidth, imageHeight);


	wbTime_stop(Compute, "Doing the computation on the GPU");

	///////////////////////////////////////////////////////
	wbTime_start(Copy, "Copying data from the GPU");
	cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageWidth * imageHeight * sizeof(float), cudaMemcpyDeviceToHost);
	wbTime_stop(Copy, "Copying data from the GPU");

	wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

	wbSolution(args, outputImage);

	cudaFree(deviceInputImageData);
	cudaFree(deviceOutputImageData);

	wbImage_delete(outputImage);
	wbImage_delete(inputImage);

	return 0;
}
