
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CUDA_equalization.h"

__global__ void calculate_Min_Max(unsigned char* Image, int channels, int* min, int* max);
__global__ void histogram_equalization(unsigned char* Image, int channels, int* min, int* max);
__device__ int New_Pixel_Value(int value, int min, int max);

void Histogram_equalization_cuda(unsigned char* Image, int Height, int Width, int channels) {
	unsigned char* Dev_image = NULL;
	int* Dev_min = NULL;
	int* Dev_max = NULL;

	cudaMalloc((void**)&Dev_image, Height * Width * channels); 
	cudaMalloc((void**)&Dev_min, channels * sizeof(int));
	cudaMalloc((void**)&Dev_max, channels * sizeof(int));

	int min[3] = { 255,255,255 };
	int max[3] = { 0,0,0 };

	cudaMemcpy(Dev_image, Image, Height * Width * channels, cudaMemcpyHostToDevice);
	cudaMemcpy(Dev_min, min, sizeof(int) * channels, cudaMemcpyHostToDevice);
	cudaMemcpy(Dev_max, max, sizeof(int) * channels, cudaMemcpyHostToDevice);

	dim3 Grid_Image(Width, Height);
	calculate_Min_Max << <Grid_Image, 1 >> > (Dev_image, channels, Dev_min, Dev_max);
	histogram_equalization << <Grid_Image, 1 >> > (Dev_image, channels, Dev_min, Dev_max);

	cudaMemcpy(Image, Dev_image, Height * Width * channels, cudaMemcpyDeviceToHost);
	 
	cudaFree(Dev_image);
}

__global__ void calculate_Min_Max(unsigned char* Image, int channels, int* min, int* max){
	int x = blockIdx.x;
	int y = blockIdx.y;

	int Image_idx = (x + y * gridDim.x)*channels;
	 
	for (int i = 0; i < channels; i++) {
		atomicMin(&min[i], Image[Image_idx + i]);
		atomicMax(&max[i], Image[Image_idx + i]);
	}
}
__global__ void  histogram_equalization(unsigned char* Image, int channels, int* min, int* max) {
	int x = blockIdx.x;
	int y = blockIdx.y;

	int Image_idx = (x + y * gridDim.x) * channels;
	for (int i = 0; i < channels; i++) {
		Image[Image_idx + i] = New_Pixel_Value(Image[Image_idx+i], min[i], max[i]);
	}
}

__device__ int New_Pixel_Value(int value, int min, int max) {
	int target_min = 0;
	int target_max = 255;

	return (target_min + (value - min) * (int)((target_max - target_min) / (max - min)));
}
