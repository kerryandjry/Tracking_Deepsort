
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <cstdio>
#include <vector>
#include <thrust/device_vector.h>

__global__ void distance(float* x, float* y, int* result, int num) {
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num; i += blockDim.x * gridDim.x) {
		float temp = (x[i] - 1) * (x[i] - 1) + (y[i] - 1) * (y[i] - 1);
		if (temp < 1) {
			result[i] = 1;
		}
		else {
			result[i] = 0;
		}
	}
}

__global__ void sum(int* a, int* b, int num) {
	int tid = threadIdx.x;
	b[0] = 0;
	__shared__ float sData[512];
	for (int count = 0; count < ceilf(num / 512); count++) {
		if (tid + count * 512 < num) {
			sData[tid] = a[tid + count * 512];
			__syncthreads();
		}
		for (int i = 256; i > 0; i /= 2) {
			if (tid < i && tid + count * 512 < num) {
				sData[tid] = sData[tid] + sData[tid + i];
			}
			__syncthreads();
		}
		if (tid == 0) {
			b[0] += sData[0];
		}
	}
}

int main() {
	int testNum = 1<<28;
	srand((int)time(0));
	//float* xSquare = new float[testNum];
	//float* ySquare = new float[testNum];
	auto xSquare = std::make_unique<float[]>(testNum);
	auto ySquare = std::make_unique<float[]>(testNum);

	cudaMallocManaged((void**)&xSquare, testNum * sizeof(float));
	cudaMallocManaged((void**)&ySquare, testNum * sizeof(float));

	for (int i = 0; i < testNum; ++i) {
		xSquare[i] = rand() % 10000 * 1.0 / 10000;
		ySquare[i] = rand() % 10000 * 1.0 / 10000;
	}
	cudaDeviceSynchronize();

	int threadNum = 1024;
	int blockNum = 512;
	int* resultGpu;
	cudaMalloc(&resultGpu, testNum * sizeof(int));
	distance <<< blockNum, threadNum >>> (xSquare.get(), ySquare.get(), resultGpu, testNum);
	cudaDeviceSynchronize();
	int* result = new int[testNum];
	cudaMemcpy(result, resultGpu, testNum * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < 10; ++i) {
		printf("(%f, %f) -> %d\n", 1.0 - xSquare[i], 1.0 - ySquare[i], result[i]);
	}
	int* bGpu;
	cudaMalloc(&bGpu, 1 * sizeof(int));
	sum <<< 1, 512 >>> (resultGpu, bGpu, testNum);

	int b[1];
	cudaMemcpy(b, bGpu, 1 * sizeof(int), cudaMemcpyDeviceToHost);
	printf("b: %d\n", b[0]);
	printf("PI: %f\n", b[0] * 4.0 / testNum);
	return 0;
}
