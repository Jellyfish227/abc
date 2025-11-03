#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void compatible_check_kernel(int* pOutputID0, int* pOutputID1, int size, int* result) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	
	// Uncomment this for experiment for large datasets, small datasets can be harmful to performance
	// if (*result == 1) {
	// 	return;
	// }
	
	if (tid < size) {
		if (pOutputID0[tid] & pOutputID1[tid])
			atomicOr(result, 1);
	}
}

extern "C" int launch_kernel(int* pOutputID0, int* pOutputID1, int size) {
	int threadsPerBlock = 256;
	int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

	// Allocate result flag in device memory and initialize to 0
	int* d_result;
	cudaMalloc(&d_result, sizeof(int));
	cudaMemset(d_result, 0, sizeof(int));

	compatible_check_kernel<<<blocksPerGrid, threadsPerBlock>>>(pOutputID0, pOutputID1, size, d_result);
	
	cudaDeviceSynchronize();

	int h_result = 0;
	cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(d_result);

	return h_result;
}