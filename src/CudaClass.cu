#include "CudaClass.h"
#include <cuda_runtime.h>
#include <iostream>

// CUDA kernel function to add two vectors
__global__ void addVectorsKernel(int* a, int* b, int* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

CudaClass::CudaClass(int size) : size(size), h_a(nullptr), h_b(nullptr), h_c(nullptr), d_a(nullptr), d_b(nullptr), d_c(nullptr) {
    allocateHostMemory();

    // Initialize host input vectors
    for (int i = 0; i < size; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }
    allocateDeviceMemory();
}

CudaClass::~CudaClass() {
    freeMemory();
}

void CudaClass::allocateHostMemory() {
    // Allocate host memory
    h_a = (int*)malloc(size * sizeof(int));
    h_b = (int*)malloc(size * sizeof(int));
    h_c = (int*)malloc(size * sizeof(int));
}

void CudaClass::allocateDeviceMemory() {
	// Allocate device memory
	cudaMalloc((void**)&d_a, size * sizeof(int));
	cudaMalloc((void**)&d_b, size * sizeof(int));
	cudaMalloc((void**)&d_c, size * sizeof(int));

	// Copy host input vectors to device
	cudaMemcpy(d_a, h_a, size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, size * sizeof(int), cudaMemcpyHostToDevice);
}

void CudaClass::freeMemory() {
    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);
}

void CudaClass::addVectors() {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    addVectorsKernel << <numBlocks, blockSize >> > (d_a, d_b, d_c, size);

    // Copy device output vector to host
    cudaMemcpy(h_c, d_c, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Verify the result
    for (int i = 0; i < size; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            std::cerr << "Error at index " << i << ": " << h_c[i] << " != " << h_a[i] + h_b[i] << std::endl;
        }
    }

    std::cout << "Addition successful!" << std::endl;
}