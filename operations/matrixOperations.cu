#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include "matrixOperations.cuh"

// Vector Addition
__global__ void addVectors(float* a, float* b, float *c, int size){
	int i = threadIdx.x;
	if (i < size)
		c[i] = a[i] + b[i];
}


// Matrix Addition
__global__ void addMatrices(float* a, float *b, float *c, int n, int m){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j < m && i < n) {
        c[i * n + j]= a[i * n + j] + b[i * n + j];
    }
}

namespace MatrixOperations {
	void vector_addition(float* a, float* b, float* c, int n){		
		float *d_a, *d_b, *d_c;
		cudaMalloc(&d_a, n*sizeof(float));
		cudaMalloc(&d_b, n*sizeof(float));
		cudaMalloc(&d_c, n*sizeof(float));
		cudaMemcpy(d_a, a, n*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_b, b, n*sizeof(float), cudaMemcpyHostToDevice);
		addVectors<<<1, n>>>(d_a, d_b, d_c, n);
		cudaMemcpy(c, d_c, n*sizeof(float), cudaMemcpyDeviceToHost);
	} 

	void matrix_addition(float* a, float* b, float* c, int n, int m) {
		float *d_a, *d_b, *d_c;
		cudaMalloc(&d_a, n*m*sizeof(float));
		cudaMalloc(&d_b, n*m*sizeof(float));
		cudaMalloc(&d_c, n*m*sizeof(float));
		cudaMemcpy(d_a, a, n*m*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_b, b, n*m*sizeof(float), cudaMemcpyHostToDevice);
		dim3 blockSize(16, 16); 
		dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y);
		addMatrices<<<gridSize, blockSize>>>(d_a, d_b, d_c, n, m);
		cudaMemcpy(c, d_c, n*m*sizeof(float), cudaMemcpyDeviceToHost);
	}

}