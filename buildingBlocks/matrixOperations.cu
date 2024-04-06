#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include "matrixOperations.cuh"
#include <cmath>

using namespace std;
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

// Matrix Scaling
__global__ void scaleMatrix(float* a, float* b, float scalar, int n, int m){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (j < m && i < n) {
        b[i * n + j]= a[i * n + j] * scalar;
    }
}

// Dot Product For 1 Dimensions Arrays:
__global__ void dotProduct(float* a, float* b, float* c, int n){
	int i = threadIdx.x;
	if(i < n) {
		c[0] = c[0] + a[i] * b[i];
	}
}


// Matrix Multiplication
// Try Strassen Algorithm????
__global__ void multiplyMatrices(float* a, float* b, float* c, int n, int m, int p){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	float temp = 0.0f;
	if (j < m && i < p) {
        for(int k = 0; k < n; k++) {
			temp += a[j * n + k] * b[k * p + i];
		}
		c[j * p + i] = temp;
    }
}


namespace MatrixOperations {
	void vector_addition(float* a, float* b, float* c, int n){		
		addVectors<<<1, n>>>(a, b, c, n);
		cudaDeviceSynchronize();
	} 

	void matrix_addition(float* a, float* b, float* c, int n, int m) {
		dim3 blockSize(16, 16); 
		dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y);
		addMatrices<<<gridSize, blockSize>>>(a, b, c, n, m);
		cudaDeviceSynchronize();
	}

	void matrix_scaling(float* a, float* b, float scalar, int n, int m) {
		dim3 blockSize(16, 16);
		dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y);
		scaleMatrix<<<gridSize, blockSize>>>(a, b, scalar, n, m);
		cudaDeviceSynchronize();

	}

	void dot_product(float* a, float *b, float* c, int n){
		addVectors<<<1, n>>>(a, b, c, n);
		cudaDeviceSynchronize();
	}

	void matrix_multiplication(float* a, float* b, float* c, int n, int m, int p) {
		dim3 blockSize(16, 16);
		dim3 gridDim((p+blockSize.x - 1)/blockSize.x, (m + blockSize.y-1)/blockSize.y);
		multiplyMatrices<<<gridDim, blockSize>>>(a, b, c, n, m, p);
		cudaDeviceSynchronize();
	}

}