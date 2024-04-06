#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include "matrixOperations.cuh"
#include <cmath>


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
__global__ void dotProduct(float* a, float* b, float& c, int n){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < n) {
		c += a[i] + b[i];
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

	void matrix_scaling(float* a, float* b, float scalar, int n, int m) {
		float *d_a, *d_b;
		cudaMalloc(&d_a, n*m*sizeof(float));
		cudaMalloc(&d_b, n*m*sizeof(float));
		cudaMemcpy(d_a, a, n*m*sizeof(float), cudaMemcpyHostToDevice);
		dim3 blockSize(16, 16);
		dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y);
		scaleMatrix<<<gridSize, blockSize>>>(d_a, d_b, scalar, n, m);
		cudaMemcpy(b, d_b, n*m*sizeof(float), cudaMemcpyDeviceToHost);
	}

	void dot_product(float* a, float *b, float& c, int n){
		float *d_a, *d_b, *d_c;
		cudaMalloc(&d_a, n*sizeof(float));
		cudaMalloc(&d_b, n*sizeof(float));
		cudaMalloc(&d_c, sizeof(float));
		cudaMemcpy(d_a, a, n*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_b, b, n*sizeof(float), cudaMemcpyHostToDevice);
		addVectors<<<1, n>>>(d_a, d_b, d_c, n);
	}

	

	void matrix_multiplication(float* a, float* b, float* c, int n, int m, int p) {
		float *d_a, *d_b, *d_c;
		cudaMalloc(&d_a, n*m*sizeof(float));
		cudaMalloc(&d_b, n*p * sizeof(float));
		cudaMalloc(&d_c, m*p*sizeof(float));
        cudaMemcpy(d_a, a, n*m*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, n*p*sizeof(float), cudaMemcpyHostToDevice);
		dim3 blockSize(16, 16);
		dim3 gridDim((p+blockSize.x - 1)/blockSize.x, (m + blockSize.y-1)/blockSize.y);
		multiplyMatrices<<<gridDim, blockSize>>>(d_a, d_b, d_c, n, m, p);
		cudaMemcpy(c, d_c, m*p*sizeof(float), cudaMemcpyDeviceToHost);
	}

	// void matrix_transpose(float *a, float *b, int n, int m) {
	// 	float *d_a, *d_b;
	// 	cudaMalloc(&d_a, n*m*sizeof(float));
	// 	cudaMalloc(&d_b, n*m*sizeof(float));
	// 	cudaMemcpy(d_a, a, n*m*sizeof(float), cudaMemcpyHostToDevice);
	// 	dim3 blockSize(16, 16);
	// 	dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y);
	// 	matrixTranspose<<<gridSize, blockSize>>>(d_a, d_b, n, m);
	// 	cudaMemcpy(b, d_b, n*m*sizeof(float), cudaMemcpyDeviceToHost);
	// }


}