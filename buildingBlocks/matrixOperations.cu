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



// Matrix Multiplication
// Try Strassen Algorithm????
// #define TILE_WIDTH 16

// __global__ void multiplyMatrices(float *d_A, float *d_B, float *d_C, int M, int N, int P) {
//     __shared__ float tile_A[TILE_WIDTH][TILE_WIDTH];
//     __shared__ float tile_B[TILE_WIDTH][TILE_WIDTH];

//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     int col = blockIdx.x * blockDim.x + threadIdx.x;
//     float sum = 0.0f;

//     for (int k = 0; k < (N + TILE_WIDTH - 1) / TILE_WIDTH; ++k) {
//         if (row < M && k*TILE_WIDTH + threadIdx.x < N)
//             tile_A[threadIdx.y][threadIdx.x] = d_A[row*N + k*TILE_WIDTH + threadIdx.x];
//         else
//             tile_A[threadIdx.y][threadIdx.x] = 0.0;

//         if (col < P && k*TILE_WIDTH + threadIdx.y < N)
//             tile_B[threadIdx.y][threadIdx.x] = d_B[(k*TILE_WIDTH + threadIdx.y)*P + col];
//         else
//             tile_B[threadIdx.y][threadIdx.x] = 0.0;

//         __syncthreads();

//         for (int n = 0; n < TILE_WIDTH; ++n) {
//             sum += tile_A[threadIdx.y][n] * tile_B[n][threadIdx.x];
//         }
//         __syncthreads();
//     }

//     if (row < M && col < P) {
//         d_C[row*P + col] = sum;
//     }
// }
__global__ void multiplyMatrices(float *d_A, float *d_B, float *d_C, int M, int N, int P) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if(row < M && col < P) {
                float sum = 0.0f;

                // compute the dot product for each row of A and col of B
                for(int i = 0; i < N; ++i) {
                        sum += d_A[row * N + i] * d_B[i * P + col];
                }
                d_C[row * P + col] = sum;
        }
}


// Sum of elements in matrix
__global__ void sumMatrix(float* a, float aSum, int M, int N){
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < M && col < N) {
		aSum += a[row * N + col];
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

	void matrix_multiplication(float* a, float *b, float* c, int n, int m, int p){
		dim3 blockSize(16, 16);
		dim3 gridDim((p+blockSize.x - 1)/blockSize.x, (m + blockSize.y-1)/blockSize.y);
		multiplyMatrices<<<gridDim, blockSize>>>(a, b, c, m, n, p);
		cudaDeviceSynchronize();
	}

}