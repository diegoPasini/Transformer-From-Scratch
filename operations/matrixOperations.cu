#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include "matrixOperations.cuh"

// Vector Addition
__global__ void addVectors(float* a, float* b, float *c, int size){
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

namespace MatrixOperations {
	void vector_addition(float* input1, float* input2, float* output_vector, int size){		
		int N = size;
		float *d_a, *d_b, *d_c;
		cudaMalloc(&d_a, N*sizeof(float));
		cudaMalloc(&d_b, N*sizeof(float));
		cudaMalloc(&d_c, N*sizeof(float));

		// Copy Data to Device
		cudaMemcpy(d_a, input1, size*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_b, input2, size*sizeof(float), cudaMemcpyHostToDevice);

		addVectors<<<1, size>>>(d_a, d_b, d_c, size);

		cudaMemcpy(output_vector, d_c, size*sizeof(float), cudaMemcpyDeviceToHost);
	} 
}