#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <cmath>

class ReLU {
	public: 
	// relu cuda forward implementation
	__global__ void relu_forward(float* a, float* b, float alpha, int size){
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		if (i < size)
			b[i] = fmaxf(alpha * a[i], b[i]);
	}	
}

