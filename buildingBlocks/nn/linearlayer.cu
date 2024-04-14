#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <cmath>


// linear layer cuda implementation
__global__ void linear_layer_forward(float* x, float* w, float* b, float* c, float alpha, int size){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < size)
		c[i] = w[i] * x[i] + b[i];
}
	

