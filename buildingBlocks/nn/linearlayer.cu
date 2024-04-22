#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <vector>


class LinearLayer {
	private:
		int input_features;
		int output_features;

		// Number of rows equal to the number of neurons in the previous layer.
		// Number of columns equal tot he number of neurons in the next layer.
		vector<vector<float>> weights;
		vector<float> bias;
	public: 
		LinearLayer(int input_features, int output_features) 
		: input_features(input_features), output_features(output_features){
			
		}
	
}
// linear layer cuda implementation
__global__ void linear_layer_forward(float* x, float* w, float* b, float* c, float alpha, int size){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < size)
		c[i] = w[i] * x[i] + b[i];
}
	
__global__ void linear_layer_backward(){
	
}

