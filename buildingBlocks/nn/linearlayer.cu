#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <cstdlib>
#include <cmath>
#include "../Tensor.cuh"
#include <memory>


using namespace std;

class LinearLayer {
	private:
		int input_features;
		int output_features;

		// Number of rows equal to the number of neurons in the previous layer.
		// Number of columns equal tot he number of neurons in the next layer.
		unique_ptr<Tensor> weights;
    	unique_ptr<Tensor> bias;
		
		void initialize_values() {
			float k = 1 / input_features; 
			random_device rd;  // a seed source for the random number engine
    		mt19937 gen(rd());
			// First we intialize the weights based on a uniform distribution 
			// X ~ U(-√(k), √(k))
			uniform_real_distribution<float> distr(-sqrt(k), sqrt(k));
			vector<float> weightsTemp(output_features * input_features);
			for (int i = 0; i < output_features; i++) {
				float generated = distr(gen);
				for(int j = 0; j < input_features; j++) {
					weightsTemp[i * input_features + j] = generated;
				}
			}

			weights = make_unique<Tensor>(weightsTemp, {input_features, output_features});
			vector<float> biasTemp(output_features);

			for (int i = 0; i < output_features; i++) {
				biasTemp[i] = distr(gen);
			}

			bias = make_unique<Tensor>(biasTemp, {output_features});
		}

		
	public: 
		LinearLayer(int input_features, int output_features) 
		: input_features(input_features), output_features(output_features) { 
			
		}

		


	
};

// linear layer cuda implementation
__global__ void linear_layer_forward(float* x, float* w, float* b, float* c, float alpha, int size){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < size)
		c[i] = w[i] * x[i] + b[i];
}
	
__global__ void linear_layer_backward(){
	return ;
}

