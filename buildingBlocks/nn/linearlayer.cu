#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <cstdlib>
#include <cmath>
#include "../Tensor.cuh"
#include "layer.h"

#include <memory>

using namespace std;

class LinearLayer : public Layer {
	private:
		int input_features;
		int output_features;
		float learning_rate;
		// Number of rows equal to the number of neurons in the previous layer.
		// Number of columns equal tot he number of neurons in the next layer.
		unique_ptr<Tensor> weights;
    	unique_ptr<Tensor> bias;
		Tensor inputs;
		Tensor outputs;
		
		void intialize_weights() {
			float k = 1 / float(input_features); 
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
			vector<int> dims = {input_features, output_features};
			weights = make_unique<Tensor>(weightsTemp, dims, string("cuda"));
			vector<float> biasTemp(output_features);

			for (int i = 0; i < output_features; i++) {
				biasTemp[i] = distr(gen);
			}
			vector<int> dim = {output_features, 1};
			bias = make_unique<Tensor>(biasTemp, dim, string("cuda"));
		}


	public: 
		LinearLayer(int input_features, int output_features, float learning_rate) 
		: learning_rate(learning_rate) { 
			this->input_features = input_features;
			this->output_features = output_features;
			intialize_weights();
		}

		// Destructor
		~LinearLayer() {
		}


		Tensor forward(Tensor x) {
			this->inputs = x;
			x = (*weights * x);
			//cout << x.toString() << endl;
			//cout << (*bias).toString() << endl;
			x = x + *bias;
			//this->outputs = x;
			return x;
		}

		Tensor backward(Tensor gammaPrev) {
			//cout << "Value: " << (-1.0f * learning_rate * multiply(gammaPrev, inputs)).toString() << endl;
			Tensor newGamma = multiply(gammaPrev, inputs);
			Tensor x = -1.0f * learning_rate * newGamma;
			vector<float> newValues(x.getTotalValues() * output_features);
			vector<float> originalValues = x.getValues();
			for(int i = 0; i < x.getTotalValues(); i++) {
				for (int j = 0; j < output_features; j++) {
					newValues[i * output_features + j] = originalValues[i];
				}
			}
			Tensor weightsBroadcast(newValues, {input_features, output_features}, "cuda");
			*weights = *weights + weightsBroadcast; 
			*bias = *bias + (-1.0f * learning_rate * gammaPrev);
			return newGamma;
		}

		string toStringWeights() {
			return (*weights).toString();
		}

		string toStringBiases() {
			return (*bias).toString();
		}


};

// // linear layer cuda implementation
// __global__ void linear_layer_forward(float* x, float* w, float* b, float* c, float alpha, int size){
// 	int i = blockDim.x * blockIdx.x + threadIdx.x;exit
// 	if (i < size)
// 		c[i] = w[i] * x[i] + b[i];
// }
	
// __global__ void linear_layer_backward(float){
// 	return ;
// }

