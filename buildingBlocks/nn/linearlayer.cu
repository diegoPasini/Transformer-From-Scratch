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
			for (int i = 0; i < output_features * input_features; i++) {
				float generated = distr(gen);
				weightsTemp[i] = generated;
				
			}
			vector<int> dims = {output_features, input_features};
			weights = make_unique<Tensor>(weightsTemp, dims, "cuda");
			vector<float> biasTemp(output_features);

			for (int i = 0; i < output_features; i++) {
				biasTemp[i] = distr(gen);
			}
			vector<int> dim = {output_features, 1};
			bias = make_unique<Tensor>(biasTemp, dim, "cuda");
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
			//cout << "Weigts: " << (x).toString() << endl;
			//cout << "Device :" << x.getDevice() << endl;
			//x.transpose();
			//cout << "Weigts: " << (*weights).toString() << endl;

			x = (*weights * x);
			cout << "Device :" << (*weights).getDevice() << endl;

			//cout << x.toString() << endl;
			//cout << (*bias).toString() << endl;
			x = x + *bias;
			//this->outputs = x;
			return x;
		}

		Tensor backward(Tensor gammaPrev) {
			//inputs.reshape({input_features, 1});
			//inputs.transpose();
			// cout << "Input: " << inputs.toString() << endl;
			// cout << "Inputs Shape: " << inputs.getDimensionsString() << endl;
			cout << "Gamma Shape: " << gammaPrev.getDimensionsString() << endl;
			cout << "GammaPrev: " << gammaPrev.toString() << endl;
			inputs.transpose();
			Tensor newGamma = gammaPrev * inputs;
			cout << "New Gamma Tranpose: " << newGamma.toString() << endl;
			Tensor x = -1.0f * learning_rate * newGamma;
			//x.transpose();
			cout << "X: " << x.toString() << endl;
			cout << "X Device " << (x).getDevice() << endl;

			cout << "X: " << x.toString() << endl;
			//vector<float> newValues(x.getTotalValues() * output_features);
			//vector<float> originalValues = x.getValues();
			// for(int i = 0; i < x.getTotalValues(); i++) {
			// 	for (int j = 0; j < output_features; j++) {
			// 		newValues[i * output_features + j] = originalValues[i];
			// 	}
			// }
			//Tensor weightsBroadcast(newValues, {input_features, output_features}, "cuda");
			//cout << "Weights: " << (*weights).toString() << endl;
			//(*weights).transpose();
			cout << "Transposed Weights: " << (*weights).toString() << endl;
			cout << "Return Weights Device: " << (*weights).getDevice() << endl;

			//cout << "Transpose Weights Shape " << (*weights).getDimensionsString() << endl;
			*weights = *weights + x; 
			cout << "Return Weights Device: " << (*weights).getDevice() << endl;

			//(*weights).transpose();
			//cout << "X Shape " << (x).getDimensionsString() << endl;

			//cout << "New Weights: " << (*weights).toString() << endl;
			//cout << "Biases: " << (*bias).toString() << endl;
			if (x.getTotalValues() != 1) {
				vector<float> biasValues(x.getTotalValues() / input_features);
				for (int i = 0; i < output_features; i++) {
					for (int j = 0; j < input_features; j++) {
						biasValues[i] += x[{i, j}]; 
					}
				}
				Tensor d_bias(biasValues, {output_features, 1}, "cuda"); 
				*bias = *bias + (d_bias);

			} else {
				*bias = *bias + (x);
			}

			cout << "New Biases: " << (*bias).toString() << endl;
			cout << "Return Weights Device: " << (*weights).getDevice() << endl;

			Tensor returnWeights = *weights;
			cout << "Return Weights Device: " << returnWeights.getDevice() << endl;

			returnWeights.transpose();
			cout << "Return Weights Device: " << returnWeights.getDevice() << endl;
			cout << "Return GamaPrev Device: " << gammaPrev.getDevice() << endl;

			Tensor outputGradient = returnWeights * gammaPrev;
			
			cout << "Output Gradient: " << (outputGradient).toString() << endl;
			inputs.transpose();
			return outputGradient;
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

