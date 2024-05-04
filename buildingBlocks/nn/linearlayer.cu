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


// Implementation of a linear layer WITH ADAM OPTIMIZER
class LinearLayer : public Layer {
	private:
		int input_features;
		int output_features;
		float learning_rate;
		// Number of rows equal to the number of neurons in the previous layer.
		// Number of columns equal to the number of neurons in the next layer.
		unique_ptr<Tensor> weights;
    	unique_ptr<Tensor> bias;
		Tensor inputs;
		Tensor outputs;
		unique_ptr<Tensor> V_dw;
		unique_ptr<Tensor> V_db;
		unique_ptr<Tensor> S_dw;
		unique_ptr<Tensor> S_db;

		float beta_1 = 0.9;
		float beta_2 = 0.999;
		float epsilon = 10^(-8);
		
		void initialize_weights() {
			float k = 1 / float(input_features); 
			random_device rd;  // a seed source for the random number engine
    		mt19937 gen(rd());
			// First we initialize the weights based on a uniform distribution 
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
			initialize_weights();
		}

		// Destructor
		~LinearLayer() {
		}


		Tensor forward(Tensor x) {
			this->inputs = x;
			x = (*weights * x);
			x = x + *bias;
			return x;
		}

		Tensor backward(Tensor gammaPrev) {
			inputs.transpose();
			Tensor newGamma = gammaPrev * inputs;
			Tensor dW = -1.0f * newGamma;

			Tensor db;
			if (dW.getTotalValues() != 1) {
				vector<float> biasValues(dW.getTotalValues() / input_features);
				for (int i = 0; i < output_features; i++) {
					for (int j = 0; j < input_features; j++) {
						biasValues[i] += x[{i, j}]; 
					}
				}
				db = Tensor(biasValues, {output_features, 1}, "cuda"); 

			} else {
				db = dW;
				*bias = *bias + (dW);
			}

			// Mometum
			if (V_dw) {
				*V_dw = ((beta_1) * *V_dw) + ((1 - beta_1) * dW);
				*V_db = ((beta_1) * *V_db) + ((1 - beta_1) * db);
				*S_dw = ((beta_2) * *S_dw) + ((1 - beta_2) * multiply(dW, dW));
				*S_db = ((beta_2) * *S_db) + ((1 - beta_2) * multiply(db, db));
				

			} else {
				*V_dw = 
				*V_db = 
				*S_dw = 
				*S_db = 
			}

			// RMS Prop 



			*weights = *weights + *V_dw; 
			*bias = *bias + (d_bias);

			//cout << "New Biases: " << (*bias).toString() << endl;
			//cout << "Return Weights Device: " << (*weights).getDevice() << endl;

			Tensor returnWeights = *weights;
			//cout << "Return Weights Device: " << returnWeights.getDevice() << endl;

			returnWeights.transpose();
			//cout << "Return Weights Device: " << returnWeights.getDevice() << endl;
			//cout << "Return GamaPrev Device: " << gammaPrev.getDevice() << endl;

			Tensor outputGradient = returnWeights * gammaPrev;
			
			//cout << "Output Gradient: " << (outputGradient).toString() << endl;
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

