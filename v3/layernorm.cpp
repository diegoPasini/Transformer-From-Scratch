#include <vector>
#include <random>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include "functions.h"
#include "matrix_operations.h"


using namespace std;

class Layernorm {
private:
    float gamma = 1.0;
    float beta = 1.0;

    vector<float> mean;
    vector<float> variance;

public:
    Layernorm() {}

    vector<vector<float>> forward(const vector<vector<float>>& inputs) {
        int batch_size = inputs.size();
        int feature_size = inputs[0].size();
        
        mean.assign(feature_size, 0.0f);
        variance.assign(feature_size, 0.0f);
        vector<vector<float>> normalized_inputs(batch_size, vector<float>(feature_size));
        
        for (int j = 0; j < feature_size; ++j) {
            for (int i = 0; i < batch_size; ++i) {
                mean[j] += inputs[i][j];
            }
            mean[j] /= batch_size;
        }
        
        for (int j = 0; j < feature_size; ++j) {
            for (int i = 0; i < batch_size; ++i) {
                variance[j] += pow(inputs[i][j] - mean[j], 2);
            }
            variance[j] /= batch_size;
        }
        
        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < feature_size; ++j) {
                normalized_inputs[i][j] = (inputs[i][j] - mean[j]) / sqrt(variance[j] + 1e-8);
                normalized_inputs[i][j] = gamma * normalized_inputs[i][j] + beta;
            }
        }
        
        return normalized_inputs;
    }
    
    vector<vector<float>> backward(const vector<vector<float>>& d_outputs, float learning_rate, int t) {
        int batch_size = d_outputs.size();
        int feature_size = d_outputs[0].size();

        vector<vector<float>> d_inputs(batch_size, vector<float>(feature_size));
        vector<float> d_mean(feature_size, 0.0f);
        vector<float> d_variance(feature_size, 0.0f);
        vector<float> d_gamma(feature_size, 0.0f);
        vector<float> d_beta(feature_size, 0.0f);

        for (int j = 0; j < feature_size; ++j) {
            for (int i = 0; i < batch_size; ++i) {
                d_mean[j] += d_outputs[i][j];
            }
            d_mean[j] /= batch_size;
        }

        for (int j = 0; j < feature_size; ++j) {
            for (int i = 0; i < batch_size; ++i) {
                d_variance[j] += pow(d_outputs[i][j] - mean[j], 2);
            }
            d_variance[j] /= batch_size;
        }

        for (int j = 0; j < feature_size; ++j) {
            for (int i = 0; i < batch_size; ++i) {
                d_gamma[j] += d_outputs[i][j] * (d_outputs[i][j] - mean[j]) / sqrt(variance[j] + 1e-8);
                d_beta[j] += d_outputs[i][j];
            }
        }

        for (int j = 0; j < feature_size; ++j) {
            gamma -= learning_rate * d_gamma[j];
            beta -= learning_rate * d_beta[j];
        }

        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < feature_size; ++j) {
                d_mean[j] += d_outputs[i][j];
                d_variance[j] += d_outputs[i][j] * (d_outputs[i][j] - mean[j]);
            }
        }

        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < feature_size; ++j) {
                d_inputs[i][j] = (1.0 / batch_size) * gamma * (d_outputs[i][j] - mean[j]) / sqrt(variance[j] + 1e-8) 
                                 - (1.0 / batch_size) * d_mean[j] 
                                 - (2.0 / batch_size) * (d_outputs[i][j] - mean[j]) * d_variance[j];
            }
        }

        return d_inputs;
    }
};