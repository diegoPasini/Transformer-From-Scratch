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
    // learnable params
    float gamma = 1.0;
    float beta = 1.0;

public:
    Layernorm() {}


    vector<vector<float>> forward(const vector<vector<float>>& inputs) {
        int batch_size = inputs.size();
        int feature_size = inputs[0].size();
        
        vector<vector<float>> normalized_inputs(batch_size, vector<float>(feature_size));
        vector<float> mean(feature_size, 0.0f);
        vector<float> variance(feature_size, 0.0f);
        
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
        
    }
};