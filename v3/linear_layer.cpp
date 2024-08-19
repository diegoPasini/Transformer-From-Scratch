#include <vector>
#include <random>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include "functions.h"
#include "matrix_operations.h"

using namespace std;

class LinearLayer {
private:
    int input_features;
    int output_features;
    bool use_bias;
    vector<vector<float>> weights;
    vector<float> bias;
    vector<vector<float>> inputs; 
    vector<vector<float>> outputs;

    // Adam optimizer parameters
    vector<vector<float>> m_weights;
    vector<vector<float>> v_weights;
    vector<float> m_bias;
    vector<float> v_bias;
    float beta1;
    float beta2;
    float epsilon;

    void initialize_weights() {
        float k = sqrt(6.0f / (input_features + output_features));
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<float> distr(-k, k);

        weights.resize(output_features, vector<float>(input_features));
        m_weights.resize(output_features, vector<float>(input_features, 0.0));
        v_weights.resize(output_features, vector<float>(input_features, 0.0));
        for (int i = 0; i < output_features; ++i) {
            for (int j = 0; j < input_features; ++j) {
                weights[i][j] = distr(gen);
            }
        }

        if (use_bias) {
            bias.resize(output_features);
            m_bias.resize(output_features, 0.0);
            v_bias.resize(output_features, 0.0);
            for (int i = 0; i < output_features; ++i) {
                bias[i] = distr(gen);
            }
        }
    }

public:
    LinearLayer(int in_features, int out_features, bool bias = true)
        : input_features(in_features), output_features(out_features), use_bias(bias), beta1(0.9), beta2(0.999), epsilon(1e-8) {
        initialize_weights();
    }

    vector<vector<float>> forward(const vector<vector<float>>& batch_inputs) {
        int batch_size = batch_inputs.size();
        inputs = batch_inputs;  // Save the entire batch of inputs
        vector<vector<float>> batch_outputs(batch_size, vector<float>(output_features));

        for (int b = 0; b < batch_size; ++b) {
            for (int i = 0; i < output_features; ++i) {
                batch_outputs[b][i] = use_bias ? bias[i] : 0.0f;
                for (int j = 0; j < input_features; ++j) {
                    batch_outputs[b][i] += weights[i][j] * batch_inputs[b][j];
                }
            }
        }
        return batch_outputs;
    }

    vector<vector<float>> backward(const vector<vector<float>>& d_outputs, float learning_rate, int t) {
        int batch_size = d_outputs.size();
        vector<vector<float>> d_weights(output_features, vector<float>(input_features, 0.0f));
        vector<vector<float>> d_inputs(batch_size, vector<float>(input_features, 0.0f));
        vector<float> d_bias(output_features, 0.0f);

        for (int b = 0; b < batch_size; ++b) {
            for (int i = 0; i < output_features; ++i) {
                for (int j = 0; j < input_features; ++j) {
                    d_weights[i][j] += d_outputs[b][i] * inputs[b][j];  
                    d_inputs[b][j] += d_outputs[b][i] * weights[i][j];  
                }
                if (use_bias) {
                    d_bias[i] += d_outputs[b][i];  
                }
            }
        }

        for (int i = 0; i < output_features; ++i) {
            for (int j = 0; j < input_features; ++j) {
                d_weights[i][j] /= batch_size;
            }
            if (use_bias) {
                d_bias[i] /= batch_size;
            }
        }

        float beta1_t = 1 - pow(beta1, t);
        float beta2_t = 1 - pow(beta2, t);

        for (int i = 0; i < output_features; ++i) {
            for (int j = 0; j < input_features; ++j) {
                m_weights[i][j] = beta1 * m_weights[i][j] + (1 - beta1) * d_weights[i][j];
                v_weights[i][j] = beta2 * v_weights[i][j] + (1 - beta2) * d_weights[i][j] * d_weights[i][j];

                float m_hat = m_weights[i][j] / (1 - pow(beta1, t));
                float v_hat = v_weights[i][j] / (1 - pow(beta2, t));

                weights[i][j] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
            }

            if (use_bias) {
                m_bias[i] = beta1 * m_bias[i] + (1 - beta1) * d_bias[i];
                v_bias[i] = beta2 * v_bias[i] + (1 - beta2) * d_bias[i] * d_bias[i];

                float m_hat_bias = m_bias[i] / (1 - pow(beta1, t));
                float v_hat_bias = v_bias[i] / (1 - pow(beta2, t));

                bias[i] -= learning_rate * m_hat_bias / (sqrt(v_hat_bias) + epsilon);
            }
        }

        return d_inputs; 
    }

    void print_weights() const {
        for (const vector<float>& row : weights) {
            for (float val : row) {
                cout << val << " ";
            }
            cout << endl;
        }
    }

    void print_bias() const {
        if (use_bias) {
            for (float val : bias) {
                cout << val << " ";
            }
            cout << endl;
        } else {
            cout << "Bias is not used in this layer." << endl;
        }
    }
};