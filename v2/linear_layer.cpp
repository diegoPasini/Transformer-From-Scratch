#include <vector>
#include <random>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include "functions.cpp"
#include "matrix_operations.cpp"

using namespace std;

class LinearLayer {
private:
    int input_features;
    int output_features;
    vector<vector<float> > weights;
    vector<float> bias;
    vector<float> inputs;
    vector<float> outputs;

    // Adam optimizer parameters
    vector<vector<float> > m_weights;
    vector<vector<float> > v_weights;
    vector<float> m_bias;
    vector<float> v_bias;
    float beta1;
    float beta2;
    float epsilon;
    int t;

    void initialize_weights() {
        float k = 1.0 / sqrt(input_features);
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

        bias.resize(output_features);
        m_bias.resize(output_features, 0.0);
        v_bias.resize(output_features, 0.0);
        for (int i = 0; i < output_features; ++i) {
            bias[i] = distr(gen);
        }
    }

public:
    LinearLayer(int in_features, int out_features)
        : input_features(in_features), output_features(out_features), beta1(0.9), beta2(0.999), epsilon(1e-8), t(0) {
        initialize_weights();
    }

    vector<vector<float> > forward(const vector<vector<float> >& input) {
        int batch_size = input.size();
        if (input[0].size() != input_features) {
            throw invalid_argument("Input size does not match input features.");
        }

        vector<vector<float> > outputs(batch_size, vector<float>(output_features));

        for (int b = 0; b < batch_size; ++b) {
            for (int i = 0; i < output_features; ++i) {
                outputs[b][i] = bias[i];
                for (int j = 0; j < input_features; ++j) {
                    outputs[b][i] += weights[i][j] * input[b][j];
                }
            }
        }

        return outputs;
    }

    vector<float> backward(const vector<float>& d_outputs, float learning_rate) {
        t++;

        vector<vector<float>> d_weights;
        vector<vector<float>> d_inputs; // this should just be a row vector, but we're matrix multiplying to get it. In the return statement we only return the first row.
        vector<vector<float>> d_outputs_2D = {d_outputs};

        broadcastMultiply(d_outputs, inputs, d_weights);
        multiplyMatrices(d_outputs_2D, weights, d_inputs);
        vector<float> d_bias = d_outputs;        

        for (int i = 0; i < output_features; ++i) {
            for (int j = 0; j < input_features; ++j) {
                m_weights[i][j] = beta1 * m_weights[i][j] + (1 - beta1) * d_weights[i][j];
                v_weights[i][j] = beta2 * v_weights[i][j] + (1 - beta2) * d_weights[i][j] * d_weights[i][j];

                float m_hat = m_weights[i][j] / (1 - pow(beta1, t));
                float v_hat = v_weights[i][j] / (1 - pow(beta2, t));

                weights[i][j] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
            }
            m_bias[i] = beta1 * m_bias[i] + (1 - beta1) * d_bias[i];
            v_bias[i] = beta2 * v_bias[i] + (1 - beta2) * d_bias[i] * d_bias[i];

            float m_hat_bias = m_bias[i] / (1 - pow(beta1, t));
            float v_hat_bias = v_bias[i] / (1 - pow(beta2, t));

            bias[i] -= learning_rate * m_hat_bias / (sqrt(v_hat_bias) + epsilon);
        }

        return d_inputs[0];
    }

    void update_weights(const vector<vector<float> >& d_weights, const vector<float>& d_bias, float learning_rate) {
        t++;
        for (int i = 0; i < output_features; ++i) {
            for (int j = 0; j < input_features; ++j) {
                m_weights[i][j] = beta1 * m_weights[i][j] + (1 - beta1) * d_weights[i][j];
                v_weights[i][j] = beta2 * v_weights[i][j] + (1 - beta2) * d_weights[i][j] * d_weights[i][j];

                float m_hat = m_weights[i][j] / (1 - pow(beta1, t));
                float v_hat = v_weights[i][j] / (1 - pow(beta2, t));

                weights[i][j] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
            }
            m_bias[i] = beta1 * m_bias[i] + (1 - beta1) * d_bias[i];
            v_bias[i] = beta2 * v_bias[i] + (1 - beta2) * d_bias[i] * d_bias[i];

            float m_hat_bias = m_bias[i] / (1 - pow(beta1, t));
            float v_hat_bias = v_bias[i] / (1 - pow(beta2, t));

            bias[i] -= learning_rate * m_hat_bias / (sqrt(v_hat_bias) + epsilon);
        }
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
        for (float val : bias) {
            cout << val << " ";
        }
        cout << endl;
    }
};
