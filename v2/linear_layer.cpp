#include <vector>
#include <random>
#include <iostream>
#include <cmath>

using namespace std;

class LinearLayer {
private:
    int input_features;
    int output_features;
    vector<vector<float>> weights;
    vector<float> bias;
    vector<float> inputs;
    vector<float> outputs;

    // Adam optimizer parameters
    vector<vector<float>> m_weights;
    vector<vector<float>> v_weights;
    vector<float> m_bias;
    vector<float> v_bias;
    float beta1 = 0.9;
    float beta2 = 0.999;
    float epsilon = 1e-8;
    int t = 0;

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
        : input_features(in_features), output_features(out_features) {
        initialize_weights();
    }

    vector<float> forward(const vector<float>& input) {
        if (input.size() != input_features) {
            throw invalid_argument("Input size does not match input features.");
        }

        inputs = input;
        outputs.resize(output_features);

        for (int i = 0; i < output_features; ++i) {
            outputs[i] = bias[i];
            for (int j = 0; j < input_features; ++j) {
                outputs[i] += weights[i][j] * input[j];
            }
        }

        return outputs;
    }

    void update_weights(const vector<vector<float>>& d_weights, const vector<float>& d_bias, float learning_rate) {
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
        for (const auto& row : weights) {
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
