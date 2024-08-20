#include <iostream>
#include <vector>
#include <cmath>
#include "attention.cpp"
#include "layernorm.cpp"
#include "linear_layer.cpp"
#include "matrix_operations.h"

using namespace std;

class MultiHeadAttention {
private:
    int num_heads;
    int dim;
    int head_dim;
    vector<LinearLayer> linear_layers;
    LinearLayer linear_out;
    Layernorm layernorm;

public:
    MultiHeadAttention(int num_heads, int dim) 
        : num_heads(num_heads), dim(dim), head_dim(dim / num_heads), 
          linear_out(dim, dim), layernorm() {
        for (int i = 0; i < num_heads; ++i) {
            linear_layers.emplace_back(dim, head_dim, false);
        }
    }

    vector<vector<float>> forward(const vector<vector<float>>& Q, const vector<vector<float>>& K, const vector<vector<float>>& V) {
        vector<vector<float>> concatenated_outputs;
        vector<vector<float>> head_outputs;

        for (int i = 0; i < num_heads; ++i) {
            vector<vector<float>> Q_proj = linear_layers[i].forward(Q);
            vector<vector<float>> K_proj = linear_layers[i].forward(K);
            vector<vector<float>> V_proj = linear_layers[i].forward(V);

            ScaledDotProductAttention attention;
            head_outputs = attention.forward(Q_proj, K_proj, V_proj);

            if (concatenated_outputs.empty()) {
                concatenated_outputs = head_outputs;
            } else {
                for (size_t j = 0; j < concatenated_outputs.size(); ++j) {
                    concatenated_outputs[j].insert(concatenated_outputs[j].end(), head_outputs[j].begin(), head_outputs[j].end());
                }
            }
        }

        vector<vector<float>> output = linear_out.forward(concatenated_outputs);
        output = layernorm.forward(output);

        return output;
    }

    vector<vector<float>> backward(const vector<vector<float>>& d_outputs, float learning_rate, int t) {
        // TODO: Implement
    }
};
