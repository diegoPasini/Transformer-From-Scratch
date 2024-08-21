#include <iostream>
#include <vector>
#include <cmath>
#include "attention.h"
#include "layernorm.h"
#include "linear_layer.h"
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
        vector<vector<float>> d_inputs;
        vector<vector<float>> d_concatenated_outputs = layernorm.backward(d_outputs, learning_rate, t);
        vector<vector<float>> d_linear_out = linear_out.backward(d_concatenated_outputs, learning_rate, t);

        int batch_size = d_linear_out.size();
        int feature_size = d_linear_out[0].size();
        int head_feature_size = feature_size / num_heads;

        vector<vector<float>> dQ(batch_size, vector<float>(dim, 0.0f));
        vector<vector<float>> dK(batch_size, vector<float>(dim, 0.0f));
        vector<vector<float>> dV(batch_size, vector<float>(dim, 0.0f));

        for (int i = 0; i < num_heads; ++i) {
            vector<vector<float>> d_head(d_linear_out.size(), vector<float>(head_feature_size));
            for (int j = 0; j < d_linear_out.size(); ++j) {
                copy(d_linear_out[j].begin() + i * head_feature_size, d_linear_out[j].begin() + (i + 1) * head_feature_size, d_head[j].begin());
            }

            ScaledDotProductAttention attention;
            vector<vector<float>> dL_dQ, dL_dK, dL_dV;
            tie(dL_dQ, dL_dK, dL_dV) = attention.backward(d_head);

            vector<vector<float>> dQ_proj = linear_layers[i].backward(dL_dQ, learning_rate, t);
            vector<vector<float>> dK_proj = linear_layers[i].backward(dL_dK, learning_rate, t);
            vector<vector<float>> dV_proj = linear_layers[i].backward(dL_dV, learning_rate, t);

            for (int j = 0; j < batch_size; ++j) {
                for (int k = 0; k < dim; ++k) {
                    dQ[j][k] += dQ_proj[j][k];
                    dK[j][k] += dK_proj[j][k];
                    dV[j][k] += dV_proj[j][k];
                }
            }
        }
        return dQ;
    }
};
