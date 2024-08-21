#include <iostream>
#include <vector>
#include <cmath>
#include "functions.h"
#include "matrix_operations.h"

using namespace std;


class ScaledDotProductAttention {
    private:
    vector<vector<float>> queries;
    vector<vector<float>> keys;
    vector<vector<float>> values;
    vector<vector<float>> attention_weights;
    vector<vector<float>> outputs;

    public:
    ScaledDotProductAttention() {}

    void applyUpperTriangularMask(vector<vector<float>>& matrix) {
        int rows = matrix.size();
        int cols = matrix[0].size();
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                if (j < i) {
                    matrix[i][j] = -INFINITY;  
                }
            }
        }
    }

    vector<vector<float>> forward(const vector<vector<float>>& Q, const vector<vector<float>>& K, const vector<vector<float>>& V) {
        queries = Q;
        keys = K;
        values = V;

        vector<vector<float>> K_transpose;
        tranpose(K, K_transpose);
        vector<vector<float>> QK;
        multiplyMatrices(Q, K_transpose, QK);

        float d_k = static_cast<float>(K[0].size());
        float scale_factor = sqrt(d_k);
        for (auto& row : QK) {
            for (auto& elem : row) {
                elem /= scale_factor;
            }
        }

        applyUpperTriangularMask(QK);
        softmax(QK);
        attention_weights = QK;
        multiplyMatrices(attention_weights, V, outputs);

        return outputs;
    }

    tuple<vector<vector<float>>, vector<vector<float>>, vector<vector<float>>> backward(const vector<vector<float>>& dL_dout) {
        vector<vector<float>> dL_dV, dL_dQ, dL_dK;

        vector<vector<float>> attention_weights_transpose;
        tranpose(attention_weights, attention_weights_transpose);
        multiplyMatrices(attention_weights_transpose, dL_dout, dL_dV);

        vector<vector<float>> values_transpose;
        tranpose(values, values_transpose);
        vector<vector<float>> dL_dattention_weights;
        multiplyMatrices(dL_dout, values_transpose, dL_dattention_weights);

        vector<vector<float>> dL_dQK = softmaxBackward(dL_dattention_weights, attention_weights);

        for (int i = 0; i < dL_dQK.size(); ++i) {
            for (int j = 0; j < dL_dQK[0].size(); ++j) {
                if (j < i) {
                    dL_dQK[i][j] = 0;
                }
            }
        }

        float d_k = static_cast<float>(keys[0].size());
        float scale_factor = sqrt(d_k);
        for (auto& row : dL_dQK) {
            for (auto& elem : row) {
                elem /= scale_factor;
            }
        }

        vector<vector<float>> K_transpose;
        tranpose(keys, K_transpose);
        multiplyMatrices(dL_dQK, K_transpose, dL_dQ);

        vector<vector<float>> Q_transpose;
        tranpose(queries, Q_transpose);
        multiplyMatrices(dL_dQK, Q_transpose, dL_dK);

        return make_tuple(dL_dQ, dL_dK, dL_dV);
    }
};