#include <iostream>
#include <vector>
#include <cmath>
#include "convolution.cpp"
#include "functions.h"
#include "linear_layer.cpp"
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

    vector<vector<float>> backward(const vector<vector<float>>& d_outputs, float learning_rate, int t) {
       // TODO: Implement
    }
};