#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

// RELU
void relu(vector<vector<float>>& batch) {
    for (auto& a : batch) {
        for (auto& val : a) {
            val = max(0.0f, val);
        }
    }
}

// RELU Backward
void relu_backward(vector<vector<float>>& d_output, const vector<vector<float>>& input) {
    for (size_t i = 0; i < d_output.size(); ++i) {
        for (size_t j = 0; j < d_output[i].size(); ++j) {
            if (input[i][j] <= 0) {
                d_output[i][j] = 0;
            }
        }
    }
}

void transpose(const vector<vector<float>>& input, vector<vector<float>>& output) {
    if (input.empty()) return;
    int rows = input.size();
    int cols = input[0].size();
    output.resize(cols, vector<float>(rows));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            output[j][i] = input[i][j];
        }
    }
}


// Softmax
void softmax(vector<vector<float>>& logits) {
    for (auto& logit : logits) {
        float max_val = *max_element(logit.begin(), logit.end());
        float sum_exp = 0.0f;
        for (auto& val : logit) {
            val = exp(val - max_val); 
            sum_exp += val;
        }
        for (auto& val : logit) {
            val /= sum_exp;
        }
    }
}

// Softmax Loss and Gradient
// true labels need to be normalized
pair<float, vector<float>> softmaxLoss(const vector<float>& a, const vector<float>& y) {
    int n = a.size();
    if (n != y.size()) {
        throw invalid_argument("Dimensions of input and output do not match.");
    }
    
    float epsilon = 1e-9; 
    float loss = 0.0f;
    vector<float> gradient(n);
    vector<float> softmax_output(n);

    float max_a = *max_element(a.begin(), a.end());
    
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        softmax_output[i] = exp(a[i] - max_a);  
        sum += softmax_output[i];
    }
    for (int i = 0; i < n; i++) {
        softmax_output[i] /= sum;
    }

    for (int i = 0; i < n; i++) {
        if (y[i] > 0) {  
            loss -= y[i] * log(softmax_output[i] + epsilon);
        }
    }

    for (int i = 0; i < n; i++) {
        gradient[i] = softmax_output[i] - y[i];
    }

    return make_pair(loss, gradient);
}

vector<vector<float>> softmaxBackward(const vector<vector<float>>& dL_dout, const vector<vector<float>>& softmax_output) {
    vector<vector<float>> dL_dinput(dL_dout.size(), vector<float>(dL_dout[0].size()));

    for (size_t i = 0; i < dL_dout.size(); ++i) {
        for (size_t j = 0; j < dL_dout[i].size(); ++j) {
            float s = softmax_output[i][j];
            dL_dinput[i][j] = s * (1 - s) * dL_dout[i][j];
        }
    }

    return dL_dinput;
}
