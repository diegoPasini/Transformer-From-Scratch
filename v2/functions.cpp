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
    
    float epsilon = 1e-9;  // Small value to prevent log(0)
    float loss = 0.0f;
    vector<float> gradient(n);
    vector<float> softmax_output(n);

    float max_a = *max_element(a.begin(), a.end());
    
    // Compute softmax
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        softmax_output[i] = exp(a[i] - max_a);  
        sum += softmax_output[i];
    }
    for (int i = 0; i < n; i++) {
        softmax_output[i] /= sum;
    }

    // Compute cross-entropy loss
    for (int i = 0; i < n; i++) {
        if (y[i] > 0) {  // Avoid computing log(0)
            loss -= y[i] * log(softmax_output[i] + epsilon);
        }
    }

    // Compute gradient (softmax_output - y)
    for (int i = 0; i < n; i++) {
        gradient[i] = softmax_output[i] - y[i];
    }

    return make_pair(loss, gradient);
}
