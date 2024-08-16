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
void softmax(vector<vector<float>>& batch) {
    for (auto& a : batch) {
        float row_sum = 0.0f;
        for (auto& val : a) {
            val = exp(val);
            row_sum += val;
        }
        for (auto& val : a) {
            val /= row_sum;
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
            loss -= y[i] * log(softmax_output[i] + 1e-9);  
        }
    }

    // Compute gradient (softmax_output - y)
    for (int i = 0; i < n; i++) {
        gradient[i] = softmax_output[i] - y[i];
    }

    return make_pair(loss, gradient);
}
