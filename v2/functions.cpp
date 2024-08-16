#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

// RELU
void relu(vector<vector<float>>& a) {
    int m = a.size();
    int n = a[0].size();
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            a[i][j] = max(0.0f, a[i][j]);
        }
    }
}

// RELU Backward
void relu_backward(vector<vector<float>>& d_output, const vector<vector<float>>& input) {
    int m = d_output.size();
    int n = d_output[0].size();
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (input[i][j] <= 0) {
                d_output[i][j] = 0;
            }
        }
    }
}

// Softmax
void softmax(vector<vector<float>>& a) {
    int m = a.size();
    int n = a[0].size();
    for (int i = 0; i < m; i++) {
        float row_sum = 0.0f;
        for (int j = 0; j < n; j++) {
            a[i][j] = exp(a[i][j]);
            row_sum += a[i][j];
        }
        for (int j = 0; j < n; j++) {
            a[i][j] /= row_sum;
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
