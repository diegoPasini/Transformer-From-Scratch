#include <iostream>
#include <vector>
#include <cmath>
#include "functions.h"
#include "linear_layer.h"
#include "layernorm.h"
#include "matrix_operations.h"

using namespace std;

class Feedforward {
    LinearLayer lin1;
    LinearLayer lin2;
    Layernorm layernorm1;
    Layernorm layernorm2;

public:
    Feedforward(int dim, int hid_dim) : 
    lin1(dim, hid_dim, false), lin2(hid_dim, dim, false), layernorm1(), layernorm2() {
    }

    vector<vector<float>> forward(vector<vector<float>> x) {
        x = lin1.forward(x);
        x = layernorm1.forward(x);
        x = lin2.forward(x);
        x = layernorm2.forward(x);
        return x;
    }

    vector<vector<float>> backward(const vector<vector<float>>& d_outputs, float learning_rate, int t) {
        vector<vector<float>> d_out = d_outputs;
        d_out = layernorm2.backward(d_out, learning_rate, t);
        d_out = lin2.backward(d_out, learning_rate, t);
        d_out = layernorm1.backward(d_out, learning_rate, t);
        d_out = lin1.backward(d_out, learning_rate, t);
        return d_out;
    }
};
