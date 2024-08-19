#include <iostream>
#include <vector>
#include <cmath>
#include "convolution.cpp"
#include "functions.h"
#include "linear_layer.cpp"
#include "matrix_operations.h"

using namespace std;


class Feedforward {
    LinearLayer lin1;
    LinearLayer lin2;


public:
    Feedforward(int dim, int hid_dim) : 
    lin1(dim, hid_dim, false), lin2(hid_dim, dim, false) {
        // TODO: Add in LayerNorm
    }

    vector<vector<float>> forward(vector<vector<float>> x) {
        x  = lin1.forward(x);
        x  = lin2.forward(x);
    }

    vector<vector<float>> backward(const vector<vector<float>> d_outputs, float learnig_rate, int t) {
        // TODO : Implement 
    }


};
