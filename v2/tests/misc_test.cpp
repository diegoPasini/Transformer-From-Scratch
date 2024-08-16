#include "../functions.cpp"
#include "../linear_layer.cpp"
#include "../matrix_operations.cpp"
#include <iostream>
#include <vector>

using namespace std; 

int main() {
    // vector<float> a = {1.0, 2.0, 3.0};
    // vector<float> y = {0.0, 0.0, 1.0};
    // auto [loss, gradient] = softmaxLoss(a, y);
    // cout << "Loss: " << loss << endl;
    // cout << "Gradient: ";
    // for (auto val : gradient) {
    //     cout << val << " ";
    // }
    // cout << endl;

    LinearLayer layer(3, 2);
    layer.print_weights();
    layer.print_bias();
    vector<float> input = {0.2, 0.3, 0.5};
    vector<float> output = layer.forward({input})[0];
    cout << "Output: ";
    for (auto val : output) {
        cout << val << " ";
    }
    cout << endl;

    vector<float> a = output;
    vector<float> y = {1,0};
    auto [loss, gradient] = softmaxLoss(a, y);
    cout << "Loss: " << loss << endl;
    cout << "Gradient: ";
    for (auto val : gradient) {
        cout << val << " ";
    }
    cout << endl;
}
