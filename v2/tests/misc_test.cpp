#include "../functions.cpp"
#include <iostream>
#include <vector>

using namespace std; 

int main() {
    vector<float> a = {1.0, 2.0, 3.0};
    vector<float> y = {0.0, 0.0, 1.0};
    auto [loss, gradient] = softmaxLoss(a, y);
    cout << "Loss: " << loss << endl;
    cout << "Gradient: ";
    for (auto val : gradient) {
        cout << val << " ";
    }
    cout << endl;
}
