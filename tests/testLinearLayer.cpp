#include "../utils/Timer.cpp"
#include "../buildingBlocks/Tensor.cuh"
#include "../buildingBlocks/nn/linearlayer.cu"
//#include "../buildingBlocks/nn/layer.cu"

#include <iostream>

using namespace std;

// Run: nvcc -o  main ./buildingBlocks/nn/linearlayer.cu ./tests/testLinearLayer.cpp ./buildingBlocks/nn/layer.cpp ./buildingBlocks/Tensor.cu ./buildingBlocks/matrixOperations.cu 
int main() {
    float lr = 0.01f;
    LinearLayer lin(3, 3, lr);
    vector<float> input = {1, 1, 1};
    vector<int> dimensions = {3, 1};
    cout << lin.toStringWeights() << endl;
    cout << lin.toStringBiases() << endl;
    Tensor y = lin.forward(Tensor(input, dimensions, "cuda"));
    cout << y.toString() << endl;
    vector<float> gammaValues = {2, 2, 2};
    vector<int> gammaDimensions = {3, 1};
    Tensor gamma(gammaValues, gammaDimensions, "cuda");
    Tensor newGamma = lin.backward(gamma);
    cout << "Weights" << endl;
    cout << lin.toStringWeights() << endl;
    cout << lin.toStringBiases() << endl;
    cout << newGamma.toString() << endl;
}
