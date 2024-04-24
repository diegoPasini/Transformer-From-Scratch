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
    cout << lin.toStringWeights() << endl;
    
}
