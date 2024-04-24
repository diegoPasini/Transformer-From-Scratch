#include "../utils/Timer.cpp"
#include "../buildingBlocks/Tensor.cuh"
//#include "../buildingBlocks/nn/linearlayer.cu"
#include <iostream>
#include <vector>

using namespace std;

int main() {
    // Define a 3x3 matrix
    vector<float> matrixValues = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    vector<int> matrixDims = {3, 3};
    Tensor matrix(matrixValues, matrixDims, "cuda");

    // Define a 3x1 vector
    vector<float> vectorValues = {15, 12, 15};
    vector<int> vectorDims = {3, 1};
    Tensor vector(vectorValues, vectorDims, "cuda");

    // Perform matrix-vector multiplication
    try {
        Tensor result = matrix * vector;
        cout << "Result of matrix-vector multiplication: " << endl;
        cout << result.toString() << endl;
    } catch (const invalid_argument& e) {
        cerr << "Error: " << e.what() << endl;
    }

    return 0;
}
