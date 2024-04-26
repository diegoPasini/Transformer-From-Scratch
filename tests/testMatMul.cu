#include "../utils/Timer.cpp"
#include "../buildingBlocks/Tensor.cuh"
#include "../buildingBlocks/matrixOperations.cuh"
//#include "../buildingBlocks/nn/linearlayer.cu"
//#include "../buildingBlocks/nn/layer.cu"

#include <iostream>

using namespace std;
using namespace MatrixOperations;

// // Run: nvcc -o  main ./buildingBlocks/nn/linearlayer.cu ./tests/testLinearLayer.cpp ./buildingBlocks/nn/layer.cpp ./buildingBlocks/Tensor.cu ./buildingBlocks/matrixOperations.cu 
int main() {
    vector<float> inputs = {1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5};
    vector<int> dims = {4, 2};
    Tensor t1(inputs, dims);
    cout << t1.toString() << endl;

    vector<float> value2 = {1, 1};
    vector<int> dims2 = {2, 1};
    Tensor t2(value2, dims2);
    cout << t2.toString() << endl;

    Tensor t3 = t1 * t2;
    cout << t3.toString() << endl;
    // Example dimensions and input data
    // Example dimensions and input data
    int n = 4, m = 2, p = 1;
    float A[n * m] = {1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5};
    float B[m * p] = {1, 1};
    float C[n * p];

    // Perform matrix multiplication
    float *d_A, *d_B, *d_C;

    // Allocate memory on the device
    cudaMalloc(&d_A, n * m * sizeof(float));
    cudaMalloc(&d_B, m * p * sizeof(float));
    cudaMalloc(&d_C, n * p * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_A, A, n * m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, m * p * sizeof(float), cudaMemcpyHostToDevice);

    matrix_multiplication(d_A, d_B, d_C, n, m, p);

    // Copy the result back to the host
    cudaMemcpy(C, d_C, n * p * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Print the result matrix
    std::cout << "Resulting matrix C (n x p): " << std::endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            std::cout << C[i * p + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}

