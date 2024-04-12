#include "../utils/Timer.cpp"
#include "../buildingBlocks/matrixOperations.cuh"
#include <iostream>
#include <vector>

int main() {
    const int N = 10000;
    const int M = 10000;
    const int P = 10000;
    Timer timer;

    // Allocate memory for the matrices
    float* matrix_a = new float[N * M];
    float* matrix_b = new float[M * P];
    float* matrix_c = new float[N * P];

    // Initialize matrices with some values
    for (int i = 0; i < N * M; ++i) {
        matrix_a[i] = 1.0f; // Example value
    }
    for (int i = 0; i < M * P; ++i) {
        matrix_b[i] = 1.0f; // Example value
    }

    // Perform matrix multiplication and time it
    timer.start();
    MatrixOperations::matrix_multiplication(matrix_a, matrix_b, matrix_c, N, M, P);
    timer.stop();

    // Output the time taken for matrix multiplication
    std::cout << "Time taken for matrix multiplication of a 10000x10000 matrix: ";
    timer.printLastTime();

    // Clean up memory
    delete[] matrix_a;
    delete[] matrix_b;
    delete[] matrix_c;

    return 0;
}
