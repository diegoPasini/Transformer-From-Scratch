#include "../buildingBlocks/matrixOperations.cuh"
#include <iostream>

using namespace MatrixOperations;
int main() {

    // VECTOR ADDITION TESTING
    int N = 100;
    float *a, *b, *c;
    a = (float *)malloc(N*sizeof(float));
    b = (float *)malloc(N*sizeof(float));
    c = (float *)malloc(N*sizeof(float));

    for (int i = 0; i < N; i++){
        a[i] = i;
        b[i] = i;
    }    

    vector_addition(a, b, c, N);    

    std::cout << "------VECTOR ADDITION TEST------" << std::endl;
    std::cout << "Added Two Identical Incremented 100 Length Vectors" << std::endl;
    std::cout << "First Five Elements of the Outputed Array: " << c[0] << ", "<< c[1] << ", "  << c[2]  << ", " <<  c[3]  << ", " << c[4] << std::endl;


    // MATRIX ADDITION TESTING
    N = 256;
    int M = 256;
    float* matrix_a, *matrix_b, *matrix_c;
    matrix_a = (float *)malloc(N*M*sizeof(float));
    matrix_b = (float *)malloc(N*M*sizeof(float));
    matrix_c = (float *)malloc(N*M*sizeof(float));
    for (int i = 0; i < N*M; i++){
        matrix_a[i] = i;
        matrix_b[i] = i;
    }    

    matrix_addition(matrix_a, matrix_b, matrix_c, N, M);
    std::cout << "------MATRIX ADDITION TEST------" << std::endl;
    std::cout << "Added Two Identical Incremented 256 * 256 matrices" << std::endl;
    bool isAdditionCorrect = true;

    for(int i = 0; i < N; i++) {
        for(int j = 0; j < M; j++) {
            float a = matrix_a[i * M + j]; 
            float b = matrix_b[i * M + j]; 
            float c = matrix_c[i * M + j]; 
            
            if (abs((a + b) - c) > 1e-5) { 
                std::cout << "Mismatch at [" << i << "][" << j << "]: " 
                        << a << " + " << b << " != " << c << std::endl;
                isAdditionCorrect = false;
            }
        }
    }

    if (isAdditionCorrect) {
        std::cout << "The addition of the two matrices is correct." << std::endl;
    } else {
        std::cout << "The addition of the two matrices is incorrect." << std::endl;
    }

    N = 256;
    M = 512;
    std::cout << "Added Two Identical 256 * 512 matrices" << std::endl;
    float* matrix_a2, *matrix_b2, *matrix_c2;
    matrix_a2 = (float *)malloc(N*M*sizeof(float));
    matrix_b2 = (float *)malloc(N*M*sizeof(float));
    matrix_c2 = (float *)malloc(N*M*sizeof(float));
    for (int i = 0; i < N*M; i++){
        matrix_a2[i] = i;
        matrix_b2[i] = i;
    }  

    matrix_addition(matrix_a2, matrix_b2, matrix_c2, N, M); 
    isAdditionCorrect = true;

    for(int i = 0; i < N; i++) {
        for(int j = 0; j < M; j++) {
            float a = matrix_a2[i * N + j]; 
            float b = matrix_b2[i * N + j];
            float c = matrix_c2[i * N + j]; 
            
            if (abs((a + b) - c) > 1e-5) { 
                std::cout << "Mismatch at [" << i << "][" << j << "]: " 
                        << a << " + " << b << " != " << c << std::endl;
                isAdditionCorrect = false;
            }
        }
    }

    if (isAdditionCorrect) {
        std::cout << "The addition of the two matrices is correct." << std::endl;
    } else {
        std::cout << "The addition of the two matrices is incorrect." << std::endl;
    }

    // Test Matrix Scaling
    std::cout << "------MATRIX Scaling TEST------" << std::endl;
    std::cout << "Multiplied a 256*256 matrix by 2.5" << std::endl;
    N = 256;
    M = 256;
    float* matrix_a3, *matrix_b3;
    matrix_a3 = (float *)malloc(N*M*sizeof(float));
    matrix_b3 = (float *)malloc(N*M*sizeof(float));
    for (int i = 0; i < N*M; i++){
        matrix_a3[i] = i;
        matrix_b3[i] = i;
    }
    matrix_scaling(matrix_a3, matrix_b3, 2.5, N, M); 
    bool isScalingCorrect = true;

    for(int i = 0; i < N; i++) {
        for(int j = 0; j < M; j++) {
            float a = matrix_a3[i * N + j]; 
            float b = matrix_b3[i * N + j];            
            if (abs((a * 2.5) - b) > 1e-5) { 
                std::cout << "Mismatch at [" << i << "][" << j << "]: " 
                        << a << " * " << 2.5 << " != " << b << std::endl;
                isScalingCorrect = false;
            }
        }
    }

    if (isScalingCorrect) {
        std::cout << "The addition of the two matrices is correct." << std::endl;
    } else {
        std::cout << "The addition of the two matrices is incorrect." << std::endl;
    }

    std::cout << "Multiplied a 256*512 matrix by 3" << std::endl;
    N = 256;
    M = 512;
    float* matrix_a4, *matrix_b4;
    matrix_a4 = (float *)malloc(N*M*sizeof(float));
    matrix_b4 = (float *)malloc(N*M*sizeof(float));
    for (int i = 0; i < N*M; i++){
        matrix_a4[i] = i;
        matrix_b4[i] = i;
    }
    matrix_scaling(matrix_a4, matrix_b4, 4, N, M); 
    isScalingCorrect = true;

    for(int i = 0; i < N; i++) {
        for(int j = 0; j < M; j++) {
            float a = matrix_a4[i * N + j]; 
            float b = matrix_b4[i * N + j];            
            if (abs((a * 4) - b) > 1e-5) { 
                std::cout << "Mismatch at [" << i << "][" << j << "]: " 
                        << a << " * " << 4 << " != " << b << std::endl;
                isScalingCorrect = false;
            }
        }
    }

    if (isScalingCorrect) {
        std::cout << "The addition of the two matrices is correct." << std::endl;
    } else {
        std::cout << "The addition of the two matrices is incorrect." << std::endl;
    }

    //Test Matrix Multiplication
    std::cout << "------MATRIX MULTIPLICATION TEST------" << std::endl;
    std::cout << "Multiplied Matrix [[1, 2], [3, 4]] and [[5, 6], [7, 8]]" << std::endl;
    N = 2;
    M = 2;
    int P = 2;
    float *mul_matrixA, *mul_matrixB, *mul_matrixC;
    mul_matrixA = (float *)malloc(N*M*sizeof(float));
    mul_matrixB = (float *)malloc(N*P*sizeof(float));
    mul_matrixC = (float *)malloc(P*M*sizeof(float));
    mul_matrixA[0] = 1; mul_matrixA[1] = 2;
    mul_matrixA[2] = 3; mul_matrixA[3] = 4;

    mul_matrixB[0] = 5; mul_matrixB[1] = 6;
    mul_matrixB[2] = 7; mul_matrixB[3] = 8;

    matrix_multiplication(mul_matrixA, mul_matrixB, mul_matrixC, N, M, P);

    std::cout << "Result Matrix: " << std::endl;
    std::cout << "[" << mul_matrixC[0] << ", " << mul_matrixC[1] << "], [" 
              << mul_matrixC[2] << ", " << mul_matrixC[3] << "]" << std::endl;

    bool isMultiplicationCorrect = true;
    if (std::abs(mul_matrixC[0] - 19) > 1e-5 || std::abs(mul_matrixC[1] - 22) > 1e-5 ||
        std::abs(mul_matrixC[2] - 43) > 1e-5 || std::abs(mul_matrixC[3] - 50) > 1e-5) {
        isMultiplicationCorrect = false;
    }

    if (isMultiplicationCorrect) {
        std::cout << "Matrix multiplication is correct." << std::endl;
    } else {
        std::cout << "Matrix multiplication is incorrect." << std::endl;
    }

    N = 2, M = 3, P = 4;

    float *A = (float *)malloc(M*N*sizeof(float)); 
    float *B = (float *)malloc(N*P*sizeof(float));
    float *C = (float *)malloc(M*P*sizeof(float));

    A[0] = 1; A[1] = 2; A[2] = 3;
    A[3] = 4; A[4] = 5; A[5] = 6;

    B[0] = 7; B[1] = 8; B[2] = 9; B[3] = 10;
    B[4] = 11; B[5] = 12; B[6] = 13; B[7] = 14;
    B[8] = 15; B[9] = 16; B[10] = 17; B[11] = 18;

    matrix_multiplication(A, B, C, M, N, P);

    std::cout << "Result Matrix C (2x4):" << std::endl;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < P; ++j) {
            std::cout << C[i * P + j] << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "------MATRIX TRANSPOSE TESTS------" << std::endl;
    std::cout << "Multiplied Matrix [[1, 2], [3, 4]] and [[5, 6], [7, 8]]" << std::endl;
    N = 2, M = 3, P = 4;

    float *A = (float *)malloc(M*N*sizeof(float)); 
    float *B = (float *)malloc(N*P*sizeof(float));
    float *C = (float *)malloc(M*P*sizeof(float));

    A[0] = 1; A[1] = 2; A[2] = 3;
    A[3] = 4; A[4] = 5; A[5] = 6;

    B[0] = 7; B[1] = 8; B[2] = 9; B[3] = 10;
    B[4] = 11; B[5] = 12; B[6] = 13; B[7] = 14;
    B[8] = 15; B[9] = 16; B[10] = 17; B[11] = 18;

    matrix_multiplication(A, B, C, M, N, P);

    std::cout << "Result Matrix C (2x4):" << std::endl;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < P; ++j) {
            std::cout << C[i * P + j] << " ";
        }
        std::cout << std::endl;
    }
}
