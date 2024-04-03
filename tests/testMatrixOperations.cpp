#include "../operations/matrixOperations.cuh"
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

}