#pragma once
#include <stdio.h>

namespace MatrixOperations {
    void vector_addition(float* input1, float* input2, float* output_vector, int size);
    void matrix_addition(float* a, float* b, float*c, int n, int m);
    void matrix_scaling(float* a, float* b, float scalar, int n, int m);
    void dot_product(float* a, float* b, float* c,  int n);
    void matrix_multiplication(float* a, float *b, float* c, int n, int m, int p);
}

