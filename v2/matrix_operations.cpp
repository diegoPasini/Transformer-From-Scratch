#include <stdlib.h>
#include <stdio.h>
#include <iostream>

using namespace std;

// Matrix Addition
// Not CUDA
void addMatrices(float *a, float *b, float *c, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            c[i * n + j] = a[i * n + j] + b[i * n + j];
        }
    }
}

// Matrix Multiplication
// Not CUDA
void multiplyMatrices(float *a, float *b, float *c, int m, int n, int p) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            for (int k = 0; k < n; k++) {
                c[i * p + j] += a[i * n + k] * b[k * p + j];
            }
        }
    }
}
