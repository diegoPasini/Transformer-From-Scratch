#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

// Matrix Addition
// Not CUDA
void addMatrices(const vector<vector<float>>& a, const vector<vector<float>>& b, vector<vector<float>>& c) {
    int m = a.size();
    int n = a[0].size();
    c.resize(m, vector<float>(n));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            c[i][j] = a[i][j] + b[i][j];
        }
    }
}

// Matrix Multiplication
// Not CUDA
void multiplyMatrices(const vector<vector<float>>& a, const vector<vector<float>>& b, vector<vector<float>>& c) {
    int m = a.size();
    int n = a[0].size();
    int p = b[0].size();
    c.resize(m, vector<float>(p, 0.0f));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            for (int k = 0; k < n; k++) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}
