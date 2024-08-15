#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

// Matrix Addition
// Not CUDA
void addMatrices(const vector<vector<float> >& a, const vector<vector<float> >& b, vector<vector<float> >& c) {
    int m = a.size();
    int n = a[0].size();
    if (m != b.size() || n != b[0].size()) {
        throw invalid_argument("Matrices of sizes " + to_string(m) + "x" + to_string(n) + " and " + to_string(b.size()) + "x" + to_string(b[0].size()) + " are not compatible for addition");
    }
    c.resize(m, vector<float>(n));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            c[i][j] = a[i][j] + b[i][j];
        }
    }
}

// Matrix Multiplication
// Not CUDA
void multiplyMatrices(const vector<vector<float> >& a, const vector<vector<float> >& b, vector<vector<float> >& c) {
    int m = a.size();
    int n = a[0].size();
    int p = b[0].size();
    if (n != b.size()) {
        throw invalid_argument("Matrices of sizes " + to_string(m) + "x" + to_string(n) + " and " + to_string(b.size()) + "x" + to_string(b[0].size()) + " are not compatible for multiplication");
    }
    c.resize(m, vector<float>(p, 0.0f));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            for (int k = 0; k < n; k++) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

// Broadcasted position multiplication
// first input is column vector, second input is row vector
// the column vector is "copied" right and the row vector is "copied" down
void broadcastMultiply(const vector<float>& a, const vector<float>& b, vector<vector<float> >& c) {
    int m = a.size();
    int n = b.size();
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            c[i][j] = a[i] * b[j];
        }
    }
}


void print2DMatrix(const vector<vector<float> >& a) {
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < a[0].size(); j++) {
            cout << a[i][j] << " ";
        }
        cout << endl;
    }
}
