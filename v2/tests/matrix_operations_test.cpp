#include "../matrix_operations.cpp"
#include <iostream>
#include <vector>

using namespace std; 

int main() {
    vector<vector<float>> m1 = {{1, 2, 3, 4}, {5, 6, 7, 8}};
    vector<vector<float>> m2 = {{1, 2}, {4, 5}, {7, 8}, {9, 10}};
    vector<vector<float>> m3;
    multiplyMatrices(m1, m2, m3);
    print2DMatrix(m3);
}
