#include "../operations/matrixOperations.cuh"
#include <iostream>

using namespace MatrixOperations;
int main() {
    int N = 100;
    float *a, *b, *c;
    
    // Allocate Memory on Host
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

}



