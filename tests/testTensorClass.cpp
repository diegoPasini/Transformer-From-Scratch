#include "../buildingBlocks/Tensor.cuh"
#include <iostream>

int main() {
    // Test Constructor, getDimensionsString, and indexing
    // Test 2 * 2 Matrix Without CUDA
    float* values = new float[4];
    for(int i = 0; i < 4; i++) {
        values[i] = i;
    }

    int* dims = new int[2];
    for(int j = 0; j < 2; j++) {
        dims[j] = 2;
    }

    Tensor t1(values, dims, 2);

    cout << t1.getDimensionsString() << endl;

    int* index = new int[2];
    index[0] = 1;
    index[1] = 1;
    float value = t1[index];
    cout << "Element at index t1: " << value << endl;

    // // Try Error Indices:
    // int* index2 = new int[2];
    // index2[0] = 2;
    // index2[1] = 2;
    // float value2 = t1[index2];

    // Test 2*3 Matrix Without CUDA
    float* values2 = new float[6];
    for(int i = 0; i < 6; i++) {
        values2[i] = i;
    }

    int* dims2 = new int[2];
    for(int j = 0; j < 2; j++) {
        dims2[j] = 2 + j;
    }

    Tensor t2(values2, dims2, 2);

    cout << t2.getDimensionsString() << endl;


    // Test 3 * 4 * 3 Matrix
    float* values3 = new float[36];
    for(int i = 0; i < 36; i++) {
        values3[i] = i;
    }

    int* dims3 = new int[3];
    dims3[0] = 3;
    dims3[1] = 4;
    dims3[2] = 3;

    Tensor t3(values3, dims3, 3);

    cout << t3.getDimensionsString() << endl;

    int* index3 = new int[3];
    index3[0] = 2;
    index3[1] = 3;
    index3[2] = 2;
    float value3 = t3[index3];
    cout << "Element at index t1: " << value3 << endl;


    // Test Reference and equality operator
    Tensor t4 = t3;
    cout << "Tensors 3 and 4 are the same: " << (t3 == t4) << endl;

    // Make same matrix as t3
    float* values4 = new float[36];
    for(int i = 0; i < 36; i++) {
        values4[i] = i;
    }

    int* dims4 = new int[3];
    dims4[0] = 3;
    dims4[1] = 4;
    dims4[2] = 3;

    Tensor t5(values4, dims4, 3);
    cout << "Tensors 3 and 4 are the same: " << (t3 == t5) << endl;
       

    // Test Reshape
    int* indexes3 = new int[3];
    indexes3[0] = 1;
    indexes3[1] = 9;
    indexes3[2] = 4;
    t5.reshape(indexes3, 3);
    cout << "Tensor 5 Shape: " << t5.getDimensionsString() << endl;


    // Test Tensor toString
    cout << t5.toString() << endl;

    // Test Broadcastable

    // Test checkSizes

    // Test Addition on Two of the same shaped array

    // Test Addition on two broadcasted arrays

    // Matrix/Vector Multiplication





    // CUDA Testing
}