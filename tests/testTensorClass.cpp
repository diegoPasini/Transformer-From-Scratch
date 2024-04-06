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

    // Test Matrix Addition
    // Test 1-dimensional tensors
    float* values1D_1 = new float[3];
    values1D_1[0] = 1;
    values1D_1[1] = 2;
    values1D_1[2] = 3;
    int* dims1D_1 = new int[1];
    dims1D_1[0] = 3;
    Tensor t1D_1(values1D_1, dims1D_1, 1);

    float* values1D_2 = new float[3];
    values1D_2[0] = 4;
    values1D_2[1] = 5;
    values1D_2[2] = 6;
    int* dims1D_2 = new int[1];
    dims1D_2[0] = 3;
    Tensor t1D_2(values1D_2, dims1D_2, 1);

    Tensor result1D = t1D_1 + t1D_2;
    cout << "Result1D: " << result1D.toString() << endl;

    // Test 2-dimensional tensors
    float* values2D_1 = new float[4];
    values2D_1[0] = 1;
    values2D_1[1] = 2;
    values2D_1[2] = 3;
    values2D_1[3] = 4;
    int* dims2D_1 = new int[2];
    dims2D_1[0] = 2;
    dims2D_1[1] = 2;
    Tensor t2D_1(values2D_1, dims2D_1, 2);

    float* values2D_2 = new float[4];
    values2D_2[0] = 5;
    values2D_2[1] = 6;
    values2D_2[2] = 7;
    values2D_2[3] = 8;
    int* dims2D_2 = new int[2];
    dims2D_2[0] = 2;
    dims2D_2[1] = 2;
    Tensor t2D_2(values2D_2, dims2D_2, 2);

    Tensor result2D = t2D_1 + t2D_2;
    cout << "Result2D: " << result2D.toString() << endl;


    // Test 3-dimensional tensors
    float* values3D_1 = new float[8];
    values3D_1[0] = 1;
    values3D_1[1] = 2;
    values3D_1[2] = 3;
    values3D_1[3] = 4;
    values3D_1[4] = 5;
    values3D_1[5] = 6;
    values3D_1[6] = 7;
    values3D_1[7] = 8;
    int* dims3D_1 = new int[3];
    dims3D_1[0] = 2;
    dims3D_1[1] = 2;
    dims3D_1[2] = 2;
    Tensor t3D_1(values3D_1, dims3D_1, 3);

    float* values3D_2 = new float[8];
    values3D_2[0] = 9;
    values3D_2[1] = 10;
    values3D_2[2] = 11;
    values3D_2[3] = 12;
    values3D_2[4] = 13;
    values3D_2[5] = 14;
    values3D_2[6] = 15;
    values3D_2[7] = 16;
    int* dims3D_2 = new int[3];
    dims3D_2[0] = 2;
    dims3D_2[1] = 2;
    dims3D_2[2] = 2;
    Tensor t3D_2(values3D_2, dims3D_2, 3);

    Tensor result3D = t3D_1 + t3D_2;
    cout << "Result3D: " << result3D.toString() << endl;

    // Test Addition on two broadcasted arrays

    // Matrix/Vector Multiplication





    // CUDA Testing
}