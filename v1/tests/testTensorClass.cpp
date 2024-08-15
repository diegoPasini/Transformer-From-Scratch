#include "../buildingBlocks/Tensor.cuh"
#include <iostream>
#include <vector>

int main() {
    using std::vector;
    using std::cout;
    using std::endl;

    // Test Constructor, getDimensionsString, and indexing
    // Test 2 * 2 Matrix Without CUDA
    vector<float> values = {0, 1, 2, 3};

    vector<int> dims = {2, 2};

    Tensor t1(values, dims);

    cout << t1.getDimensionsString() << endl;

    vector<int> index = {1, 1};
    float value = t1[index];
    cout << "Element at index t1: " << value << endl;

    // Test 2*3 Matrix Without CUDA
    vector<float> values2 = {0, 1, 2, 3, 4, 5};

    vector<int> dims2 = {2, 3};

    Tensor t2(values2, dims2);

    cout << t2.getDimensionsString() << endl;

    // Test 3 * 4 * 3 Matrix
    vector<float> values3(36);
    for(int i = 0; i < 36; i++) {
        values3[i] = i;
    }

    vector<int> dims3 = {3, 4, 3};

    Tensor t3(values3, dims3);

    cout << t3.getDimensionsString() << endl;

    vector<int> index3 = {2, 3, 2};
    float value3 = t3[index3];
    cout << "Element at index t3: " << value3 << endl;

    // Test Reference and equality operator
    Tensor t4 = t3;
    cout << "Tensors 3 and 4 are the same: " << (t3 == t4) << endl;

    // Make same matrix as t3
    vector<float> values4(36);
    for(int i = 0; i < 36; i++) {
        values4[i] = i;
    }

    vector<int> dims4 = {3, 4, 3};

    Tensor t5(values4, dims4);
    cout << "Tensors 3 and 5 are the same: " << (t3 == t5) << endl;

    // Test Reshape
    vector<int> newShape3 = {1, 9, 4};
    t5.reshape(newShape3);
    cout << "Tensor 5 Shape: " << t5.getDimensionsString() << endl;

    // Test Tensor toString
    cout << t5.toString() << endl;

    // Test Matrix Addition
    // Test 1-dimensional tensors
    vector<float> values1D_1 = {1, 2, 3};
    vector<int> dims1D_1 = {3};
    Tensor t1D_1(values1D_1, dims1D_1);

    vector<float> values1D_2 = {4, 5, 6};
    vector<int> dims1D_2 = {3};
    Tensor t1D_2(values1D_2, dims1D_2);

    Tensor result1D = t1D_1 + t1D_2;
    cout << "Result1D: " << result1D.toString() << endl;

    // Test 2-dimensional tensors
    vector<float> values2D_1 = {1, 2, 3, 4};
    vector<int> dims2D_1 = {2, 2};
    Tensor t2D_1(values2D_1, dims2D_1);

    vector<float> values2D_2 = {5, 6, 7, 8};
    vector<int> dims2D_2 = {2, 2};
    Tensor t2D_2(values2D_2, dims2D_2);

    Tensor result2D = t2D_1 + t2D_2;
    cout << "Result2D: " << result2D.toString() << endl;

    // Test 3-dimensional tensors
    vector<float> values3D_1 = {1, 2, 3, 4, 5, 6, 7, 8};
    vector<int> dims3D_1 = {2, 2, 2};
    Tensor t3D_1(values3D_1, dims3D_1);

    vector<float> values3D_2 = {9, 10, 11, 12, 13, 14, 15, 16};
    vector<int> dims3D_2 = {2, 2, 2};
    Tensor t3D_2(values3D_2, dims3D_2);

    Tensor result3D = t3D_1 + t3D_2;
    cout << "Result3D:" << result3D.toString() << endl;

    // Matrix/Vector Multiplication
    Tensor t10 = t1;
    cout << "Tensor 1: " << t1.toString() << endl;
    cout << "Tensor 10: " << t10.toString() << endl;
    Tensor result4 = t1 * t10;
    cout << "Result4: " << result4.toString() << endl;

    Tensor t11 = t3D_1;
    Tensor result5 = t3D_1 * t11;
    cout << "Result5: " << result5.toString() << endl;

    vector<float> values5 = {0, 1, 2, 3, 4, 5, 6, 7};
    vector<int> dims5 = {2, 2, 2};

    Tensor tensorMultCuda(values5, dims5);
    Tensor newTensor = tensorMultCuda;
    cout << "Cuda Tensor" << tensorMultCuda.toString() << endl;
    cout << "Cuda Tensor 2" << newTensor.toString() << endl;

    Tensor multipliedTensor = tensorMultCuda * newTensor;
    cout << "One D Dot Product " << multipliedTensor.toString() << endl;

    // CUDA Testing
}
