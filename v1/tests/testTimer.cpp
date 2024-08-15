#include "../utils/Timer.cpp"
#include "../buildingBlocks/Tensor.cuh"
#include <iostream>

int main() {
    Timer t;

    float* values4 = new float[10000 * 10000];
    for(int i = 0; i < 10000 * 10000; i++) {
        values4[i] = i;
    }

    int* dims4 = new int[2];
    dims4[0] = 10000;
    dims4[1] = 10000;

    Tensor t1(values4, dims4 "cuda");
    Tensor t2 = t1;

    t.start();
    Tensor t3 = t2 * t1;
    t.stop();
    t.printLastTime();
    return 0;
}
