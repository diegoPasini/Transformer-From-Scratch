#include "../utils/Timer.cpp"
#include "../buildingBlocks/Tensor.cuh"
#include <iostream>
#include <vector>
#include <string>

using namespace std;

int main() {
    Timer t;

    std::vector<float> values4(1000 * 1000, 0.0f);
    
    int s = 1000 * 1000;
    std::vector<int> newVec = {1000, 1000};
    string s1 = "cuda";
    t.start();
    Tensor t1(values4, newVec, s1);
    t.stop();
    t.printLastTime();
    Timer tim1;
    tim1.start();
    Tensor t2 = t1;
    tim1.stop();
    tim1.printLastTime();

    Timer tim3;
    tim3.start();
    Tensor t3 = t2 * t1;
    tim3.stop();
    tim3.printLastTime();
    return 0;
}
