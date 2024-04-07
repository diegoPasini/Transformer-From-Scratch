#include "../utils/Timer.cpp"
#include "../buildingBlocks/Tensor.cuh"
#include <iostream>

int main() {
    Timer t;

    t.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(201));
    t.stop();
    t.printLastTime();

    t.start();
    std::this_thread::sleep_for(std::chrono::seconds(3));
    t.stop();

    t.printLoggedTimes(); // This will print 2 seconds and 3 seconds

    return 0;
}
