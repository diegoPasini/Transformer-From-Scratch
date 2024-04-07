#include "../utils/Logger.cpp"
#include "../buildingBlocks/Tensor.cuh"
#include <iostream>

int main() {

    Logger::log("Hello, World!");
    Logger::debug("Hello, World!");
    Logger::info("Hello, World!");
    Logger::warning("Hello, World!");
    Logger::error("Hello, World!");
    Logger::logRaw("Raw Hello World!");
    
    return 0;
}