#ifndef MATRIX_OPERATIONS_H
#define MATRIX_OPERATIONS_H

#include <vector>
#include <stdexcept>
#include <string>

void addMatrices(const std::vector<std::vector<float>> &a, const std::vector<std::vector<float>> &b, std::vector<std::vector<float>> &c);
void multiplyMatrices(const std::vector<std::vector<float>> &a, const std::vector<std::vector<float>> &b, std::vector<std::vector<float>> &c);
void print2DMatrix(const std::vector<std::vector<float>> &a);
void broadcastMultiply(const std::vector<float>& a, const std::vector<float>& b, std::vector<std::vector<float>>& c);

#endif 