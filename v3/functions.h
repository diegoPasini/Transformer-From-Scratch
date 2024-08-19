#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <vector>
#include <utility>
void relu(std::vector<std::vector<float>> &a);
void relu_backward(std::vector<std::vector<float>> &d_output, const std::vector<std::vector<float>> &input);
void softmax(std::vector<std::vector<float>> &a);
void tranpose(const std::vector<std::vector<float>>& input, std::vector<std::vector<float>>& output);
std::pair<float, std::vector<float>> softmaxLoss(const std::vector<float> &a, const std::vector<float> &y);

#endif
