#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <vector>
#include <utility>

void relu(std::vector<std::vector<float>> &a);
void softmax(std::vector<std::vector<float>> &a);
std::pair<float, std::vector<float>> softmaxLoss(const std::vector<float> &a, const std::vector<float> &y);

#endif
