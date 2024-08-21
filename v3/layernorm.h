#ifndef LAYERNORM_H
#define LAYERNORM_H

#include <vector>

class Layernorm {
private:
    float gamma;
    float beta;

    std::vector<float> mean;
    std::vector<float> variance;

public:
    Layernorm();

    std::vector<std::vector<float>> forward(const std::vector<std::vector<float>>& inputs);

    std::vector<std::vector<float>> backward(const std::vector<std::vector<float>>& d_outputs, float learning_rate, int t);
};

#endif // LAYERNORM_H
