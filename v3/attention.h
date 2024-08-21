#ifndef SCALED_DOT_PRODUCT_ATTENTION_H
#define SCALED_DOT_PRODUCT_ATTENTION_H

#include <vector>

class ScaledDotProductAttention {
private:
    std::vector<std::vector<float>> queries;
    std::vector<std::vector<float>> keys;
    std::vector<std::vector<float>> values;
    std::vector<std::vector<float>> attention_weights;
    std::vector<std::vector<float>> outputs;

public:
    ScaledDotProductAttention() {}

    void applyUpperTriangularMask(std::vector<std::vector<float>>& matrix);

    std::vector<std::vector<float>> forward(const std::vector<std::vector<float>>& Q, const std::vector<std::vector<float>>& K, const std::vector<std::vector<float>>& V);

    std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<float>>, std::vector<std::vector<float>>> backward(const std::vector<std::vector<float>>& dL_dout);
};

#endif // SCALED_DOT_PRODUCT_ATTENTION_H
