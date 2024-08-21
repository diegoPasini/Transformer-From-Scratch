#ifndef LINEAR_LAYER_H
#define LINEAR_LAYER_H

#include <vector>

class LinearLayer {
private:
    int input_features;
    int output_features;
    bool use_bias;
    std::vector<std::vector<float>> weights;
    std::vector<float> bias;
    std::vector<std::vector<float>> inputs; 
    std::vector<std::vector<float>> outputs;

    std::vector<std::vector<float>> m_weights;
    std::vector<std::vector<float>> v_weights;
    std::vector<float> m_bias;
    std::vector<float> v_bias;
    float beta1;
    float beta2;
    float epsilon;

    void initialize_weights();

public:
    LinearLayer(int in_features, int out_features, bool bias = true);

    std::vector<std::vector<float>> forward(const std::vector<std::vector<float>>& batch_inputs);

    std::vector<std::vector<float>> backward(const std::vector<std::vector<float>>& d_outputs, float learning_rate, int t);

    void print_weights() const;

    void print_bias() const;
};

#endif 
