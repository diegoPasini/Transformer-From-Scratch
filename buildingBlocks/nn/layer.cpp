#include "../Tensor.cuh"
#include <string>

class Layer {
public:
    
    virtual ~Layer() = default;
    virtual Tensor forward(Tensor x) = 0;
    virtual Tensor backward(Tensor x) = 0;

private:

    int inputFeatures;
    int outputFeatures;
    // Tensor weights;

};
