#include "../Tensor.cuh"
#include <string>

class Layer {
public:
    
    virtual ~Layer() = default;
    virtual Tensor forward(Tensor x);
    virtual Tensor backward(Tensor x);
    virtual string toString();

private:

    int inputFeatures;
    int outputFeatures;

};
