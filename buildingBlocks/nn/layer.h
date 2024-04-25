#include "../Tensor.cuh"
#include <string>
#ifndef LAYER_H  
#define LAYER_H 

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

#endif // LAYER_H

