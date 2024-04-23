#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <cmath>
#include "Layer.cpp"

class SoftmaxLayer : public Layer {
    public:

        int size;

        SoftmaxLayer(int inputSize) {
            size = inputSize;
        }

        __global__ void forward(float* a, float* b, int size){
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            float sum = 0.0f;

            // sum of exps
            for (int j = 0; j < size; j++) {
                sum += exp(a[j]);
            }

            // division
            if (i < size) {
                b[i] = exp(a[i]) / sum;
            }
        }

        __global__ void backward(float* outputDerivatives, float* inputDerivatives) {
            // inputDerivatives is for the vector inputted to softmax; it's what we want to compute
            // outputDerivatives is the vector given to us; it's the derivatives of the output of the softmax layer w/ respect to the loss
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            if (i < size) {
                inputDerivatives[i] = outputDerivatives[i] * (1 - outputDerivatives[i]);
            }
        }
};
