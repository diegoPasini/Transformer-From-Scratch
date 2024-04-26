#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <string>
#include "layer.h"
//#include "linearlayer.cu"
#include <cmath>
#include <numeric>
#include "../Tensor.cuh"


class Sigmoid : public Layer {
    public:
        int input_features;
        int output_features = 1;
        float output;
        Sigmoid(int input_features) {
            this->input_features = input_features;
        }
        
        ~Sigmoid() {}

        // NEED CUDA IMPLEMENTATION FOR THIS
        Tensor forward(Tensor x) {
            float sum_of_elems = 0;
            for(int i = 0; i < x.getTotalValues(); i++) {
                sum_of_elems += x.getValues()[i];
            }
            output = 1 / (1 + exp(-sum_of_elems));
            return Tensor({output}, {1}, "cuda");
        }   


        Tensor backward(Tensor gammaPrev) {
            gammaPrev = output * (1-output) * gammaPrev;
            vector<float> values = gammaPrev.getValues();
            vector<float> newValues(input_features);
            for (int i = 0; i < input_features; i++) {
                newValues[i] = values[0];
            }
            return Tensor(newValues, {1, input_features}, "cuda");;
        }
};