#include <vector>
#include "../Tensor.cuh"
#include "Layer.h"
#include <memory>
#include "losses.cpp"
#include "loss.h"

using namespace std;

class Module {

    private:
        
        std::vector<std::unique_ptr<Layer>> layers;
        std::unique_ptr<Loss> loss;

    public:

        void forward(Tensor input) {
            std::unique_ptr<Tensor> temp = std::make_unique<Tensor>(layers[0]->forward(input));
            for (int i = 1; i < layers.size(); i++) {
                temp = std::make_unique<Tensor>(layers[i]->forward(*temp));
            }
        }

        void backward() {
            
            // Tensor loss_input = loss.forward_loss(Tensor loss)
            // layers[layers.size() - 1].backward(loss_input);
            // for (int i = layers.size() - 1; i >= 0; i--) {
                
            //     temp = make_unique<Tensor>(layers[i]->backward(*temp));
            // }
        }
};
