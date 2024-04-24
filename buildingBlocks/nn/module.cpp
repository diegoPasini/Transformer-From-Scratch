#include <vector>
#include "../Tensor.cuh"
#include "Layer.cpp"
#include <memory>
#include "losses.cpp"
#include "loss.cpp"

using namespace std;

class Module {

    private:
        
        std::vector<std::unique_ptr<Layer>> layers;
        std::unique_ptr<Loss> loss;

    public:

        void forward() {
            // each layer will have an input
        }

        void backward() {
            // forward();
            // Tensor loss_input = loss.forward_loss(Tensor loss)
            // layers[layers.size() - 1].backward(loss_input);
            // for (int i = layers.size() - 1; i >= 0; i--) {
                
            //     temp = make_unique<Tensor>(layers[i]->backward(*temp));
            // }
        }
};
