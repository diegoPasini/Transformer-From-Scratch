#include <vector>
#include "../Tensor.cuh"
#include "Layer.cpp"
#include <memory>
#include "losses.cpp"

using namespace std;

class Module {

    private:
        
        std::vector<std::unique_ptr<Layer>> layers;
        

    public:

        void backward() {
            unique_ptr<Tensor> temp;
            for (int i = layers.size() - 1; i >= 0; i--) {
                temp = make_unique<Tensor>(layers[i]->backward(*temp));
            }
        }
}
