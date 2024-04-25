#include "../utils/Timer.cpp"
#include "../buildingBlocks/Tensor.cuh"
#include "../buildingBlocks/nn/linearlayer.cu"
#include "../buildingBlocks/nn/sigmoid.cu"
#include "../buildingBlocks/nn/mse.cpp"
#include <iostream>
#include <vector>
#include <random>


using namespace std;

random_device rd;
mt19937 gen(rd()); 


int main() {
    
    uniform_int_distribution<> distrib(0, 3);
    vector<float> datasetValues = {0, 0, 0, 1, 1, 0, 1, 1};
    vector<int> datasetDims = {4, 2};
    Tensor dataset(datasetValues, datasetDims, "cuda");

    vector<float> actualValues = {0, 1, 1, 0};
    vector<int> actualDims = {4};
    Tensor actual(actualValues, actualDims, "cuda");
    float lr = 0.1f;
    LinearLayer lin1(2, 1, lr);
    LinearLayer lin2(1, 1, lr);
    Sigmoid sig(1);
    MSE mse_loss(1);
    int epochs = 10;
    int iterations = 3000;
    //vector<int> datasetDims2 = {4, 2};
    for (int i = 0; i < epochs; i++) {
        float lossAvg = 0;
        cout << "Epoch: " << i << endl;
        for (int j = 0; j < iterations; j++) {
            //cout << "Iteration: " << i << endl;
            int index = distrib(gen);
            //cout << "Index: " << index << endl;
            //vector<int> datasetDims2 = {4, 2};
            float val1 = dataset[{index, 0}];
            float val2 = dataset[{index, 1}];
            Tensor input({val1, val2}, {2, 1}, "cuda");
            Tensor x = lin1.forward(input);
            //cout << x.toString()<< endl;
            x = lin2.forward(x);
            //cout << x.toString()<< endl;

            x = sig.forward(x);
            float prediction = x[{0, 0}];
            //cout << "Prediction: " << prediction << endl;
            // if(prediction > 0.5){
            //     prediction = 1;
            // } else {
            //     prediction = 0;
            // }
            float loss = mse_loss.forward_loss({prediction},{actual[{index}]});
            lossAvg += loss;
            //cout << "Loss: " << loss << endl;
            float dloss = mse_loss.backward_loss(prediction, actual[{index}]);
            Tensor gamma({dloss}, {1}, "cuda");
            Tensor y = sig.backward(gamma);
            //cout << lin2.toStringWeights() << endl;
            y = lin2.backward(y);
            //cout << lin2.toStringWeights() << endl;
            //cout << y.toString() << endl;
            y = lin1.backward(y);
        }
        cout << "Loss: " << lossAvg / iterations << endl;
        
    }


    
}
