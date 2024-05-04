#include "../utils/Timer.cpp"
#include "../buildingBlocks/Tensor.cuh"
#include "../buildingBlocks/nn/linearlayer.cu"
#include "../buildingBlocks/nn/sigmoid.cu"
#include "../buildingBlocks/nn/bcelogits.cpp"
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
    float lr = 0.75f;
    LinearLayer lin1(2, 4, lr);
    LinearLayer lin2(4, 4, lr);

    LinearLayer lin3(4, 1, lr);
    Sigmoid sig(1);
    BCEWithLogitsLoss loss(1);
    int epochs = 10;
    int iterations = 1000;
    for (int i = 0; i < epochs; i++) {
        float lossAvg = 0;
        cout << "Epoch: " << i << endl;
        for (int j = 0; j < iterations; j++) {
            vector<float> predictions(4);
            for (int k = 0; k < 4; k++) {
                float val1 = dataset[{k, 0}];
                float val2 = dataset[{k, 1}];
                Tensor input({val1, val2}, {2, 1}, "cuda");
                Tensor x = lin1.forward(input);
                x = lin2.forward(x);
                x = lin3.forward(x);
                x = sig.forward(x);

                float prediction = x[{0, 0}];
                predictions[k] = prediction;
            }

            float lossValue = loss.forward_loss(predictions, actualValues);
            lossAvg += lossValue;
            

            //cout << "Loss: " << lossValue << endl;
            for (int l = 0; l < 4; l++) {
                float dloss = loss.backward_loss(predictions[l], actualValues[l]);
                Tensor gamma({dloss}, {1, 1}, "cuda");
                Tensor y = sig.backward(gamma);
                y = lin3.backward(y);
                y = lin2.backward(y);
                y = lin1.backward(y);
            }
        }

        cout << "Loss: " << lossAvg/iterations << endl;
    }

    // After training, predicting XOR values for all inputs
    vector<float> inputs = {0, 0, 0, 1, 1, 0, 1, 1};
    cout << "Predicting XOR values:" << endl;
    for (int i = 0; i < 4; i++) {
        Tensor input({inputs[i * 2], inputs[i * 2 + 1]}, {2, 1}, "cuda");
        Tensor x = lin1.forward(input);
        x = lin2.forward(x);
        x = lin3.forward(x);
        x = sig.forward(x);
        float prediction = x[{0, 0}];
        cout << "Input: " << inputs[i * 2] << ", " << inputs[i * 2 + 1] << " - Prediction: " << prediction << endl;
    }    
}
