#include <iostream>
#include <vector>
#include "attention.h"
#include "feedforward.cpp"
#include "embedding.cpp"
#include "functions.h"
#include "multi_head_attention.cpp"
#include "linear_layer.h"

using namespace std;

const int vocab_size = 10000;
const int embedding_dim = 512;
const int num_heads = 8;
const int feedforward_dim = 2048;
const int num_layers = 6;
const float learning_rate = 0.001;
const int num_epochs = 10;

int main() {
    // Initialize components
    Embedding embedding_layer(vocab_size, embedding_dim);
    vector<MultiHeadAttention> attention_layers;
    vector<Feedforward> feedforward_layers;

    for (int i = 0; i < num_layers; ++i) {
        attention_layers.emplace_back(num_heads, embedding_dim);
        feedforward_layers.emplace_back(embedding_dim, feedforward_dim);
    }


    vector<vector<int>> input_tokens = {{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}};
    vector<vector<vector<float>>> embedded_inputs = embedding_layer.forward(input_tokens);

    vector<vector<float>> x = embedded_inputs[0];
    for (int i = 0; i < num_layers; ++i) {
        x = attention_layers[i].forward(x, x, x);
        x = feedforward_layers[i].forward(x);
    }

    vector<float> target = {0.1, 0.2, 0.3, 0.4, 0.5};

    float loss;
    vector<float> d_loss;
    tie(loss, d_loss) = softmaxLoss(x[0], target);

    cout << "Initial loss: " << loss << endl;

    vector<vector<float>> d_outputs = {d_loss};
    for (int i = num_layers - 1; i >= 0; --i) {
        d_outputs = feedforward_layers[i].backward(d_outputs, learning_rate, 1);
        d_outputs = attention_layers[i].backward(d_outputs, learning_rate, 1);
    }

    embedding_layer.backward(d_outputs, input_tokens[0], learning_rate);

    cout << "Training complete." << endl;

    return 0;
}
