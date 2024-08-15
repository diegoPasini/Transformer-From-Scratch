#include "mnist_reader.h"
#include "convolution.cpp"
#include "functions.h"
#include "linear_layer.cpp"
#include "matrix_operations.h"
#include <iostream>

using namespace std;

int main() {
    // Load MNIST dataset
    string train_images_path = "../MNIST Dataset/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte";
    string train_labels_path = "../MNIST Dataset/train-labels-idx1-ubyte/train-labels-idx1-ubyte";
    
    vector<vector<float> > train_images = read_mnist_images(train_images_path);
    vector<uint8_t> train_labels = read_mnist_labels(train_labels_path);

    cout << "Number of training images: " << train_images.size() << endl;
    cout << "Number of training labels: " << train_labels.size() << endl;

    Convolution conv1(1, 8, 3, 1, 1); // 1 input channel, 8 output channels, 3x3 kernel, stride 1, padding 1
    Convolution conv2(8, 16, 3, 1, 1); // 8 input channels, 16 output channels, 3x3 kernel, stride 1, padding 1
    LinearLayer fc1(16 * 28 * 28, 128); // Fully connected layer, input size 16*28*28, output size 128
    LinearLayer fc2(128, 10); // Fully connected layer, input size 128, output size 10

    vector<vector<vector<float> > > input_image(1, vector<vector<float> >(28, vector<float>(28)));
    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            input_image[0][i][j] = train_images[0][i * 28 + j];
        }
    }

    auto conv1_output = conv1.forward(input_image);
    for (auto& channel : conv1_output) {
        relu(channel);
    }
    auto conv2_output = conv2.forward(conv1_output);
    for (auto& channel : conv2_output) {
        relu(channel);
    }

    vector<vector<float> > flattened_output(1);
    for (const auto& channel : conv2_output) {
        for (const auto& row : channel) {
            for (float val : row) {
                flattened_output[0].push_back(val);
            }
        }
    }

    vector<vector<float> > fc1_output = fc1.forward(flattened_output);
    relu(fc1_output);
    vector<vector<float> > fc2_output = fc2.forward(fc1_output);
    softmax(fc2_output);

    cout << "Output of the network: ";
    for (float val : fc2_output[0]) {
        cout << val << " ";
    }
    cout << endl;

    return 0;
}
