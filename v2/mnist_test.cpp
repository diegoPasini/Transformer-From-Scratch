#include "mnist_reader.h"
#include <iostream>

using namespace std; 

int main() {
    string train_images_path = "../MNIST Dataset/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte";
    string train_labels_path = "../MNIST Dataset/train-labels-idx1-ubyte/train-labels-idx1-ubyte";
    
    vector<vector<float> > train_images = read_mnist_images(train_images_path);
    vector<uint8_t> train_labels = read_mnist_labels(train_labels_path);

    cout << "Number of training images: " << train_images.size() << endl;
    cout << "Number of training labels: " << train_labels.size() << endl;

    cout << "Label of the first image: " << static_cast<int>(train_labels[0]) << endl;
    cout << "First 10 pixel values of the first image: ";
    for (int i = 0; i < 10; i++) {
        cout << train_images[0][i] << " ";
    }
    cout << endl;

    return 0;
}