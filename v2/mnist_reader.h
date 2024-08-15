#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cassert>

using namespace std;

// Download Dataset at https://www.kaggle.com/datasets/hojjatk/mnist-dataset?resource=download

uint32_t read_int(ifstream &file) {
    uint32_t num = 0;
    file.read(reinterpret_cast<char *>(&num), sizeof(num));
    return __builtin_bswap32(num); 
}

vector<vector<float> > read_mnist_images(const string &path) {
    ifstream file(path, ios::binary);
    assert(file.is_open());

    uint32_t magic_number = read_int(file);
    uint32_t num_images = read_int(file);
    uint32_t num_rows = read_int(file);
    uint32_t num_cols = read_int(file);

    vector<vector<float> > images(num_images, vector<float>(num_rows * num_cols));

    for (size_t i = 0; i < num_images; i++) {
        for (size_t j = 0; j < num_rows * num_cols; j++) {
            uint8_t pixel = 0;
            file.read(reinterpret_cast<char *>(&pixel), sizeof(pixel));
            images[i][j] = pixel / 255.0f;  // Normalize pixel values to [0, 1]
        }
    }

    return images;
}


vector<uint8_t> read_mnist_labels(const string &path) {
    ifstream file(path, ios::binary);
    assert(file.is_open());

    uint32_t magic_number = read_int(file);
    uint32_t num_labels = read_int(file);

    vector<uint8_t> labels(num_labels);

    for (size_t i = 0; i < num_labels; i++) {
        uint8_t label = 0;
        file.read(reinterpret_cast<char *>(&label), sizeof(label));
        labels[i] = label;
    }

    return labels;
}