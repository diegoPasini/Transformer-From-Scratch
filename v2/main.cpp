#include "mnist_reader.h"
#include "convolution.cpp"
#include "functions.h"
#include "linear_layer.cpp"
#include "matrix_operations.h"
#include <iostream>
#include <random>
#include <chrono>

using namespace std;
using namespace std::chrono;

vector<vector<float>> get_batch(const vector<vector<float>>& images, int batch_size, int batch_index) {
    vector<vector<float>> batch(batch_size, vector<float>(images[0].size()));
    for (int i = 0; i < batch_size; ++i) {
        batch[i] = images[batch_index * batch_size + i];
    }
    return batch;
}

vector<uint8_t> get_batch_labels(const vector<uint8_t>& labels, int batch_size, int batch_index) {
    vector<uint8_t> batch_labels(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        batch_labels[i] = labels[batch_index * batch_size + i];
    }
    return batch_labels;
}

void normalize_images(vector<vector<float>>& images) {
    for (auto& image : images) {
        float mean = accumulate(image.begin(), image.end(), 0.0f) / image.size();
        float sq_sum = inner_product(image.begin(), image.end(), image.begin(), 0.0f);
        float stdev = sqrt(sq_sum / image.size() - mean * mean);
        for (auto& pixel : image) {
            pixel = (pixel - mean) / stdev;
        }
    }
}

void test_model(LinearLayer& fc1, vector<vector<float>> test_images, const vector<uint8_t>& test_labels, int batch_size) {
    cout << "RUNNING TEST LOOP" << endl;
    int correct_predictions = 0;
    float total_loss = 0.0f;
    normalize_images(test_images); // Normalize test images before testing
    for (int batch_index = 0; batch_index < test_images.size() / batch_size; ++batch_index) {
        vector<vector<float>> batch_images = get_batch(test_images, batch_size, batch_index);
        vector<uint8_t> batch_labels = get_batch_labels(test_labels, batch_size, batch_index);

        vector<vector<float>> fc1_output = fc1.forward(batch_images);
        softmax(fc1_output);

        for (int i = 0; i < batch_size; ++i) {
            vector<float> label_one_hot(10, 0.0f);
            label_one_hot[batch_labels[i]] = 1.0f;
            auto [loss, gradient] = softmaxLoss(fc1_output[i], label_one_hot);
            total_loss += loss;

            int predicted_label = distance(fc1_output[i].begin(), max_element(fc1_output[i].begin(), fc1_output[i].end()));
            if (predicted_label == batch_labels[i]) {
                correct_predictions++;
            }
        }
    }

    float accuracy = static_cast<float>(correct_predictions) / test_images.size();
    cout << "Test Accuracy: " << accuracy << endl;
    cout << "Test Loss: " << total_loss / test_images.size() << endl;
}

int main() {
    // Load MNIST dataset
    string train_images_path = "../MNIST Dataset/train-images-idx3-ubyte/train-images-idx3-ubyte";
    string train_labels_path = "../MNIST Dataset/train-labels-idx1-ubyte/train-labels-idx1-ubyte";
    string test_images_path = "../MNIST Dataset/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte";
    string test_labels_path = "../MNIST Dataset/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte";
    
    vector<vector<float>> train_images = read_mnist_images(train_images_path);
    vector<uint8_t> train_labels = read_mnist_labels(train_labels_path);
    vector<vector<float>> test_images = read_mnist_images(test_images_path);
    vector<uint8_t> test_labels = read_mnist_labels(test_labels_path);

    cout << "Number of training images: " << train_images.size() << endl;
    cout << "Number of training labels: " << train_labels.size() << endl;
    cout << "Number of test images: " << test_images.size() << endl;
    cout << "Number of test labels: " << test_labels.size() << endl;

    LinearLayer fc1(28 * 28, 10); 

    int batch_size = 100;
    int num_epochs = 10;
    float learning_rate = 0.0001;
    int microbatch_size = 10; 

    int t = 1;
    int log_interval = 10; // Log loss every 10 batches

    normalize_images(train_images); // Normalize training images before training

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        cout << "Starting epoch " << epoch + 1 << " of " << num_epochs << endl;
        auto epoch_start = high_resolution_clock::now(); // Start timer for epoch

        for (int batch_index = 0; batch_index < train_images.size() / batch_size; ++batch_index) {
            auto start = high_resolution_clock::now(); // Start timer for batch

            vector<vector<float>> batch_images = get_batch(train_images, batch_size, batch_index);
            vector<uint8_t> batch_labels = get_batch_labels(train_labels, batch_size, batch_index);

            vector<vector<float>> fc1_output = fc1.forward(batch_images);
            relu(fc1_output);

            // BACKPROPOGATION
            vector<vector<float>> d_fc1_output(batch_size, vector<float>(10));
        
            float total_loss = 0.0f;
            vector<float> average_gradient(10, 0.0f);
            for (int i = 0; i < batch_size; ++i) {
                vector<float> label_one_hot(10, 0.0f);
                label_one_hot[batch_labels[i]] = 1.0f;
                pair<float, vector<float>> loss_and_gradient = softmaxLoss(fc1_output[i], label_one_hot);
                float loss = loss_and_gradient.first;
                vector<float> gradient = loss_and_gradient.second;
                d_fc1_output[i] = gradient;
                total_loss += loss;

                for (int j = 0; j < 10; ++j) {
                    average_gradient[j] += gradient[j];
                }
            }

            for (int j = 0; j < 10; ++j) {
                average_gradient[j] /= batch_size;
            }

            if (batch_index % log_interval == 0) {
                float average_loss = total_loss / batch_size;
                cout << "Batch " << batch_index + 1 << " Average Loss: " << average_loss << endl;
            }

            fc1.backward(d_fc1_output, learning_rate, t);

            ++t;
        }

        auto epoch_end = high_resolution_clock::now(); // End timer for epoch
        auto epoch_duration = duration_cast<milliseconds>(epoch_end - epoch_start);
        cout << "Epoch " << epoch + 1 << " completed in " << epoch_duration.count() << " milliseconds." << endl;

        test_model(fc1, test_images, test_labels, batch_size);
    }

    return 0;
}
