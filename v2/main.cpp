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
    cout << "Number of training images: " << test_images.size() << endl;
    cout << "Number of training labels: " << test_labels.size() << endl;

    Convolution conv1(1, 8, 3, 1, 1); // 1 input channel, 8 output channels, 3x3 kernel, stride 1, padding 1
    Convolution conv2(8, 16, 3, 1, 1); // 8 input channels, 16 output channels, 3x3 kernel, stride 1, padding 1
    LinearLayer fc1(16 * 28 * 28, 128); // Fully connected layer, input size 16*28*28, output size 128
    LinearLayer fc2(128, 10); // Fully connected layer, input size 128, output size 10

    int batch_size = 100;
    int num_epochs = 10;
    float learning_rate = 0.001;
    int microbatch_size = 10; // Define microbatch size

    // For Adam:
    int t = 1;

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        cout << "Starting epoch " << epoch + 1 << " of " << num_epochs << endl;
        for (int batch_index = 0; batch_index < train_images.size() / batch_size; ++batch_index) {
            auto start = high_resolution_clock::now(); // Start timer

            cout << "Processing Training Batch " << batch_index + 1 << " of " << train_images.size() / batch_size << endl;
            
            vector<vector<float>> batch_images = get_batch(train_images, batch_size, batch_index);
            vector<uint8_t> batch_labels = get_batch_labels(train_labels, batch_size, batch_index);

            vector<vector<vector<vector<float>>>> conv1_output(batch_size);
            for (int i = 0; i < batch_size; ++i) {
                vector<vector<vector<float>>> input_image(1, vector<vector<float>>(28, vector<float>(28)));
                for (int j = 0; j < 28; ++j) {
                    for (int k = 0; k < 28; ++k) {
                        input_image[0][j][k] = batch_images[i][j * 28 + k];
                    }
                }
                conv1_output[i] = conv1.forward(input_image);
                for (auto& channel : conv1_output[i]) {
                    relu(channel);
                }
            }

            vector<vector<vector<vector<float>>>> conv2_output(batch_size);
            for (int i = 0; i < batch_size; ++i) {
                conv2_output[i] = conv2.forward(conv1_output[i]);
                for (auto& channel : conv2_output[i]) {
                    relu(channel);
                }
            }

            vector<vector<float>> flattened_output(batch_size, vector<float>(16 * 28 * 28));
            for (int i = 0; i < batch_size; ++i) {
                int index = 0;
                for (const auto& channel : conv2_output[i]) {
                    for (const auto& row : channel) {
                        for (float val : row) {
                            flattened_output[i][index++] = val;
                        }
                    }
                }
            }

            vector<vector<float>> fc1_output = fc1.forward(flattened_output);
            relu(fc1_output);
            vector<vector<float>> fc2_output = fc2.forward(fc1_output);
            softmax(fc2_output);

            // BACKPROPOGATION
            cout << "Starting Backpropogation" << endl;
            vector<vector<float>> d_fc2_output(batch_size, vector<float>(10));
            vector<vector<float>> d_fc1_output(batch_size, vector<float>(128));
            vector<vector<float>> d_flattened_output(batch_size, vector<float>(16 * 28 * 28));
        
            float total_loss = 0.0f;
            for (int i = 0; i < batch_size; ++i) {
                vector<float> label_one_hot(10, 0.0f);
                label_one_hot[batch_labels[i]] = 1.0f;
                pair<float, vector<float>> loss_and_gradient = softmaxLoss(fc2_output[i], label_one_hot);
                float loss = loss_and_gradient.first;
                vector<float> gradient = loss_and_gradient.second;
                d_fc2_output[i] = gradient;
                total_loss += loss;

                // Print the gradients
                cout << "Gradients for sample " << i + 1 << ": ";
                for (const float& grad : gradient) {
                    cout << grad << " ";
                }
                cout << endl;
            }
            float average_loss = total_loss / batch_size;
            cout << "Average Loss for batch: " << average_loss << endl;

            for (int i = 0; i < batch_size; ++i) {
                d_fc1_output[i] = fc2.backward(d_fc2_output[i], learning_rate, t);
            }

            for (int i = 0; i < batch_size; ++i) {
                d_flattened_output[i] = fc1.backward(d_fc1_output[i], learning_rate, t);
            }

            // Backpropagation for conv2
            vector<vector<vector<vector<float>>>> d_conv2_output(batch_size);
            for (int i = 0; i < batch_size; ++i) {
                d_conv2_output[i] = vector<vector<vector<float>>>(16, vector<vector<float>>(28, vector<float>(28, 0.0f)));
                int index = 0;
                for (int c = 0; c < 16; ++c) {
                    for (int h = 0; h < 28; ++h) {
                        for (int w = 0; w < 28; ++w) {
                            d_conv2_output[i][c][h][w] = d_flattened_output[i][index++];
                        }
                    }
                }
                for (auto& channel : d_conv2_output[i]) {
                    relu_backward(channel, conv2_output[i][&channel - &d_conv2_output[i][0]]);
                }
                d_conv2_output[i] = conv2.backward(d_conv2_output[i], conv1_output[i]);
            }

            // Backpropagation for conv1
            vector<vector<vector<vector<float>>>> d_conv1_output(batch_size);
            for (int i = 0; i < batch_size; ++i) {
                for (auto& channel : d_conv2_output[i]) {
                    relu_backward(channel, conv1_output[i][&channel - &d_conv2_output[i][0]]);
                }
                d_conv1_output[i] = conv1.backward(d_conv2_output[i], vector<vector<vector<float>>>(1, vector<vector<float>>(28, vector<float>(28))));
            }

            ++t;

            auto end = high_resolution_clock::now(); // End timer
            auto duration = duration_cast<milliseconds>(end - start);
            cout << "Training step took " << duration.count() << " milliseconds." << endl;
        }
        cout << "Epoch " << epoch + 1 << " completed." << endl;
    }

    // Testing loop
    cout << "RUNNING TEST LOOP" << endl;
    int correct_predictions = 0;
    float total_loss = 0.0f;
    for (int batch_index = 0; batch_index < test_images.size() / batch_size; ++batch_index) {
        cout << "Image " << batch_index << " of " << test_images.size() << endl;
        vector<vector<float>> batch_images = get_batch(test_images, batch_size, batch_index);
        vector<uint8_t> batch_labels = get_batch_labels(test_labels, batch_size, batch_index);

        vector<vector<vector<vector<float>>>> conv1_output(batch_size);
        for (int i = 0; i < batch_size; ++i) {
            vector<vector<vector<float>>> input_image(1, vector<vector<float>>(28, vector<float>(28)));
            for (int j = 0; j < 28; ++j) {
                for (int k = 0; k < 28; ++k) {
                    input_image[0][j][k] = batch_images[i][j * 28 + k];
                }
            }
            conv1_output[i] = conv1.forward(input_image);
            for (auto& channel : conv1_output[i]) {
                relu(channel);
            }
        }

        vector<vector<vector<vector<float>>>> conv2_output(batch_size);
        for (int i = 0; i < batch_size; ++i) {
            conv2_output[i] = conv2.forward(conv1_output[i]);
            for (auto& channel : conv2_output[i]) {
                relu(channel);
            }
        }

        vector<vector<float>> flattened_output(batch_size, vector<float>(16 * 28 * 28));
        for (int i = 0; i < batch_size; ++i) {
            int index = 0;
            for (const auto& channel : conv2_output[i]) {
                for (const auto& row : channel) {
                    for (float val : row) {
                        flattened_output[i][index++] = val;
                    }
                }
            }
        }

        vector<vector<float>> fc1_output = fc1.forward(flattened_output);
        relu(fc1_output);
        vector<vector<float>> fc2_output = fc2.forward(fc1_output);
        softmax(fc2_output);

        for (int i = 0; i < batch_size; ++i) {
            vector<float> label_one_hot(10, 0.0f);
            label_one_hot[batch_labels[i]] = 1.0f;
            auto [loss, gradient] = softmaxLoss(fc2_output[i], label_one_hot);
            total_loss += loss;

            int predicted_label = distance(fc2_output[i].begin(), max_element(fc2_output[i].begin(), fc2_output[i].end()));
            if (predicted_label == batch_labels[i]) {
                correct_predictions++;
            }
        }
    }

    float accuracy = static_cast<float>(correct_predictions) / test_images.size();
    cout << "Test Accuracy: " << accuracy << endl;
    cout << "Test Loss: " << total_loss / test_images.size() << endl;

    return 0;
}
