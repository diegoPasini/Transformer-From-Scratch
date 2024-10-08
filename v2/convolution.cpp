#include <vector>
#include <iostream>
#include <cmath>
#include <random>

using namespace std;

class Convolution {
private:
    int input_channels;
    int output_channels;
    int kernel_size;
    int stride;
    int padding;
    vector<vector<vector<vector<float>>>> kernels; // [output_channels][input_channels][kernel_size][kernel_size]
    vector<float> biases;

    void initialize_kernels() {
        float k = sqrt(2.0f / input_channels);
        random_device rd;
        mt19937 gen(rd());
        normal_distribution<float> distr(0.0f, k);
        
        kernels.resize(output_channels);
        for (auto& kernel : kernels) {
            kernel.resize(input_channels);
            for (auto& channel : kernel) {
                channel.resize(kernel_size, vector<float>(kernel_size));
                for (auto& row : channel) {
                    for (auto& val : row) {
                        val = distr(gen);
                    }
                }
            }
        }

        biases.resize(output_channels);
        for (auto& bias : biases) {
            bias = distr(gen);
        }
    }

    vector<vector<float>> pad_input(const vector<vector<float>>& input) {
        int padded_size = input.size() + 2 * padding;
        vector<vector<float>> padded_input(padded_size, vector<float>(padded_size, 0.0f));

        for (int i = 0; i < input.size(); ++i) {
            for (int j = 0; j < input[0].size(); ++j) {
                padded_input[i + padding][j + padding] = input[i][j];
            }
        }

        return padded_input;
    }

public:
    Convolution(int in_channels, int out_channels, int k_size, int strd, int pad)
        : input_channels(in_channels), output_channels(out_channels), kernel_size(k_size), stride(strd), padding(pad) {
        initialize_kernels();
    }

    vector<vector<vector<float>>> forward(const vector<vector<vector<float>>>& input) {
        int input_height = input[0].size();
        int input_width = input[0][0].size();
        int output_height = (input_height - kernel_size + 2 * padding) / stride + 1;
        int output_width = (input_width - kernel_size + 2 * padding) / stride + 1;

        vector<vector<vector<float>>> output(output_channels, vector<vector<float>>(output_height, vector<float>(output_width, 0.0f)));

        for (int oc = 0; oc < output_channels; ++oc) {
            for (int oh = 0; oh < output_height; ++oh) {
                for (int ow = 0; ow < output_width; ++ow) {
                    float sum = biases[oc];
                    for (int ic = 0; ic < input_channels; ++ic) {
                        for (int kh = 0; kh < kernel_size; ++kh) {
                            for (int kw = 0; kw < kernel_size; ++kw) {
                                int ih = oh * stride + kh - padding;
                                int iw = ow * stride + kw - padding;
                                if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                                    sum += input[ic][ih][iw] * kernels[oc][ic][kh][kw];
                                }
                            }
                        }
                    }
                    output[oc][oh][ow] = sum;
                }
            }
        }

        return output;
    }

    vector<vector<vector<float>>> backward(const vector<vector<vector<float>>>& d_output, const vector<vector<vector<float>>>& input) {
        int input_height = input[0].size();
        int input_width = input[0][0].size();
        int output_height = d_output[0].size();
        int output_width = d_output[0][0].size();

        vector<vector<vector<float>>> d_input(input_channels, vector<vector<float>>(input_height, vector<float>(input_width, 0.0f)));
        vector<vector<vector<vector<float>>>> d_kernels(output_channels, vector<vector<vector<float>>>(input_channels, vector<vector<float>>(kernel_size, vector<float>(kernel_size, 0.0f))));
        vector<float> d_biases(output_channels, 0.0f);

        for (int oc = 0; oc < output_channels; ++oc) {
            for (int oh = 0; oh < output_height; ++oh) {
                for (int ow = 0; ow < output_width; ++ow) {
                    float d_out_val = d_output[oc][oh][ow];
                    d_biases[oc] += d_out_val;
                    for (int ic = 0; ic < input_channels; ++ic) {
                        for (int kh = 0; kh < kernel_size; ++kh) {
                            for (int kw = 0; kw < kernel_size; ++kw) {
                                int ih = oh * stride + kh - padding;
                                int iw = ow * stride + kw - padding;
                                if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                                    d_kernels[oc][ic][kh][kw] += d_out_val * input[ic][ih][iw];
                                    d_input[ic][ih][iw] += d_out_val * kernels[oc][ic][kh][kw];
                                }
                            }
                        }
                    }
                }
            }
        }

        // Update kernels and biases
        for (int oc = 0; oc < output_channels; ++oc) {
            biases[oc] -= d_biases[oc];
            for (int ic = 0; ic < input_channels; ++ic) {
                for (int kh = 0; kh < kernel_size; ++kh) {
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        kernels[oc][ic][kh][kw] -= d_kernels[oc][ic][kh][kw];
                    }
                }
            }
        }

        return d_input;
    }

    void print_kernels() const {
        for (const auto& kernel : kernels) {
            for (const auto& channel : kernel) {
                for (const auto& row : channel) {
                    for (float val : row) {
                        cout << val << " ";
                    }
                    cout << endl;
                }
                cout << endl;
            }
            cout << endl;
        }
    }

    void print_biases() const {
        for (float val : biases) {
            cout << val << " ";
        }
        cout << endl;
    }
};