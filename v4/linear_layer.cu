#include <cuda_runtime.h>
#include <vector>

using namespace std;

__global__ void linearLayerKernel(const float* input, const float* weights, const float* bias, float* output, int input_dim, int output_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < output_dim) {
        float sum = 0.0f;
        for (int i = 0; i < input_dim; ++i) {
            sum += input[i] * weights[i * output_dim + idx];
        }
        output[idx] = sum + bias[idx];
    }
}

void linearLayer(const vector<float>& input, const vector<float>& weights, const vector<float>& bias, vector<float>& output, int input_dim, int output_dim) {
    float *d_input, *d_weights, *d_bias, *d_output;
    size_t input_size = input_dim * sizeof(float);
    size_t weights_size = input_dim * output_dim * sizeof(float);
    size_t bias_size = output_dim * sizeof(float);
    size_t output_size = output_dim * sizeof(float);

    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_weights, weights_size);
    cudaMalloc(&d_bias, bias_size);
    cudaMalloc(&d_output, output_size);

    cudaMemcpy(d_input, input.data(), input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights.data(), weights_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias.data(), bias_size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (output_dim + threadsPerBlock - 1) / threadsPerBlock;
    linearLayerKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_weights, d_bias, d_output, input_dim, output_dim);

    cudaMemcpy(output.data(), d_output, output_size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_bias);
    cudaFree(d_output);
}