#include <cuda_runtime.h>
#include <vector>
#include <iostream>

__global__ void addMatricesKernel(const float* a, const float* b, float* c, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < cols && idy < rows) {
        int index = idy * cols + idx;
        c[index] = a[index] + b[index];
    }
}

__global__ void multiplyMatricesKernel(const float* a, const float* b, float* c, int m, int n, int p) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < p) {
        float value = 0.0f;
        for (int k = 0; k < n; ++k) {
            value += a[row * n + k] * b[k * p + col];
        }
        c[row * p + col] = value;
    }
}

__global__ void transposeKernel(const float* input, float* output, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < cols && idy < rows) {
        output[idx * rows + idy] = input[idy * cols + idx];
    }
}

void addMatrices(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b, std::vector<std::vector<float>>& c) {
    int rows = a.size();
    int cols = a[0].size();
    float *d_a, *d_b, *d_c;
    size_t size = rows * cols * sizeof(float);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, a[0].data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b[0].data(), size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);
    addMatricesKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, rows, cols);

    cudaMemcpy(c[0].data(), d_c, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

void multiplyMatrices(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b, std::vector<std::vector<float>>& c) {
    int m = a.size();
    int n = a[0].size();
    int p = b[0].size();
    float *d_a, *d_b, *d_c;
    size_t size_a = m * n * sizeof(float);
    size_t size_b = n * p * sizeof(float);
    size_t size_c = m * p * sizeof(float);

    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    cudaMemcpy(d_a, a[0].data(), size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b[0].data(), size_b, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((p + threadsPerBlock.x - 1) / threadsPerBlock.x, (m + threadsPerBlock.y - 1) / threadsPerBlock.y);
    multiplyMatricesKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, m, n, p);

    cudaMemcpy(c[0].data(), d_c, size_c, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

void transpose(const std::vector<std::vector<float>>& input, std::vector<std::vector<float>>& output) {
    int rows = input.size();
    int cols = input[0].size();
    float *d_input, *d_output;
    size_t size = rows * cols * sizeof(float);

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, input[0].data(), size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);
    transposeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, rows, cols);

    cudaMemcpy(output[0].data(), d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}