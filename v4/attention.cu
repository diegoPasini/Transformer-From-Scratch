#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include "matrix_operations.cu"

__global__ void applyUpperTriangularMaskKernel(float* matrix, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < cols && idy < rows && idx < idy) {
        matrix[idy * cols + idx] = -INFINITY;
    }
}

__global__ void softmaxKernel(float* matrix, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < cols && idy < rows) {
        float max_val = -INFINITY;
        for (int i = 0; i < cols; ++i) {
            max_val = fmaxf(max_val, matrix[idy * cols + i]);
        }
        float sum_exp = 0.0f;
        for (int i = 0; i < cols; ++i) {
            matrix[idy * cols + i] = expf(matrix[idy * cols + i] - max_val);
            sum_exp += matrix[idy * cols + i];
        }
        for (int i = 0; i < cols; ++i) {
            matrix[idy * cols + i] /= sum_exp;
        }
    }
}

void applyUpperTriangularMask(std::vector<std::vector<float>>& matrix) {
    int rows = matrix.size();
    int cols = matrix[0].size();
    float* d_matrix;
    size_t size = rows * cols * sizeof(float);

    cudaMalloc(&d_matrix, size);
    cudaMemcpy(d_matrix, matrix[0].data(), size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);
    applyUpperTriangularMaskKernel<<<blocksPerGrid, threadsPerBlock>>>(d_matrix, rows, cols);

    cudaMemcpy(matrix[0].data(), d_matrix, size, cudaMemcpyDeviceToHost);
    cudaFree(d_matrix);
}

void softmax(std::vector<std::vector<float>>& matrix) {
    int rows = matrix.size();
    int cols = matrix[0].size();
    float* d_matrix;
    size_t size = rows * cols * sizeof(float);

    cudaMalloc(&d_matrix, size);
    cudaMemcpy(d_matrix, matrix[0].data(), size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);
    softmaxKernel<<<blocksPerGrid, threadsPerBlock>>>(d_matrix, rows, cols);

    cudaMemcpy(matrix[0].data(), d_matrix, size, cudaMemcpyDeviceToHost);
    cudaFree(d_matrix);
}

std::vector<std::vector<float>> scaled_dot_product_attention(const std::vector<std::vector<float>>& Q, const std::vector<std::vector<float>>& K, const std::vector<std::vector<float>>& V) {
    int d_k = Q[0].size();
    std::vector<std::vector<float>> K_T(K[0].size(), std::vector<float>(K.size()));
    transpose(K, K_T);

    std::vector<std::vector<float>> QK(Q.size(), std::vector<float>(K_T[0].size()));
    multiplyMatrices(Q, K_T, QK);

    for (auto& row : QK)
        for (auto& val : row)
            val /= sqrt(d_k);

    applyUpperTriangularMask(QK);
    softmax(QK);

    std::vector<std::vector<float>> output(QK.size(), std::vector<float>(V[0].size()));
    multiplyMatrices(QK, V, output);

    return output;
}