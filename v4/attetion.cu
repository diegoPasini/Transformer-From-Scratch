#include <vector>
#include <cmath>
#include <cuda_runtime.h>

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float value = 0.0f;
        for (int e = 0; e < K; ++e) {
            value += A[row * K + e] * B[e * N + col];
        }
        C[row * N + col] = value;
    }
}

void matmul(const std::vector<std::vector<float>>& A, const std::vector<std::vector<float>>& B, std::vector<std::vector<float>>& C) {
    int M = A.size();
    int K = A[0].size();
    int N = B[0].size();

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N);

    for (int i = 0; i < M; ++i)
        for (int j = 0; j < K; ++j)
            h_A[i * K + j] = A[i][j];

    for (int i = 0; i < K; ++i)
        for (int j = 0; j < N; ++j)
            h_B[i * N + j] = B[i][j];

    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);

    cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            C[i][j] = h_C[i * N + j];

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void softmax(std::vector<std::vector<float>>& mat) {
    for (auto& row : mat) {
        float max_val = *max_element(row.begin(), row.end());
        float sum_exp = 0.0f;
        for (auto& val : row) {
            val = exp(val - max_val);
            sum_exp += val;
        }
        for (auto& val : row) {
            val /= sum_exp;
        }
    }
}

std::vector<std::vector<float>> scaled_dot_product_attention(const std::vector<std::vector<float>>& Q, const std::vector<std::vector<float>>& K, const std::vector<std::vector<float>>& V) {
    int d_k = Q[0].size();
    std::vector<std::vector<float>> K_T(K[0].size(), std::vector<float>(K.size()));
    for (size_t i = 0; i < K.size(); ++i)
        for (size_t j = 0; j < K[0].size(); ++j)
            K_T[j][i] = K[i][j];

    std::vector<std::vector<float>> QK(Q.size(), std::vector<float>(K_T[0].size()));
    matmul(Q, K_T, QK);

    for (auto& row : QK)
        for (auto& val : row)
            val /= sqrt(d_k);

    softmax(QK);

    std::vector<std::vector<float>> output(QK.size(), std::vector<float>(V[0].size()));
    matmul(QK, V, output);

    return output;
}
