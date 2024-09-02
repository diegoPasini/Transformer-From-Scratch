#include <vector>
#include <cmath>
#include <cuda_runtime.h>

__global__ void flash_attention_kernel(const float* Q, const float* K, const float* V, float* output, int batch_size, int num_heads, int seq_len, int d_k) {
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int seq_idx = threadIdx.x;

    extern __shared__ float shared_mem[];
    float* shared_Q = shared_mem;
    float* shared_K = shared_mem + d_k;
    float* shared_V = shared_mem + 2 * d_k;
    float* shared_output = shared_mem + 3 * d_k;

    if (seq_idx < d_k) {
        shared_Q[seq_idx] = Q[batch_idx * num_heads * seq_len * d_k + head_idx * seq_len * d_k + seq_idx];
        shared_K[seq_idx] = K[batch_idx * num_heads * seq_len * d_k + head_idx * seq_len * d_k + seq_idx];
        shared_V[seq_idx] = V[batch_idx * num_heads * seq_len * d_k + head_idx * seq_len * d_k + seq_idx];
    }
    __syncthreads();

    float sum = 0.0f;
    for (int i = 0; i < d_k; ++i) {
        sum += shared_Q[i] * shared_K[i];
    }
    sum /= sqrtf(d_k);
    sum = expf(sum);

    float sum_exp = 0.0f;
    for (int i = 0; i < d_k; ++i) {
        sum_exp += expf(shared_Q[i] * shared_K[i] / sqrtf(d_k));
    }

    for (int i = 0; i < d_k; ++i) {
        shared_output[i] = sum * shared_V[i] / sum_exp;
    } 
    __syncthreads();

    if (seq_idx < d_k) {
        output[batch_idx * num_heads * seq_len * d_k + head_idx * seq_len * d_k + seq_idx] = shared_output[seq_idx];
    }
}

void flash_attention(const std::vector<std::vector<std::vector<float>>>& Q, const std::vector<std::vector<std::vector<float>>>& K, const std::vector<std::vector<std::vector<float>>>& V, std::vector<std::vector<std::vector<float>>>& output) {
    int batch_size = Q.size();
    int num_heads = Q[0].size();
    int seq_len = Q[0][0].size();
    int d_k = Q[0][0][0].size();

    float *d_Q, *d_K, *d_V, *d_output;
    cudaMalloc(&d_Q, batch_size * num_heads * seq_len * d_k * sizeof(float));
    cudaMalloc(&d_K, batch_size * num_heads * seq_len * d_k * sizeof(float));
    cudaMalloc(&d_V, batch_size * num_heads * seq_len * d_k * sizeof(float));
    cudaMalloc(&d_output, batch_size * num_heads * seq_len * d_k * sizeof(float));

    std::vector<float> h_Q(batch_size * num_heads * seq_len * d_k);
    std::vector<float> h_K(batch_size * num_heads * seq_len * d_k);
    std::vector<float> h_V(batch_size * num_heads * seq_len * d_k);
    std::vector<float> h_output(batch_size * num_heads * seq_len * d_k);

    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < num_heads; ++h) {
            for (int s = 0; s < seq_len; ++s) {
                for (int d = 0; d < d_k; ++d) {
                    h_Q[b * num_heads * seq_len * d_k + h * seq_len * d_k + s * d_k + d] = Q[b][h][s][d];
                    h_K[b * num_heads * seq_len * d_k + h * seq_len * d_k + s * d_k + d] = K[b][h][s][d];
                    h_V[b * num_heads * seq_len * d_k + h * seq_len * d_k + s * d_k + d] = V[b][h][s][d];
                }
            }
        }
    }

    cudaMemcpy(d_Q, h_Q.data(), batch_size * num_heads * seq_len * d_k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.data(), batch_size * num_heads * seq_len * d_k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), batch_size * num_heads * seq_len * d_k * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blocks(batch_size, num_heads);
    dim3 threads(seq_len);
    size_t shared_mem_size = 4 * d_k * sizeof(float);
    flash_attention_kernel<<<blocks, threads, shared_mem_size>>>(d_Q, d_K, d_V, d_output, batch_size, num_heads, seq_len, d_k);

    cudaMemcpy(h_output.data(), d_output, batch_size * num_heads * seq_len * d_k * sizeof(float), cudaMemcpyDeviceToHost);

    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < num_heads; ++h) {
            for (int s = 0; s < seq_len; ++s) {
                for (int d = 0; d < d_k; ++d) {
                    output[b][h][s][d] = h_output[b * num_heads * seq_len * d_k + h * seq_len * d_k + s * d_k + d];
                }
            }
        }
    }

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_output);
}
