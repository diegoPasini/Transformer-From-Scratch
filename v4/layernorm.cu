#include <vector>
#include <cuda_runtime.h>

__global__ void layerNormKernel(const float* input, float* output, float* gamma, float* beta, int size, float epsilon) {
    extern __shared__ float shared_mem[];
    float* mean = shared_mem;
    float* variance = shared_mem + 1;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float val = input[idx];
        atomicAdd(mean, val);
        __syncthreads();

        if (threadIdx.x == 0) {
            *mean /= size;
        }
        __syncthreads();

        float diff = val - *mean;
        atomicAdd(variance, diff * diff);
        __syncthreads();

        if (threadIdx.x == 0) {
            *variance /= size;
        }
        __syncthreads();

        float norm_val = (val - *mean) / sqrtf(*variance + epsilon);
        output[idx] = norm_val * gamma[idx] + beta[idx];
    }
}

void layerNorm(const std::vector<float>& input, std::vector<float>& output, std::vector<float>& gamma, std::vector<float>& beta, float epsilon) {
    int size = input.size();
    float *d_input, *d_output, *d_gamma, *d_beta;
    size_t bytes = size * sizeof(float);

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMalloc(&d_gamma, bytes);
    cudaMalloc(&d_beta, bytes);

    cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, gamma.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, beta.data(), bytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    layerNormKernel<<<blocksPerGrid, threadsPerBlock, 2 * sizeof(float)>>>(d_input, d_output, d_gamma, d_beta, size, epsilon);

    cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_gamma);
    cudaFree(d_beta);
}
