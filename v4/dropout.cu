#include <curand_kernel.h>

__global__ void dropout_kernel(float* input, float* output, float dropout_prob, int size, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        float rand_val = curand_uniform(&state);
        output[idx] = (rand_val > dropout_prob) ? input[idx] / (1.0f - dropout_prob) : 0.0f;
    }
}

void dropout(float* input, float* output, float dropout_prob, int size) {
    float *d_input, *d_output;
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));

    cudaMemcpy(d_input, input, size * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    unsigned long long seed = 1234ULL;
    dropout_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, dropout_prob, size, seed);

    cudaMemcpy(output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}
