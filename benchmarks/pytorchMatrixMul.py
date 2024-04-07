import torch
import time

# Without CUDA
start_time = torch.cuda.Event(enable_timing=True)
end_time = torch.cuda.Event(enable_timing=True)

start_cpu = time.time()
matrix_cpu = torch.randn(10_000, 10_000)
copy_matrix_cpu = matrix_cpu.clone()
result_cpu = torch.matmul(matrix_cpu, copy_matrix_cpu)
end_cpu = time.time()
cpu_time = end_cpu - start_cpu

# With CUDA
if torch.cuda.is_available():
    start_time.record()
    matrix_cuda = torch.randn(10_000, 10_000, device='cuda')
    copy_matrix_cuda = matrix_cuda.clone()
    result_cuda = matrix_cuda @ copy_matrix_cuda
    end_time.record()
    # Waits for everything to finish running
    torch.cuda.synchronize()
    cuda_time = start_time.elapsed_time(end_time) / 1000  # Convert milliseconds to seconds
else:
    cuda_time = None

print(cpu_time, cuda_time)