import torch
import time

def benchmark_matrix_operations():
    # Start measuring time
    start_time = time.time()

    # Create a 10,000 x 10,000 matrix with values equal to their index
    # The index for each value is calculated as the row index times the number of columns plus the column index
    nrows, ncols = 10000, 10000
    
    matrix = torch.arange(nrows * ncols, device="cuda").float().reshape((nrows, ncols))

    # Time after creation
    creation_time = time.time() - start_time
    print(f"Matrix creation time: {creation_time:.5f} seconds.")

    # Copy the matrix
    start_time = time.time()
    matrix_copy = matrix.clone()
    copy_time = time.time() - start_time
    print(f"Matrix copying time: {copy_time:.5f} seconds.")

    # Multiply the matrix by 2
    start_time = time.time()
    matrix_multiplied = matrix.matmul(matrix)
    multiplication_time = time.time() - start_time
    print(f"Matrix multiplication time: {multiplication_time:.5f} seconds.")

    return creation_time, copy_time, multiplication_time

benchmark_matrix_operations()