# Transformer in C++ and CUDA

# Compilation for testing
run `nvcc -o <exe name> <file1.cu> <file2.cu> <testFile.cpp>`
run `./<exe name>`

# Transformer Block Dimensions
- d_model: embedding dimension
- d_head: head embedding q/k/v dimension
- seq: input sequence length

1. Input into Transformer Block (residual stream):
- (batch, seq, d_model)

2. LayerNorm

3. LinearLayer for each: query, key, value, output
- W_q, W_k, W_v, W_o: (d_model, d_head) ?
- b_q, b_k, b_v, b_o: (d_head) ? do these need batch dim
- query, key, and value vectors: (batch, seq, head number, d_head)

4. Dot product Q and K along d_head
- result is (batch, head number, seq, seq)

5. Scale and Mask

6. Softmax

7. Weighted average of value vectors
- result: (batch, seq, head_index, d_model)

8. Multiply by long W_o and add b_o
- result: (batch, seq, head_idx, d_model)

9. Sum over attention heads
- (batch, seq, d_model)

10. Add 9 back to the residual stream (the og input)
- (batch, seq, d_model)
- this is the residual connection?

11. LayerNorm

12. MLP layer
- W_in, b_in: (d_model, 4 x d_model)
- W_out, b_out: (4 x d_model, d_model)
- Middle layer: (batch, seq, 4 x d_model)

13. Add 12 back to residual stream
- final output: (batch, seq, d_model)
