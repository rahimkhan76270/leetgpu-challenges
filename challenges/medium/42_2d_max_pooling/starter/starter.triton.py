import torch
import triton
import triton.language as tl

# input_ptr, output_ptr are tensors on the GPU
def solve(input_ptr, output_ptr, N, C, H, W, kernel_size, stride, padding):
   pass