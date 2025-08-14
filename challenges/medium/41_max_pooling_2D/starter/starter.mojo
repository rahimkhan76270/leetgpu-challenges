from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

fn max_pooling_2d_kernel(input: UnsafePointer[Float32], output: UnsafePointer[Float32],
                         N: Int32, C: Int32, H: Int32, W: Int32,
                         kernel_size: Int32, stride: Int32, padding: Int32):
    pass

# input_ptr, output_ptr are device pointers (i.e. pointers to memory on the GPU)
@export                         
def solve(input_ptr: UnsafePointer[Float32], output_ptr: UnsafePointer[Float32],
          N: Int32, C: Int32, H: Int32, W: Int32,
          kernel_size: Int32, stride: Int32, padding: Int32):
    # Calculate output dimensions
    let H_out = (H + 2 * padding - kernel_size) // stride + 1
    let W_out = (W + 2 * padding - kernel_size) // stride + 1
    
    var ctx = DeviceContext()
    
    # Launch kernel with appropriate grid and block dimensions
    ctx.enqueue_function[max_pooling_2d_kernel](
        input_ptr, output_ptr, N, C, H, W, kernel_size, stride, padding,
        grid_dim = (H_out, C, N),
        block_dim = (16, 16, 1)
    )
    
    ctx.synchronize()
