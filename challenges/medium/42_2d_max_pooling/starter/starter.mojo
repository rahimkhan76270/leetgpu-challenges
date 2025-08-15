from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

# input_ptr, output_ptr are device pointers (i.e. pointers to memory on the GPU)
@export                         
def solve(input_ptr: UnsafePointer[Float32], output_ptr: UnsafePointer[Float32],
          N: Int32, C: Int32, H: Int32, W: Int32,
          kernel_size: Int32, stride: Int32, padding: Int32):
    pass
