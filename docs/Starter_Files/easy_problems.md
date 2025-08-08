# Easy Problems Starter File Creation Guide

Easy problems typically involve straightforward GPU operations with minimal complexity.

Easy problems typically feature:
- **Simple algorithms**: Vector operations, element-wise transformations
- **Basic memory patterns**: Linear access patterns, minimal synchronization
- **Straightforward parameters**: Usually 2-4 parameters (inputs, outputs, sizes)
- **Single kernel**: One main kernel function per solution
- **No complex data structures**: Arrays, simple matrices
- **Element-wise operations**: Each thread processes one element independently


### CUDA Starter Template

```cuda
#include <cuda_runtime.h>

__global__ void kernel_name(const float* input, float* output, int N) {
    // TODO: Implement kernel logic
    // Each thread processes one element
    // Use threadIdx.x + blockIdx.x * blockDim.x to get global index
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    kernel_name<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
```

### Triton Starter Template

```python
# The use of PyTorch in Triton programs is not allowed for the purposes of fair benchmarking.
import triton
import triton.language as tl

@triton.jit
def kernel_name(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    input_ptr = input_ptr.to(tl.pointer_type(tl.float32))
    output_ptr = output_ptr.to(tl.pointer_type(tl.float32)) 
    
    # TODO: Implement kernel logic
    # Use tl.program_id(0) to get block index
    # Use tl.program_id(1) to get thread ndex within block

# input_ptr, output_ptr are raw device pointers
def solve(input_ptr: int, output_ptr: int, N: int):    
    BLOCK_SIZE = 1024
    # define grid
    kernel_name[grid](input_ptr, output_ptr, N, BLOCK_SIZE)
```

### Mojo Starter Template

```mojo
from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

fn kernel_name(input: UnsafePointer[Float32], output: UnsafePointer[Float32], N: Int32):
    # TODO: Implement kernel logic
    # Use thread_idx() to get thread index
    # Use block_idx() to get block index
    pass

@export
def solve(input: UnsafePointer[Float32], output: UnsafePointer[Float32], N: Int32):
    # define block, thread size
    var ctx = DeviceContext()

    # Launch the kernel using enqueue_function
    ctx.enqueue_function[kernel_name](
        input_ptr, output_ptr, N,
        grid_dim  = num_blocks,     # Number of blocks in 1D grid
        block_dim = BLOCK_SIZE      # Number of threads per block
    )

    ctx.synchronize()
    # TODO: Implement solve function
    pass
```

### PyTorch Starter Template

```python
import torch

def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    # TODO: Implement solution using PyTorch operations
    pass
```

### TinyGrad Starter Template

```python
import tinygrad

def solve(input: tinygrad.Tensor, output: tinygrad.Tensor, N: int):
    # TODO: Implement solution using TinyGrad operations
    pass
```

### Common Mistakes

1. **Missing bounds checking**: Always check `idx < N`
2. **Incorrect grid/block sizing**: Ensure all elements are covered
3. **Wrong memory access patterns**: Ensure coalesced access
4. **Missing synchronization**: Call `cudaDeviceSynchronize()`
5. **Incorrect pointer types**: Use proper type conversions in Triton

