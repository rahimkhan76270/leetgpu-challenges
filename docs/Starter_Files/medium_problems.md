# Medium Problems Starter File Creation Guide

Medium problems introduce more complex algorithms and memory access patterns while maintaining reasonable implementation complexity.

Medium problems typically feature:
- **Multi-step algorithms**: Operations requiring multiple kernel launches or complex logic
- **Advanced memory patterns**: Shared memory usage, reduction operations
- **Multiple parameters**: 3-6 parameters including algorithm-specific settings
- **Complex indexing**: 2D/3D indexing, strided access patterns
- **Inter-thread communication**: Some cooperation between threads within blocks
- **Algorithm-specific optimizations**: Custom memory layouts, specialized kernels

## Starter File Structure

### CUDA Starter Template

```cuda
#include <cuda_runtime.h>

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    // TODO: Implement solve function
    // May require multiple kernel launches
    // Consider memory allocation for intermediate results
}
```

### Triton Starter Template

```python
# The use of PyTorch in Triton programs is not allowed for the purposes of fair benchmarking.
import triton
import triton.language as tl

def solve(input_ptr: int, output_ptr: int, N: int):
    # TODO: Implement solve function
    # May require multiple kernel launches
    # Consider intermediate memory allocation
    pass
```

### Mojo Starter Template

```mojo
from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

@export
def solve(input: UnsafePointer[Float32], output: UnsafePointer[Float32], N: Int32):
    # TODO: Implement solve function
    # May require multiple kernel launches or complex logic
    pass
```

### PyTorch Starter Template

```python
import torch

def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    # TODO: Implement solution using PyTorch operations
    # May require multiple tensor operations
    pass
```

### TinyGrad Starter Template

```python
import tinygrad

def solve(input: tinygrad.Tensor, output: tinygrad.Tensor, N: int):
    # TODO: Implement solution using TinyGrad operations
    # May require multiple tensor operations
    pass
```


## Common Mistakes to Avoid

1. **Missing synchronization**: Use `__syncthreads()` when needed
2. **Incorrect shared memory usage**: Avoid bank conflicts
3. **Memory leaks**: Free allocated memory
4. **Race conditions**: Ensure proper thread coordination
5. **Numerical instability**: Use proper techniques (e.g., softmax with max subtraction)

