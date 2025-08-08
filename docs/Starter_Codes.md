# Starter Code Creation Process for LeetGPU Challenges

A starter code is a template file that provides the basic structure and function signatures for implementing GPU-accelerated algorithms in LeetGPU challenges. It gives users a runnable foundation while leaving the core algorithmic logic as their task.

## Major Components

- **Function Signatures:** Standardized `solve` function with consistent parameters across all frameworks
- **Framework-Specific Templates:** CUDA, Triton, Mojo, PyTorch, and TinyGrad implementations
- **Memory Management:** Proper device pointer handling and memory allocation patterns
- **Kernel Structure:** Basic kernel function templates with grid/block sizing
- **Error Handling:** Bounds checking and synchronization primitives


### Identify Framework Requirements

Each framework has specific requirements:

**CUDA:**
- Kernel functions with `__global__` qualifier(for easy problems)
- `extern "C"` solve function for framework integration
- Proper memory management and synchronization
- Grid and block size calculations

**Triton:**
- `@triton.jit` decorator for kernel compilation
- Pointer type conversions for data types
- Block size and grid calculations
- PyTorch restriction compliance

**Mojo:**
- `@export` decorator for framework integration
- Proper GPU imports and memory types
- Device context management
- Function parameter types

**PyTorch/TinyGrad:**
- Tensor-based function signatures
- GPU tensor parameters
- Simple, direct implementations

## Designing Function Signatures

### Step 1: Define Core Parameters

Based on the algorithm requirements, determine:

1. **Input parameters**: What data does the algorithm need?
2. **Output parameters**: Where should results be written?
3. **Size parameters**: What dimensions are involved?
4. **Configuration parameters**: Any algorithm-specific settings?

### Step 2: Choose Parameter Names

Use clear, descriptive names that follow conventions:

**Common Patterns:**
- Single input/output: `input`, `output`
- Multiple inputs: `A`, `B`, `C` or `Q`, `K`, `V`
- Dimensions: `N`, `M`, `K`, `rows`, `cols`
- Algorithm-specific: `kernel_size`, `stride`, `padding`

### Step 3: Determine Data Types

**Standard Types:**
- **CUDA**: `const float*` for inputs, `float*` for outputs, `int` for sizes
- **Triton**: `int` for pointers, `int` for sizes
- **Mojo**: `UnsafePointer[Float32]` for data, `Int32` for sizes
- **PyTorch/TinyGrad**: `Tensor` for data, `int` for sizes


Refer Starter_Files for difficulty wise starter code templates.