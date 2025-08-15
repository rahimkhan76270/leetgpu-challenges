#include <cuda_runtime.h>

// input_ptr, output_ptr are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input_ptr, float* output_ptr,
                      int N, int C, int H, int W,
                      int kernel_size, int stride, int padding) {
  
}