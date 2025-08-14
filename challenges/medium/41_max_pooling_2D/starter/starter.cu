#include <cuda_runtime.h>

__global__ void max_pooling_2d(const float* input, float* output,
                               int N, int C, int H, int W,
                               int kernel_size, int stride, int padding) {

}

// input_ptr, output_ptr are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input_ptr, float* output_ptr,
                      int N, int C, int H, int W,
                      int kernel_size, int stride, int padding) {
    // Calculate output dimensions
    int H_out = (H + 2 * padding - kernel_size) / stride + 1;
    int W_out = (W + 2 * padding - kernel_size) / stride + 1;
    
    // Define block and grid dimensions
    dim3 blockDim(16, 16, 1);
    dim3 gridDim((H_out + blockDim.x - 1) / blockDim.x,
                  C,
                  N);
    
    // Launch kernel
    max_pooling_2d<<<gridDim, blockDim>>>(input_ptr, output_ptr, N, C, H, W, 
                                          kernel_size, stride, padding);
    cudaDeviceSynchronize();
}
