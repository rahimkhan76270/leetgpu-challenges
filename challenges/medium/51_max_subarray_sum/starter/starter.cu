#include <cuda_runtime.h>
#include <climits>

#define BLOCKDIM 256

__global__ void window_sum_kernel(const int *input, int *output, int n, int window_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n - window_size + 1) {
        int val = 0;
        for (int i = 0; i < window_size; i++) {
            val += input[idx + i];
        }
        output[idx] = val;
    }
}

__global__ void reduce_max(const int *input, int *output, int n) {
    __shared__ int temp[BLOCKDIM];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    temp[tid] = (idx < n) ? input[idx] : INT_MIN;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride && tid + stride < BLOCKDIM) {
            temp[tid] = max(temp[tid], temp[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMax(output, temp[0]);
    }
}

extern "C" void solve(const int* input, int* output, int N, int window_size) {
    int window_count = N - window_size + 1;
    int *window_sum;
    cudaMalloc(&window_sum, window_count * sizeof(int));

    // Launch window sum kernel
    int blocks = (window_count + BLOCKDIM - 1) / BLOCKDIM;
    window_sum_kernel<<<blocks, BLOCKDIM>>>(input, window_sum, N, window_size);
    cudaDeviceSynchronize();

    // Initialize output to INT_MIN
    cudaMemset(output, 0, sizeof(int)); // Set to 0 first
    int host_min = INT_MIN;
    cudaMemcpy(output, &host_min, sizeof(int), cudaMemcpyHostToDevice);

    // Launch reduction kernel
    reduce_max<<<blocks, BLOCKDIM>>>(window_sum, output, window_count);
    cudaDeviceSynchronize();

    cudaFree(window_sum);
}
