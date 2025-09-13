#include <cuda_runtime.h>

__global__ void swiglu_kernel(const float* input, float* output, int halfN) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if (idx<halfN)
    {
        float val=input[idx];
        float silu=val/(1+expf(-val));
        output[idx]=silu*input[idx+halfN];
    }
    
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    swiglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    cudaDeviceSynchronize();
}