#include <cuda_runtime.h>


__global__ void SiLUKernel(const float* input, float* output, int N){
    unsigned int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if (idx<N)
    {
        float val=input[idx];
        output[idx]=val/(1+expf(-val));
    }
    
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int threads=256;
    int blocks=(N+threads-1)/threads;
    SiLUKernel<<<blocks,threads>>>(input,output,N);
    cudaDeviceSynchronize();
}
