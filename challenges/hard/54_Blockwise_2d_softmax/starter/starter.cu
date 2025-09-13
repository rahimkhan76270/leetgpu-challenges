#include <cuda_runtime.h>
#include<vector>
#define BLOCKDIM 16

__global__ void BlockwiseOnlineSoftmaxKernel(const float* input,float* output,float* l,float *m,int rows,int cols)
{
    int tx=threadIdx.x;
    int ty=threadIdx.y;
    int bx=blockIdx.x;
    int by=blockIdx.y;
    int row=by*BLOCKDIM+ty;
    int col=bx*BLOCKDIM+tx;
    if (row<rows && col<cols)
    {
        
    }
    
}
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    float *m,*l;
    std::vector<float> vec_inf(rows,-INFINITY);
    cudaMalloc(&m,rows*sizeof(float));
    cudaMalloc(&l,rows*sizeof(float));
    cudaMemset(l,0,rows*sizeof(float));
    cudaMemcpy(m,vec_inf.data(),rows*sizeof(float),cudaMemcpyDefault);
    dim3 block(BLOCKDIM, BLOCKDIM);
    dim3 grid((cols + block.x - 1) / block.x, (rows+block.y-1)/block.y);
    BlockwiseOnlineSoftmaxKernel<<<grid, block>>>(input, output,l,m, rows, cols);
    cudaDeviceSynchronize();
    cudaFree(l);
    cudaFree(m);
}
