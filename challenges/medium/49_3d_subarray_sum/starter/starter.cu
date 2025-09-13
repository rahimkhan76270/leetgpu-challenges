#include <cuda_runtime.h>
#define BLOCKDIM 256
__global__ void reduce_subarray3d(const int* input, int* output, int N, int M, int K, int S_DEP, int E_DEP, int S_ROW, int E_ROW, int S_COL, int E_COL){
    int tid=threadIdx.x;
    int idx=blockIdx.x*blockDim.x+tid;
    int dep=idx/(M*K);
    int rem1=idx%(M*K);
    int row=rem1/K;
    int col=rem1%K;
    __shared__ int temp[BLOCKDIM];
    if(dep>=S_DEP && dep<=E_DEP && row>=S_ROW && row<=E_ROW && col>=S_COL && col<=E_COL)
    {
        temp[tid]=input[idx];
    }
    else
    {
        temp[tid]=0;
    }
    __syncthreads();

    for(int stride=BLOCKDIM/2;stride>0;stride/=2)
    {
        if(tid<stride)
        {
            temp[tid]+=temp[tid+stride];
        }
        __syncthreads();
    }
    if(tid==0) atomicAdd(output,temp[0]);
}
// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int M, int K, int S_DEP, int E_DEP, int S_ROW, int E_ROW, int S_COL, int E_COL) {
    int blocks=(N*M*K+BLOCKDIM-1)/BLOCKDIM;
    reduce_subarray3d<<<blocks,BLOCKDIM>>>(input,output,N,M,K,S_DEP,E_DEP,S_ROW,E_ROW,S_COL,E_COL);
    cudaDeviceSynchronize();
}