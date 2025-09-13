#include <cuda_runtime.h>

__device__ float distance(float x1,float y1,float x2,float y2)
{
    return sqrtf((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2));
}

__global__ void KmeansKernel(const float* data_x, const float* data_y, int* labels,
           float* initial_centroid_x, float* initial_centroid_y,
           float* final_centroid_x, float* final_centroid_y,
           int sample_size, int k, int max_iterations)
{
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if (idx>=sample_size) return;

    
    
}

// data_x, data_y, labels, initial_centroid_x, initial_centroid_y,
// final_centroid_x, final_centroid_y are device pointers 
extern "C" void solve(const float* data_x, const float* data_y, int* labels,
           float* initial_centroid_x, float* initial_centroid_y,
           float* final_centroid_x, float* final_centroid_y,
           int sample_size, int k, int max_iterations) {

}
