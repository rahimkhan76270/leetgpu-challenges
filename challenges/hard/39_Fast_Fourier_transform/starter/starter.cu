#include <cuda_runtime.h>
#include <math_constants.h>  
#include <cstdio>
#include <cmath>

#define PI CUDART_PI_F

static inline int ilog2_int(int n) {
    int l = 0;
    while ((1 << l) < n) ++l;
    return l;
}

static inline bool is_power_of_two(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

__device__ __forceinline__ void load_complex(const float* a, int idx, float &xr, float &xi) {
    xr = a[2 * idx];
    xi = a[2 * idx + 1];
}

__device__ __forceinline__ void store_complex(float* a, int idx, float xr, float xi) {
    a[2 * idx]     = xr;
    a[2 * idx + 1] = xi;
}

__global__ void bitreverse_reorder_kernel(const float* __restrict__ signal,
                                          float* __restrict__ spectrum,
                                          int N, int log2N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    unsigned int ri = __brev((unsigned int)i) >> (32 - log2N);

    float xr, xi;
    load_complex(signal, i, xr, xi);
    store_complex(spectrum, ri, xr, xi);
}


__global__ void fft_stage_kernel(float* __restrict__ data, int N, int m, int half)
{
    int bfly = blockIdx.x * blockDim.x + threadIdx.x;
    int total_bfly = N >> 1;
    if (bfly >= total_bfly) return;

    
    int block_id = bfly / half;      // which m-sized block
    int j        = bfly % half;      // butterfly index within the block

    int base = block_id * m;
    int i1   = base + j;
    int i2   = i1 + half;

    float u_r, u_i, v_r, v_i;
    load_complex(data, i1, u_r, u_i);
    load_complex(data, i2, v_r, v_i);

    float angle = -2.0f * PI * (float)j / (float)m;
    float c = cosf(angle);
    float s = sinf(angle);

    float t_r = c * v_r - s * v_i;
    float t_i = c * v_i + s * v_r;

    float out1_r = u_r + t_r;
    float out1_i = u_i + t_i;
    float out2_r = u_r - t_r;
    float out2_i = u_i - t_i;

    store_complex(data, i1, out1_r, out1_i);
    store_complex(data, i2, out2_r, out2_i);
}

__global__ void dft_double_kernel(const float* __restrict__ signal,
                                  float* __restrict__ spectrum,
                                  int N)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= N) return;

    double sum_r = 0.0, sum_i = 0.0;
    for (int n = 0; n < N; ++n)
    {
        double xr = (double)signal[2 * n];
        double xi = (double)signal[2 * n + 1];
        double angle = -2.0 * M_PI * (double)k * (double)n / (double)N;
        double c = cos(angle);
        double s = sin(angle);
        // (xr + i*xi) * (c + i*s)
        sum_r += xr * c - xi * s;
        sum_i += xr * s + xi * c;
    }
    spectrum[2 * k]     = (float)sum_r;
    spectrum[2 * k + 1] = (float)sum_i;
}

extern "C" void solve(const float* signal, float* spectrum, int N)
{
    if (N <= 1) {
        if (N == 1) {
            // Copy single sample
            cudaMemcpy(spectrum, signal, sizeof(float) * 2, cudaMemcpyDeviceToDevice);
        }
        cudaDeviceSynchronize();
        return;
    }

    const int threads = 256;

    if (!is_power_of_two(N)) {
        int blocks = (N + threads - 1) / threads;
        dft_double_kernel<<<blocks, threads>>>(signal, spectrum, N);
        cudaDeviceSynchronize();
        return;
    }

    
    int log2N = ilog2_int(N);
    {
        int blocks = (N + threads - 1) / threads;
        bitreverse_reorder_kernel<<<blocks, threads>>>(signal, spectrum, N, log2N);
    }

    for (int s = 1; s <= log2N; ++s) {
        int m    = 1 << s;
        int half = m >> 1;

        int total_bfly = N >> 1; 
        int blocks = (total_bfly + threads - 1) / threads;

        fft_stage_kernel<<<blocks, threads>>>(spectrum, N, m, half);
    }

    cudaDeviceSynchronize();
}
