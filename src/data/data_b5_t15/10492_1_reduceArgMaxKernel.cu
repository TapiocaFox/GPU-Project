#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;

static __device__ float E = 2.718281828;



__global__ void reduceArgMaxKernel(float *src, float *dst, float *arg, int dim_size, int block_size)
{
int di = blockIdx.x * block_size + threadIdx.x;
int si = di * dim_size;
float now = src[si], max = now;
int maxi = 0;
for (int i = 1; i < dim_size; i++) {
now = src[si+i];
if (now > max) {
max = now;
maxi = i;
}
}
dst[di] = max;
arg[di] = maxi;
}

int main(int argc, char **argv) {
    cudaSetDevice(0);
    
    // Grid: (5, 5), Block: (32, 32)
    dim3 gridBlock(5, 5);
    dim3 threadBlock(32, 32);
    
    // Allocate device memory
    int dim_size = 10;
    int block_size = 32;
    int num_blocks = 5 * 5;
    float *d_src = NULL;
    float *d_dst = NULL;
    float *d_arg = NULL;
    cudaMalloc(&d_src, num_blocks * block_size * dim_size * sizeof(float));
    cudaMalloc(&d_dst, num_blocks * block_size * sizeof(float));
    cudaMalloc(&d_arg, num_blocks * block_size * sizeof(float));
    
    // Warmup
    cudaFree(0);
    reduceArgMaxKernel<<<gridBlock, threadBlock>>>(d_src, d_dst, d_arg, dim_size, block_size);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < 10; ++i) {
        reduceArgMaxKernel<<<gridBlock, threadBlock>>>(d_src, d_dst, d_arg, dim_size, block_size);
    }
    cudaDeviceSynchronize();
    
    // Measure execution time
    auto start = steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        reduceArgMaxKernel<<<gridBlock, threadBlock>>>(d_src, d_dst, d_arg, dim_size, block_size);
    }
    cudaDeviceSynchronize();
    auto end = steady_clock::now();
    
    auto usecs = duration_cast<duration<float, microseconds::period>>(end - start);
    float avg_time_us = usecs.count() / 1000.0f;
    float avg_time_ms = avg_time_us / 1000.0f;
    
    cout << "[" << avg_time_us << "," << avg_time_ms << "]" << endl;
    
    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_arg);
    
    return 0;
}
