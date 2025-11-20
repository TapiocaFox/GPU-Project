#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;

__global__ void reduce(int *g_idata, int *g_odata) {

__shared__ int sdata[256];

// each thread loads one element from global to shared mem
// note use of 1D thread indices (only) in this kernel
int i = blockIdx.x*blockDim.x + threadIdx.x;

sdata[threadIdx.x] = g_idata[i];

__syncthreads();
// do reduction in shared mem
for (int s = 1; s < blockDim.x; s *= 2)
{
int index = 2 * s * threadIdx.x;;

if (index < blockDim.x)
{
sdata[index] += sdata[index + s];
}
__syncthreads();
}

// write result for this block to global mem
if (threadIdx.x == 0)
atomicAdd(g_odata, sdata[0]);
}

int main(int argc, char **argv) {
    cudaSetDevice(0);
    
    // Grid: (5, 5), Block: (32, 32)
    dim3 gridBlock(5, 5);
    dim3 threadBlock(32, 32);
    
    // Allocate device memory
    int size = 5 * 32 * 5 * 32;
    int *d_g_idata = NULL;
    int *d_g_odata = NULL;
    cudaMalloc(&d_g_idata, size * sizeof(int));
    cudaMalloc(&d_g_odata, sizeof(int));
    
    // Warmup
    cudaFree(0);
    reduce<<<gridBlock, threadBlock>>>(d_g_idata, d_g_odata);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < 10; ++i) {
        reduce<<<gridBlock, threadBlock>>>(d_g_idata, d_g_odata);
    }
    cudaDeviceSynchronize();
    
    // Measure execution time
    auto start = steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        reduce<<<gridBlock, threadBlock>>>(d_g_idata, d_g_odata);
    }
    cudaDeviceSynchronize();
    auto end = steady_clock::now();
    
    auto usecs = duration_cast<duration<float, microseconds::period>>(end - start);
    float avg_time_us = usecs.count() / 1000.0f;
    float avg_time_ms = avg_time_us / 1000.0f;
    
    cout << "[" << avg_time_us << "," << avg_time_ms << "]" << endl;
    
    cudaFree(d_g_idata);
    cudaFree(d_g_odata);
    
    return 0;
}
