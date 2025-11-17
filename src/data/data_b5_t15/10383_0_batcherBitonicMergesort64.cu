#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;

// http://en.wikipedia.org/wiki/Bitonic_sort

__global__ void batcherBitonicMergesort64(float * d_out, const float * d_in)
{
// you are guaranteed this is called with <<<1, 64, 64*4>>>
extern __shared__ float sdata[];
int tid  = threadIdx.x;
sdata[tid] = d_in[tid];
__syncthreads();

for (int stage = 0; stage <= 5; stage++)
{
for (int substage = stage; substage >= 0; substage--)
{
// TODO
}
}

d_out[tid] = sdata[tid];
}

int main(int argc, char **argv) {
    cudaSetDevice(0);
    
    // Grid: (5, 5), Block: (32, 32)
    dim3 gridBlock(5, 5);
    dim3 threadBlock(32, 32);
    
    // Allocate device memory
    int size = 64;
    float *d_in = NULL;
    float *d_out = NULL;
    cudaMalloc(&d_in, size * sizeof(float));
    cudaMalloc(&d_out, size * sizeof(float));
    
    // Warmup
    cudaFree(0);
    batcherBitonicMergesort64<<<gridBlock, threadBlock, size * sizeof(float)>>>(d_out, d_in);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < 10; ++i) {
        batcherBitonicMergesort64<<<gridBlock, threadBlock, size * sizeof(float)>>>(d_out, d_in);
    }
    cudaDeviceSynchronize();
    
    // Measure execution time
    auto start = steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        batcherBitonicMergesort64<<<gridBlock, threadBlock, size * sizeof(float)>>>(d_out, d_in);
    }
    cudaDeviceSynchronize();
    auto end = steady_clock::now();
    
    auto usecs = duration_cast<duration<float, microseconds::period>>(end - start);
    float avg_time_us = usecs.count() / 1000.0f;
    float avg_time_ms = avg_time_us / 1000.0f;
    
    cout << "[" << avg_time_us << "," << avg_time_ms << "]" << endl;
    
    cudaFree(d_in);
    cudaFree(d_out);
    
    return 0;
}
