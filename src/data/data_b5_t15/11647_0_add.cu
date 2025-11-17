#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;

#define N 10000000 //input data size: 10,000,000
#define BLOCKSIZE 1024

/* prefix sum */

using namespace std;

__global__ void add(double* in, double* out, int offset, int n){

int gid = threadIdx.x + blockIdx.x * blockDim.x;
if(gid >= n) return ;

out[gid] = in[gid];
if(gid >= offset)
out[gid] += in[gid-offset];
}

int main(int argc, char **argv) {
    cudaSetDevice(0);
    
    // Grid: (5, 5), Block: (32, 32)
    dim3 gridBlock(5, 5);
    dim3 threadBlock(32, 32);
    
    // Allocate device memory
    int n = 5 * 32 * 5 * 32;
    int offset = 1;
    double *d_in = NULL;
    double *d_out = NULL;
    cudaMalloc(&d_in, n * sizeof(double));
    cudaMalloc(&d_out, n * sizeof(double));
    
    // Warmup
    cudaFree(0);
    add<<<gridBlock, threadBlock>>>(d_in, d_out, offset, n);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < 10; ++i) {
        add<<<gridBlock, threadBlock>>>(d_in, d_out, offset, n);
    }
    cudaDeviceSynchronize();
    
    // Measure execution time
    auto start = steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        add<<<gridBlock, threadBlock>>>(d_in, d_out, offset, n);
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
