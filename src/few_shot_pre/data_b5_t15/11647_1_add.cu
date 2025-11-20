#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;

__global__ void add(int* in, int offset, int n){

int gid = threadIdx.x + blockIdx.x * blockDim.x;
if(gid >= n) return ;

extern __shared__ int temp[];

temp[threadIdx.x] = in[gid];

__syncthreads(); //can only control threads in a block.
if(threadIdx.x >= offset){
in[threadIdx.x] += temp[threadIdx.x-offset];
} else if(gid >= offset){
in[threadIdx.x] += in[gid-offset];
}
in[gid] = temp[threadIdx.x];
}

int main(int argc, char **argv) {
    cudaSetDevice(0);
    
    // Grid: (5, 5), Block: (32, 32)
    dim3 gridBlock(5, 5);
    dim3 threadBlock(32, 32);
    
    // Allocate device memory
    int n = 5 * 32 * 5 * 32;
    int offset = 1;
    int *d_in = NULL;
    cudaMalloc(&d_in, n * sizeof(int));
    
    // Warmup
    cudaFree(0);
    add<<<gridBlock, threadBlock, 32 * 32 * sizeof(int)>>>(d_in, offset, n);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < 10; ++i) {
        add<<<gridBlock, threadBlock, 32 * 32 * sizeof(int)>>>(d_in, offset, n);
    }
    cudaDeviceSynchronize();
    
    // Measure execution time
    auto start = steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        add<<<gridBlock, threadBlock, 32 * 32 * sizeof(int)>>>(d_in, offset, n);
    }
    cudaDeviceSynchronize();
    auto end = steady_clock::now();
    
    auto usecs = duration_cast<duration<float, microseconds::period>>(end - start);
    float avg_time_us = usecs.count() / 1000.0f;
    float avg_time_ms = avg_time_us / 1000.0f;
    
    cout << "[" << avg_time_us << "," << avg_time_ms << "]" << endl;
    
    cudaFree(d_in);
    
    return 0;
}
