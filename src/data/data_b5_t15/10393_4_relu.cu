#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;

extern "C"
__global__ void relu(double* A,  double* ret, int rlen, int clen) {
int tid = blockIdx.x * blockDim.x + threadIdx.x;
int ix = tid / clen;
int iy = tid % clen;
if(ix < rlen && iy < clen) {
ret[tid] = max(0.0, A[tid]);
}
}

int main(int argc, char **argv) {
    cudaSetDevice(0);
    
    // Grid: (5, 5), Block: (32, 32)
    dim3 gridBlock(5, 5);
    dim3 threadBlock(32, 32);
    
    // Allocate device memory
    int rlen = 160;
    int clen = 160;
    double *d_A = NULL;
    double *d_ret = NULL;
    cudaMalloc(&d_A, rlen * clen * sizeof(double));
    cudaMalloc(&d_ret, rlen * clen * sizeof(double));
    
    // Warmup
    cudaFree(0);
    relu<<<gridBlock, threadBlock>>>(d_A, d_ret, rlen, clen);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < 10; ++i) {
        relu<<<gridBlock, threadBlock>>>(d_A, d_ret, rlen, clen);
    }
    cudaDeviceSynchronize();
    
    // Measure execution time
    auto start = steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        relu<<<gridBlock, threadBlock>>>(d_A, d_ret, rlen, clen);
    }
    cudaDeviceSynchronize();
    auto end = steady_clock::now();
    
    auto usecs = duration_cast<duration<float, microseconds::period>>(end - start);
    float avg_time_us = usecs.count() / 1000.0f;
    float avg_time_ms = avg_time_us / 1000.0f;
    
    cout << "[" << avg_time_us << "," << avg_time_ms << "]" << endl;
    
    cudaFree(d_A);
    cudaFree(d_ret);
    
    return 0;
}
