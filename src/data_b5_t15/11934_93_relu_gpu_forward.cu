#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;

__global__ void relu_gpu_forward(float *out, float *in, int64_t N) {
int tid = blockIdx.x * blockDim.x + threadIdx.x;
if (tid < N)
out[tid] = in[tid] > 0 ? in[tid] : 0;
}

int main(int argc, char **argv) {
    cudaSetDevice(0);
    
    // Grid: (5, 5), Block: (32, 32)
    dim3 gridBlock(5, 5);
    dim3 threadBlock(32, 32);
    
    // Allocate device memory
    int64_t N = 5 * 32 * 5 * 32;
    float *d_in = NULL;
    float *d_out = NULL;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    
    // Warmup
    cudaFree(0);
    relu_gpu_forward<<<gridBlock, threadBlock>>>(d_out, d_in, N);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < 10; ++i) {
        relu_gpu_forward<<<gridBlock, threadBlock>>>(d_out, d_in, N);
    }
    cudaDeviceSynchronize();
    
    // Measure execution time
    auto start = steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        relu_gpu_forward<<<gridBlock, threadBlock>>>(d_out, d_in, N);
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
