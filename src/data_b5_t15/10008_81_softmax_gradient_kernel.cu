#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;

#define SOFTMAX_NUM_THREADS 32

__global__ void softmax_gradient_kernel( const int dim, const float* Y, const float* dY, float* dX) {
Y += blockIdx.x * dim;
dY += blockIdx.x * dim;
dX += blockIdx.x * dim;
const int idx = threadIdx.x;
__shared__ float reduction_buffer[SOFTMAX_NUM_THREADS];
float tmp;

// A two-level reduction to compute the inner products.
tmp = 0;
for (int i = idx; i < dim; i += blockDim.x) {
tmp += dY[i] * Y[i];
}
reduction_buffer[idx] = tmp;
__syncthreads();
if (idx == 0) {
tmp = reduction_buffer[0];
for (int i = 1; i < blockDim.x; ++i)
tmp += reduction_buffer[i];
reduction_buffer[0] = tmp;
}
__syncthreads();
// Compute gradient.
tmp = reduction_buffer[0];
for (int i = idx; i < dim; i += blockDim.x) {
dX[i] = Y[i] * (dY[i] - tmp);
}
}

int main(int argc, char **argv) {
    cudaSetDevice(0);
    
    // Grid: (5, 5), Block: (32, 32)
    dim3 gridBlock(5, 5);
    dim3 threadBlock(32, 32);
    
    // Allocate device memory
    int dim = 100;
    float *d_Y = NULL;
    float *d_dY = NULL;
    float *d_dX = NULL;
    cudaMalloc(&d_Y, 5 * 5 * dim * sizeof(float));
    cudaMalloc(&d_dY, 5 * 5 * dim * sizeof(float));
    cudaMalloc(&d_dX, 5 * 5 * dim * sizeof(float));
    
    // Warmup
    cudaFree(0);
    softmax_gradient_kernel<<<gridBlock, threadBlock>>>(dim, d_Y, d_dY, d_dX);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < 10; ++i) {
        softmax_gradient_kernel<<<gridBlock, threadBlock>>>(dim, d_Y, d_dY, d_dX);
    }
    cudaDeviceSynchronize();
    
    // Measure execution time
    auto start = steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        softmax_gradient_kernel<<<gridBlock, threadBlock>>>(dim, d_Y, d_dY, d_dX);
    }
    cudaDeviceSynchronize();
    auto end = steady_clock::now();
    
    auto usecs = duration_cast<duration<float, microseconds::period>>(end - start);
    float avg_time_us = usecs.count() / 1000.0f;
    float avg_time_ms = avg_time_us / 1000.0f;
    
    cout << "[" << avg_time_us << "," << avg_time_ms << "]" << endl;
    
    cudaFree(d_Y);
    cudaFree(d_dY);
    cudaFree(d_dX);
    
    return 0;
}
