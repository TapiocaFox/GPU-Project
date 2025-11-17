#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <iostream>
#include <vector_types.h>
using namespace std;
using namespace std::chrono;

__device__ __forceinline__ float imag(const float2& val)
{
return val.y;
}
__global__ void NormalizeKernel(const float *normalization_factor, int w, int h, int s, float *image)
{
int i = threadIdx.y + blockDim.y * blockIdx.y;
int j = threadIdx.x + blockDim.x * blockIdx.x;

if (i >= h || j >= w) return;

const int pos = i * s + j;

float scale = normalization_factor[pos];

float invScale = (scale == 0.0f) ? 1.0f : (1.0f / scale);

image[pos] *= invScale;
}

int main(int argc, char **argv) {
    cudaSetDevice(0);
    
    // Grid: (5, 5), Block: (32, 32)
    dim3 gridBlock(5, 5);
    dim3 threadBlock(32, 32);
    
    // Allocate device memory
    int w = 160;
    int h = 160;
    int s = 160;
    float *d_normalization_factor = NULL;
    float *d_image = NULL;
    cudaMalloc(&d_normalization_factor, h * s * sizeof(float));
    cudaMalloc(&d_image, h * s * sizeof(float));
    
    // Warmup
    cudaFree(0);
    NormalizeKernel<<<gridBlock, threadBlock>>>(d_normalization_factor, w, h, s, d_image);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < 10; ++i) {
        NormalizeKernel<<<gridBlock, threadBlock>>>(d_normalization_factor, w, h, s, d_image);
    }
    cudaDeviceSynchronize();
    
    // Measure execution time
    auto start = steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        NormalizeKernel<<<gridBlock, threadBlock>>>(d_normalization_factor, w, h, s, d_image);
    }
    cudaDeviceSynchronize();
    auto end = steady_clock::now();
    
    auto usecs = duration_cast<duration<float, microseconds::period>>(end - start);
    float avg_time_us = usecs.count() / 1000.0f;
    float avg_time_ms = avg_time_us / 1000.0f;
    
    cout << "[" << avg_time_us << "," << avg_time_ms << "]" << endl;
    
    cudaFree(d_normalization_factor);
    cudaFree(d_image);
    
    return 0;
}
