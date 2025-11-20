#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;

#define BLUE 0
#define GREEN 1
#define RED 2
#define maxLum 100.0f

__device__ float logarithmic_mapping(float k, float q, float val_pixel){
return (log10(1 + q * val_pixel))/(log10(1 + k * maxLum));
}
__global__ void tonemap_logarithmic(float* imageIn, float* imageOut, int width, int height, int channels, int depth, float q, float k){
int Row = blockDim.y * blockIdx.y + threadIdx.y;
int Col = blockDim.x * blockIdx.x + threadIdx.x;

if(Row < height && Col < width) {
imageOut[(Row*width+Col)*3+BLUE] = logarithmic_mapping(k, q, imageIn[(Row*width+Col)*3+BLUE]);
imageOut[(Row*width+Col)*3+GREEN] = logarithmic_mapping(k, q, imageIn[(Row*width+Col)*3+GREEN]);
imageOut[(Row*width+Col)*3+RED] = logarithmic_mapping(k, q, imageIn[(Row*width+Col)*3+RED]);
}
}

int main(int argc, char **argv) {
    cudaSetDevice(0);
    
    // Grid: (5, 5), Block: (32, 32)
    dim3 gridBlock(5, 5);
    dim3 threadBlock(32, 32);
    
    // Allocate device memory
    int width = 160;
    int height = 160;
    int channels = 3;
    int depth = 1;
    float q = 0.5f;
    float k = 1.0f;
    float *d_imageIn = NULL;
    float *d_imageOut = NULL;
    cudaMalloc(&d_imageIn, width * height * channels * sizeof(float));
    cudaMalloc(&d_imageOut, width * height * channels * sizeof(float));
    
    // Warmup
    cudaFree(0);
    tonemap_logarithmic<<<gridBlock, threadBlock>>>(d_imageIn, d_imageOut, width, height, channels, depth, q, k);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < 10; ++i) {
        tonemap_logarithmic<<<gridBlock, threadBlock>>>(d_imageIn, d_imageOut, width, height, channels, depth, q, k);
    }
    cudaDeviceSynchronize();
    
    // Measure execution time
    auto start = steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        tonemap_logarithmic<<<gridBlock, threadBlock>>>(d_imageIn, d_imageOut, width, height, channels, depth, q, k);
    }
    cudaDeviceSynchronize();
    auto end = steady_clock::now();
    
    auto usecs = duration_cast<duration<float, microseconds::period>>(end - start);
    float avg_time_us = usecs.count() / 1000.0f;
    float avg_time_ms = avg_time_us / 1000.0f;
    
    cout << "[" << avg_time_us << "," << avg_time_ms << "]" << endl;
    
    cudaFree(d_imageIn);
    cudaFree(d_imageOut);
    
    return 0;
}
