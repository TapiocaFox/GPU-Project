#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;

#define BLOCK_SIZE  16
#define HEADER_SIZE 122
#define BLOCK_SIZE_SH 18

typedef unsigned char BYTE;

__global__ void gpu_grayscale(int width, int height, float *image, float *image_out)
{
const int h = blockIdx.y*blockDim.y + threadIdx.y;
const int w = blockIdx.x*blockDim.x + threadIdx.x;

int offset_out = h * width;
int offset = offset_out * 3;

if(h < height && w < width)
{
float *pixel = &image[offset + w * 3];
image_out[offset_out + w] = pixel[0] * 0.0722f + // B
pixel[1] * 0.7152f + // G
pixel[2] * 0.2126f;  // R
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
    float *d_image = NULL;
    float *d_image_out = NULL;
    cudaMalloc(&d_image, width * height * 3 * sizeof(float));
    cudaMalloc(&d_image_out, width * height * sizeof(float));
    
    // Warmup
    cudaFree(0);
    gpu_grayscale<<<gridBlock, threadBlock>>>(width, height, d_image, d_image_out);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < 10; ++i) {
        gpu_grayscale<<<gridBlock, threadBlock>>>(width, height, d_image, d_image_out);
    }
    cudaDeviceSynchronize();
    
    // Measure execution time
    auto start = steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        gpu_grayscale<<<gridBlock, threadBlock>>>(width, height, d_image, d_image_out);
    }
    cudaDeviceSynchronize();
    auto end = steady_clock::now();
    
    auto usecs = duration_cast<duration<float, microseconds::period>>(end - start);
    float avg_time_us = usecs.count() / 1000.0f;
    float avg_time_ms = avg_time_us / 1000.0f;
    
    cout << "[" << avg_time_us << "," << avg_time_ms << "]" << endl;
    
    cudaFree(d_image);
    cudaFree(d_image_out);
    
    return 0;
}
