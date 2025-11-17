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

__device__ float gpu_applyFilter(float *image, int stride, float *matrix, int filter_dim)
{
float pixel = 0.0f;

for (int h = 0; h < filter_dim; h++)
{
int offset        = h * stride;
int offset_kernel = h * filter_dim;

for (int w = 0; w < filter_dim; w++)
{
pixel += image[offset + w] * matrix[offset_kernel + w];
}
}

return pixel;
}
__global__ void gpu_gaussian(int width, int height, float *image, float *image_out)
{
float gaussian[9] = { 1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f,
2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f,
1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f };

int index_x = blockIdx.x * blockDim.x + threadIdx.x;
int index_y = blockIdx.y * blockDim.y + threadIdx.y;

__shared__ float sh_block[BLOCK_SIZE_SH * BLOCK_SIZE_SH];


if (index_x < (width - 2) && index_y < (height - 2))
{
int offset_t = index_y * width + index_x;
int offset   = (index_y + 1) * width + (index_x + 1);
int offset_shared = threadIdx.y * BLOCK_SIZE_SH + threadIdx.x;

sh_block[offset_shared] = image[offset_t];
__syncthreads();

if((threadIdx.y == BLOCK_SIZE - 1))
{
sh_block[offset_shared + BLOCK_SIZE_SH] = image[offset_t + width];
sh_block[offset_shared + BLOCK_SIZE_SH*2] = image[offset_t + 2*width];
}
__syncthreads();

if((threadIdx.x == BLOCK_SIZE - 1))
{
sh_block[offset_shared + 1] = image[offset_t + 1];
sh_block[offset_shared + 2] = image[offset_t + 2];
}
__syncthreads();

if((threadIdx.y == BLOCK_SIZE - 1) && (threadIdx.x == BLOCK_SIZE - 1))
{
sh_block[offset_shared + BLOCK_SIZE_SH + 1] = image[offset_t + width + 1];
sh_block[offset_shared + BLOCK_SIZE_SH*2 + 1] = image[offset_t + width*2 + 1];
sh_block[offset_shared + BLOCK_SIZE_SH + 2] = image[offset_t + width + 2];
sh_block[offset_shared + BLOCK_SIZE_SH*2 + 2] = image[offset_t + width*2 + 2];
}
__syncthreads();

image_out[offset] = gpu_applyFilter(&sh_block[offset_shared],
BLOCK_SIZE_SH, gaussian, 3);
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
    cudaMalloc(&d_image, width * height * sizeof(float));
    cudaMalloc(&d_image_out, width * height * sizeof(float));
    
    // Warmup
    cudaFree(0);
    gpu_gaussian<<<gridBlock, threadBlock>>>(width, height, d_image, d_image_out);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < 10; ++i) {
        gpu_gaussian<<<gridBlock, threadBlock>>>(width, height, d_image, d_image_out);
    }
    cudaDeviceSynchronize();
    
    // Measure execution time
    auto start = steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        gpu_gaussian<<<gridBlock, threadBlock>>>(width, height, d_image, d_image_out);
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
