#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;

struct uchar3 {
    unsigned char x, y, z;
};

__global__ void grayscaleVer2D(uchar3* input, uchar3* output, int imageWidth, int imageHeight){
int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
int tid_y = threadIdx.y + blockIdx.y * blockDim.y;
if(tid_x > imageWidth || tid_y > imageHeight) return;
int tid = (int)(tid_x + tid_y * imageWidth);
output[tid].x = (input[tid].x + input[tid].y + input[tid].z) / 3;
output[tid].z = output[tid].y = output[tid].x;
}

int main(int argc, char **argv) {
    cudaSetDevice(0);
    
    // Grid: (5, 5), Block: (32, 32)
    dim3 gridBlock(5, 5);
    dim3 threadBlock(32, 32);
    
    // Allocate device memory
    int imageWidth = 160;
    int imageHeight = 160;
    uchar3 *d_input = NULL;
    uchar3 *d_output = NULL;
    cudaMalloc(&d_input, imageWidth * imageHeight * sizeof(uchar3));
    cudaMalloc(&d_output, imageWidth * imageHeight * sizeof(uchar3));
    
    // Warmup
    cudaFree(0);
    grayscaleVer2D<<<gridBlock, threadBlock>>>(d_input, d_output, imageWidth, imageHeight);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < 10; ++i) {
        grayscaleVer2D<<<gridBlock, threadBlock>>>(d_input, d_output, imageWidth, imageHeight);
    }
    cudaDeviceSynchronize();
    
    // Measure execution time
    auto start = steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        grayscaleVer2D<<<gridBlock, threadBlock>>>(d_input, d_output, imageWidth, imageHeight);
    }
    cudaDeviceSynchronize();
    auto end = steady_clock::now();
    
    auto usecs = duration_cast<duration<float, microseconds::period>>(end - start);
    float avg_time_us = usecs.count() / 1000.0f;
    float avg_time_ms = avg_time_us / 1000.0f;
    
    cout << "[" << avg_time_us << "," << avg_time_ms << "]" << endl;
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
}
