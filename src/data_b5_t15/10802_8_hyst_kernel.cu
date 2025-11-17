#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;

__global__ void hyst_kernel(unsigned char *data, unsigned char *out, int rows, int cols) {
// Establish our high and low thresholds as floats
float lowThresh  = 10;
float highThresh = 70;

// These variables are offset by one to avoid seg. fault errors
// As such, this kernel ignores the outside ring of pixels
const int row = blockIdx.y * blockDim.y + threadIdx.y + 1;
const int col = blockIdx.x * blockDim.x + threadIdx.x + 1;
const int pos = row * cols + col;

const unsigned char EDGE = 255;

unsigned char magnitude = data[pos];

if(magnitude >= highThresh)
out[pos] = EDGE;
else if(magnitude <= lowThresh)
out[pos] = 0;
else {
float med = (highThresh + lowThresh) / 2;

if(magnitude >= med)
out[pos] = EDGE;
else
out[pos] = 0;
}
}

int main(int argc, char **argv) {
    cudaSetDevice(0);
    
    // Grid: (5, 5), Block: (32, 32)
    dim3 gridBlock(5, 5);
    dim3 threadBlock(32, 32);
    
    // Allocate device memory
    int rows = 160;
    int cols = 160;
    unsigned char *d_data = NULL;
    unsigned char *d_out = NULL;
    cudaMalloc(&d_data, rows * cols * sizeof(unsigned char));
    cudaMalloc(&d_out, rows * cols * sizeof(unsigned char));
    
    // Warmup
    cudaFree(0);
    hyst_kernel<<<gridBlock, threadBlock>>>(d_data, d_out, rows, cols);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < 10; ++i) {
        hyst_kernel<<<gridBlock, threadBlock>>>(d_data, d_out, rows, cols);
    }
    cudaDeviceSynchronize();
    
    // Measure execution time
    auto start = steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        hyst_kernel<<<gridBlock, threadBlock>>>(d_data, d_out, rows, cols);
    }
    cudaDeviceSynchronize();
    auto end = steady_clock::now();
    
    auto usecs = duration_cast<duration<float, microseconds::period>>(end - start);
    float avg_time_us = usecs.count() / 1000.0f;
    float avg_time_ms = avg_time_us / 1000.0f;
    
    cout << "[" << avg_time_us << "," << avg_time_ms << "]" << endl;
    
    cudaFree(d_data);
    cudaFree(d_out);
    
    return 0;
}
