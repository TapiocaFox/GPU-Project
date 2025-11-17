#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;

__global__ void matrixMultiply(float * A, float * B, float * C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns) {
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
if ((row < numCRows) && (col < numCColumns)) {
float value = 0;
#pragma unroll
for (int k = 0; k < numAColumns; ++k)
value += A[row * numAColumns + k] * B[k * numBColumns + col];
C[row * numCColumns + col] = value;
}
}

int main(int argc, char **argv) {
    cudaSetDevice(0);
    
    // Grid: (5, 5), Block: (32, 32)
    dim3 gridBlock(5, 5);
    dim3 threadBlock(32, 32);
    
    // Allocate device memory
    int numARows = 5 * 32;
    int numAColumns = 5 * 32;
    int numBRows = 5 * 32;
    int numBColumns = 5 * 32;
    int numCRows = numARows;
    int numCColumns = numBColumns;
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    cudaMalloc(&d_A, numARows * numAColumns * sizeof(float));
    cudaMalloc(&d_B, numBRows * numBColumns * sizeof(float));
    cudaMalloc(&d_C, numCRows * numCColumns * sizeof(float));
    
    // Warmup
    cudaFree(0);
    matrixMultiply<<<gridBlock, threadBlock>>>(d_A, d_B, d_C, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < 10; ++i) {
        matrixMultiply<<<gridBlock, threadBlock>>>(d_A, d_B, d_C, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
    }
    cudaDeviceSynchronize();
    
    // Measure execution time
    auto start = steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        matrixMultiply<<<gridBlock, threadBlock>>>(d_A, d_B, d_C, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
    }
    cudaDeviceSynchronize();
    auto end = steady_clock::now();
    
    auto usecs = duration_cast<duration<float, microseconds::period>>(end - start);
    float avg_time_us = usecs.count() / 1000.0f;
    float avg_time_ms = avg_time_us / 1000.0f;
    
    cout << "[" << avg_time_us << "," << avg_time_ms << "]" << endl;
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
}
