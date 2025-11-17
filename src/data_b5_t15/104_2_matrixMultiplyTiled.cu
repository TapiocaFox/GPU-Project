#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;

#define TILE_WIDTH 32

__global__ void matrixMultiplyTiled(float * A, float * B, float * C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns) {
__shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
__shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];
unsigned int tx = threadIdx.x;
unsigned int ty = threadIdx.y;
unsigned int col = blockIdx.x * TILE_WIDTH + tx;
unsigned int row = blockIdx.y * TILE_WIDTH + ty;
float acc = 0;

for (int t = 0; t < (numAColumns-1)/TILE_WIDTH + 1; ++t) {
unsigned int ATilePitch = t * TILE_WIDTH + tx;
unsigned int BTilePitch = t * TILE_WIDTH + ty;

if (row < numARows && ATilePitch < numAColumns)
ds_A[ty][tx] = A[row * numAColumns + ATilePitch];
else
ds_A[ty][tx] = 0;

if (col < numBColumns && BTilePitch < numBRows)
ds_B[ty][tx] = B[BTilePitch * numBColumns + col];
else
ds_B[ty][tx] = 0;

__syncthreads();
#pragma unroll
for (int k = 0; k < TILE_WIDTH; ++k) acc += ds_A[ty][k] * ds_B[k][tx];
__syncthreads();
}

if (row < numCRows && col < numCColumns) C[row * numCColumns + col] = acc;
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
    matrixMultiplyTiled<<<gridBlock, threadBlock>>>(d_A, d_B, d_C, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < 10; ++i) {
        matrixMultiplyTiled<<<gridBlock, threadBlock>>>(d_A, d_B, d_C, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
    }
    cudaDeviceSynchronize();
    
    // Measure execution time
    auto start = steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        matrixMultiplyTiled<<<gridBlock, threadBlock>>>(d_A, d_B, d_C, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
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
