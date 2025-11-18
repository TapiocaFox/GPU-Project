#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;

#define NTHREADS 512

__global__ void gpuSwapCol(int rows, float * dArray, int coli, int * dColj, int * dPivot)
{
int rowIndex = blockIdx.x * blockDim.x + threadIdx.x;

if(rowIndex >= rows)
return;

int colj = coli + (*dColj);
float fholder;

fholder = dArray[rowIndex+coli*rows];
dArray[rowIndex+coli*rows] = dArray[rowIndex+colj*rows];
dArray[rowIndex+colj*rows] = fholder;

if((blockIdx.x == 0) && (threadIdx.x == 0)) {
int iholder = dPivot[coli];
dPivot[coli] = dPivot[colj];
dPivot[colj] = iholder;
}
}

int main(int argc, char **argv) {
    cudaSetDevice(0);
    
    // Grid: (5, 5), Block: (32, 32)
    dim3 gridBlock(5, 5);
    dim3 threadBlock(32, 32);
    
    // Allocate device memory
    int rows = 5 * 32 * 5 * 32;
    int cols = 100;
    int coli = 0;
    float *d_dArray = NULL;
    int *d_dColj = NULL;
    int *d_dPivot = NULL;
    cudaMalloc(&d_dArray, rows * cols * sizeof(float));
    cudaMalloc(&d_dColj, sizeof(int));
    cudaMalloc(&d_dPivot, cols * sizeof(int));
    
    int h_dColj = 1;
    cudaMemcpy(d_dColj, &h_dColj, sizeof(int), cudaMemcpyHostToDevice);
    
    // Warmup
    cudaFree(0);
    gpuSwapCol<<<gridBlock, threadBlock>>>(rows, d_dArray, coli, d_dColj, d_dPivot);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < 10; ++i) {
        gpuSwapCol<<<gridBlock, threadBlock>>>(rows, d_dArray, coli, d_dColj, d_dPivot);
    }
    cudaDeviceSynchronize();
    
    // Measure execution time
    auto start = steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        gpuSwapCol<<<gridBlock, threadBlock>>>(rows, d_dArray, coli, d_dColj, d_dPivot);
    }
    cudaDeviceSynchronize();
    auto end = steady_clock::now();
    
    auto usecs = duration_cast<duration<float, microseconds::period>>(end - start);
    float avg_time_us = usecs.count() / 1000.0f;
    float avg_time_ms = avg_time_us / 1000.0f;
    
    cout << "[" << avg_time_us << "," << avg_time_ms << "]" << endl;
    
    cudaFree(d_dArray);
    cudaFree(d_dColj);
    cudaFree(d_dPivot);
    
    return 0;
}
