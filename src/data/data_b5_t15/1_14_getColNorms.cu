#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;

#define NTHREADS 512

// Updates the column norms by subtracting the Hadamard-square of the
// Householder vector.
//
// N.B.:  Overflow incurred in computing the square should already have
// been detected in the original norm construction.

__global__ void getColNorms(int rows, int cols, float * da, int lda, float * colNorms)
{
int colIndex = threadIdx.x + blockIdx.x * blockDim.x;
float
sum = 0.f, term,
* col;

if(colIndex >= cols)
return;

col = da + colIndex * lda;

for(int i = 0; i < rows; i++) {
term = col[i];
term *= term;
sum += term;
}

colNorms[colIndex] = sum;
}

int main(int argc, char **argv) {
    cudaSetDevice(0);
    
    // Grid: (5, 5), Block: (32, 32)
    dim3 gridBlock(5, 5);
    dim3 threadBlock(32, 32);
    
    // Allocate device memory
    int rows = 100;
    int cols = 5 * 32 * 5 * 32;
    int lda = rows;
    float *d_da = NULL;
    float *d_colNorms = NULL;
    cudaMalloc(&d_da, rows * cols * sizeof(float));
    cudaMalloc(&d_colNorms, cols * sizeof(float));
    
    // Warmup
    cudaFree(0);
    getColNorms<<<gridBlock, threadBlock>>>(rows, cols, d_da, lda, d_colNorms);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < 10; ++i) {
        getColNorms<<<gridBlock, threadBlock>>>(rows, cols, d_da, lda, d_colNorms);
    }
    cudaDeviceSynchronize();
    
    // Measure execution time
    auto start = steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        getColNorms<<<gridBlock, threadBlock>>>(rows, cols, d_da, lda, d_colNorms);
    }
    cudaDeviceSynchronize();
    auto end = steady_clock::now();
    
    auto usecs = duration_cast<duration<float, microseconds::period>>(end - start);
    float avg_time_us = usecs.count() / 1000.0f;
    float avg_time_ms = avg_time_us / 1000.0f;
    
    cout << "[" << avg_time_us << "," << avg_time_ms << "]" << endl;
    
    cudaFree(d_da);
    cudaFree(d_colNorms);
    
    return 0;
}
