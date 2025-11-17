#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;

#define TILE_WIDTH 32

__global__ void MatrixMulSh( float *Md , float *Nd , float *Pd , const int WIDTH )
{

//Taking shared array to break the MAtrix in Tile widht and fatch them in that array per ele
__shared__ float Mds [TILE_WIDTH][TILE_WIDTH];
__shared__ float Nds [TILE_WIDTH][TILE_WIDTH];

// calculate thread id
unsigned int col = TILE_WIDTH*blockIdx.x + threadIdx.x;
unsigned int row = TILE_WIDTH*blockIdx.y + threadIdx.y;

for (int m = 0 ; m<WIDTH/TILE_WIDTH ; m++ ) // m indicate number of phase
{
Mds[threadIdx.y][threadIdx.x] =  Md[row*WIDTH + (m*TILE_WIDTH + threadIdx.x)];
Nds[threadIdx.y][threadIdx.x] =  Nd[ ( m*TILE_WIDTH + threadIdx.y) * WIDTH + col];
__syncthreads() ; // for syncronizeing the threads

// Do for tile
for ( int k = 0; k<TILE_WIDTH ; k++ )
Pd[row*WIDTH + col]+= Mds[threadIdx.x][k] * Nds[k][threadIdx.y];
__syncthreads() ; // for syncronizeing the threads

}
}

int main(int argc, char **argv) {
    cudaSetDevice(0);
    
    // Grid: (5, 5), Block: (32, 32)
    dim3 gridBlock(5, 5);
    dim3 threadBlock(32, 32);
    
    // Allocate device memory
    const int WIDTH = 160;
    float *d_Md = NULL;
    float *d_Nd = NULL;
    float *d_Pd = NULL;
    cudaMalloc(&d_Md, WIDTH * WIDTH * sizeof(float));
    cudaMalloc(&d_Nd, WIDTH * WIDTH * sizeof(float));
    cudaMalloc(&d_Pd, WIDTH * WIDTH * sizeof(float));
    
    // Warmup
    cudaFree(0);
    MatrixMulSh<<<gridBlock, threadBlock>>>(d_Md, d_Nd, d_Pd, WIDTH);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < 10; ++i) {
        MatrixMulSh<<<gridBlock, threadBlock>>>(d_Md, d_Nd, d_Pd, WIDTH);
    }
    cudaDeviceSynchronize();
    
    // Measure execution time
    auto start = steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        MatrixMulSh<<<gridBlock, threadBlock>>>(d_Md, d_Nd, d_Pd, WIDTH);
    }
    cudaDeviceSynchronize();
    auto end = steady_clock::now();
    
    auto usecs = duration_cast<duration<float, microseconds::period>>(end - start);
    float avg_time_us = usecs.count() / 1000.0f;
    float avg_time_ms = avg_time_us / 1000.0f;
    
    cout << "[" << avg_time_us << "," << avg_time_ms << "]" << endl;
    
    cudaFree(d_Md);
    cudaFree(d_Nd);
    cudaFree(d_Pd);
    
    return 0;
}
