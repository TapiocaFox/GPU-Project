#include "includes.h"

/*
* matrix multiplication C += A*B
*  -> CUDA kernel
*     (implementation adopted from Kirk&Hwu:
*      "Programming Massively Parallel Processors, chapter 4)
*  -> Features: none (basic tiled version, using only global memory)
*/

/*
* matrix multiplication C += A*B
*  -> CUDA kernel
*     (implementation adopted from Kirk&Hwu:
*      "Programming Massively Parallel Processors, chapter 5)
*  -> Features:
*     - tiled matrix multiplication with use of shared memory
*/

/*
* matrix multiplication C += A*B
*  -> CUDA kernel
*     (implementation adopted from Kirk&Hwu:
*      "Programming Massively Parallel Processors, chapter 5)
*  -> Features:
*     - tiled matrix multiplication with use of shared memory
*     - coalesced memory access
*     - overlapping loads of subsequent tile pairs (using registers & shared memory)
*/

__global__ void matrixMultKernel_tiled(float* Ad, float* Bd, float* Cd, int n)
{
__shared__ float A_shared[TILE_SIZE][TILE_SIZE];
__shared__ float B_shared[TILE_SIZE][TILE_SIZE];

int row = blockIdx.y*TILE_SIZE + threadIdx.y;
int column = blockIdx.x*TILE_SIZE + threadIdx.x;

if(row >=n || column >=n)
{
return;
}

float Celem = 0.0;
for(int m = 0; m < n/TILE_SIZE;m++)
{
*(*(A_shared + threadIdx.y) + threadIdx.x) = *(Ad + row*n + m*TILE_SIZE + threadIdx.x);
*(*(B_shared + threadIdx.y) + threadIdx.x) = *(Bd + (m*TILE_SIZE + threadIdx.y)*n + column);
__syncthreads();

for(int k  = 0;k<TILE_SIZE;k++)
{
Celem += *(*(A_shared + threadIdx.y) + k)*(*(*(B_shared + k) + threadIdx.x));
}
__syncthreads();

}
*(Cd +  row*n + column) = Celem;

}