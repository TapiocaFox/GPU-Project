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

__global__ void matrixMultKernel_global(float* Ad, float* Bd, float* Cd, int n)
{
int row = blockIdx.y*blockDim.y + threadIdx.y;
int column = blockIdx.x*blockDim.x + threadIdx.x;

if(row >=n || column >=n)
{
return;
}

float Celem = 0.0;
for(int j = 0;j<n;j++)
{
Celem += *(Ad + row*n + j)*(*(Bd + j*n + column));
}

*(Cd + row*n + column) = Celem;

}