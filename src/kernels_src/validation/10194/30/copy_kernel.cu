#include "includes.h"
__global__ void copy_kernel(int N,  float *X, int OFFX, int INCX, float *Y, int OFFY, int INCY)
{
int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
if(i < N) Y[i*INCY + OFFY] = X[i*INCX + OFFX];
}