#include "includes.h"
__global__ void mul_kernel(int N, float *X, int INCX, float *Y, int INCY)
{
int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
if(i < N) Y[i*INCY] *= X[i*INCX];
}