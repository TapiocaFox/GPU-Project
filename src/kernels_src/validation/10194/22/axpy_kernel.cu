#include "includes.h"
__global__ void axpy_kernel(int N, float ALPHA, float *X, int OFFX, int INCX,  float *Y, int OFFY, int INCY)
{
int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
if(i < N) Y[OFFY+i*INCY] += ALPHA*X[OFFX+i*INCX];
}