#include "includes.h"
__global__ void pow_kernel(int N, float ALPHA, float *X, int INCX, float *Y, int INCY)
{
int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
if(i < N) Y[i*INCY] = pow(X[i*INCX], ALPHA);
}