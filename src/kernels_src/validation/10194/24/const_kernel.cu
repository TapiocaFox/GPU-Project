#include "includes.h"
__global__ void const_kernel(int N, float ALPHA, float *X, int INCX)
{
int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
if(i < N) X[i*INCX] = ALPHA;
}