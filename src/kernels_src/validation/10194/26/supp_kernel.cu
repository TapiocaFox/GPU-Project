#include "includes.h"
__global__ void supp_kernel(int N, float ALPHA, float *X, int INCX)
{
int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
if(i < N) {
if((X[i*INCX] * X[i*INCX]) < (ALPHA * ALPHA)) X[i*INCX] = 0;
}
}