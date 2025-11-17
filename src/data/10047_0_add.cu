#include "includes.h"
//nvcc -ptx vectoradd_float.cu -o vectoradd_float.ptx

extern "C"
__global__ void add(int n, float *a, float *b, float *sum)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i<n)
{
sum[i] = a[i] + b[i];
}
}