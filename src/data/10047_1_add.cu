#include "includes.h"
//nvcc -ptx vectoradd_double.cu -o vectoradd_double.ptx

extern "C"
__global__ void add(int n, double *a, double *b, double *sum)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i<n)
{
sum[i] = a[i] + b[i];
}
}