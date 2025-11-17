#include "includes.h"

static __device__ float E = 2.718281828;




__global__ void reduceArgMaxKernel(float *src, float *dst, float *arg, int dim_size, int block_size)
{
int di = blockIdx.x * block_size + threadIdx.x;
int si = di * dim_size;
float now = src[si], max = now;
int maxi = 0;
for (int i = 1; i < dim_size; i++) {
now = src[si+i];
if (now > max) {
max = now;
maxi = i;
}
}
dst[di] = max;
arg[di] = maxi;
}