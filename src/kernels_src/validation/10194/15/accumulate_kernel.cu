#include "includes.h"
__global__ void accumulate_kernel(float *x, int n, int groups, float *sum)
{
int k;
int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
if (i >= groups) return;
sum[i] = 0;
for(k = 0; k < n; ++k){
sum[i] += x[k*groups + i];
}
}