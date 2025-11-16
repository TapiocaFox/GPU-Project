#include "includes.h"
__global__ void mask_kernel(int n,  float *x, float mask_num, float *mask)
{
int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
if(i < n && mask[i] == mask_num) x[i] = mask_num;
}