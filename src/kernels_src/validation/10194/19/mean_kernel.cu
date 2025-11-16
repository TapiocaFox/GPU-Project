#include "includes.h"
__global__ void  mean_kernel(float *x, int batch, int filters, int spatial, float *mean)
{
float scale = 1./(batch * spatial);
int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
if (i >= filters) return;
int j,k;
mean[i] = 0;
for(j = 0; j < batch; ++j){
for(k = 0; k < spatial; ++k){
int index = j*filters*spatial + i*spatial + k;
mean[i] += x[index];
}
}
mean[i] *= scale;
}