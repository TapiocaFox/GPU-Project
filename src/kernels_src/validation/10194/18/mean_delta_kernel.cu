#include "includes.h"
__global__ void mean_delta_kernel(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta)
{
int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
if (i >= filters) return;
int j,k;
mean_delta[i] = 0;
for (j = 0; j < batch; ++j) {
for (k = 0; k < spatial; ++k) {
int index = j*filters*spatial + i*spatial + k;
mean_delta[i] += delta[index];
}
}
mean_delta[i] *= (-1./sqrt(variance[i] + .000001f));
}