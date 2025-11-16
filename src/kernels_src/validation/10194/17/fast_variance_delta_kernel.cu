#include "includes.h"
__global__ void  fast_variance_delta_kernel(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta)
{
const int threads = BLOCK;
__shared__ float local[threads];

int id = threadIdx.x;
local[id] = 0;

int filter = blockIdx.x;

int i, j;
for(j = 0; j < batch; ++j){
for(i = 0; i < spatial; i += threads){
int index = j*spatial*filters + filter*spatial + i + id;

local[id] += (i+id < spatial) ? delta[index]*(x[index] - mean[filter]) : 0;
}
}

if(id == 0){
variance_delta[filter] = 0;
for(i = 0; i < threads; ++i){
variance_delta[filter] += local[i];
}
variance_delta[filter] *= -.5 * pow(variance[filter] + .000001f, (float)(-3./2.));
}
}