#include "includes.h"
__global__ void fast_mean_delta_kernel(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta)
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
local[id] += (i+id < spatial) ? delta[index] : 0;
}
}

if(id == 0){
mean_delta[filter] = 0;
for(i = 0; i < threads; ++i){
mean_delta[filter] += local[i];
}
mean_delta[filter] *= (-1./sqrt(variance[filter] + .000001f));
}
}