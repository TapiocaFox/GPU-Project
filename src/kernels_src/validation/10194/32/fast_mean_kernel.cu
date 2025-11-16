#include "includes.h"
__global__ void  fast_mean_kernel(float *x, int batch, int filters, int spatial, float *mean)
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
local[id] += (i+id < spatial) ? x[index] : 0;
}
}

if(id == 0){
mean[filter] = 0;
for(i = 0; i < threads; ++i){
mean[filter] += local[i];
}
mean[filter] /= spatial * batch;
}
}