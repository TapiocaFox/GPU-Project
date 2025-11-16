#include "includes.h"
__global__ void  fast_variance_kernel(float *x, float *mean, int batch, int filters, int spatial, float *variance)
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

local[id] += (i+id < spatial) ? pow((x[index] - mean[filter]), 2) : 0;
}
}

if(id == 0){
variance[filter] = 0;
for(i = 0; i < threads; ++i){
variance[filter] += local[i];
}
variance[filter] /= (spatial * batch - 1);
}
}