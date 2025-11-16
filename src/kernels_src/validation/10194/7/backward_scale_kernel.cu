#include "includes.h"
__global__ void backward_scale_kernel(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates)
{
__shared__ float part[BLOCK];
int i,b;
int filter = blockIdx.x;
int p = threadIdx.x;
float sum = 0;
for(b = 0; b < batch; ++b){
for(i = 0; i < size; i += BLOCK){
int index = p + i + size*(filter + n*b);
sum += (p+i < size) ? delta[index]*x_norm[index] : 0;
}
}
part[p] = sum;
__syncthreads();
if (p == 0) {
for(i = 0; i < BLOCK; ++i) scale_updates[filter] += part[i];
}
}