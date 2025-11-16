#include "includes.h"
__global__ void backward_bias_kernel(float *bias_updates, float *delta, int batch, int n, int size)
{
__shared__ float part[BLOCK];
int i,b;
int filter = blockIdx.x;
int p = threadIdx.x;
float sum = 0;
for(b = 0; b < batch; ++b){
for(i = 0; i < size; i += BLOCK){
int index = p + i + size*(filter + n*b);
sum += (p+i < size) ? delta[index] : 0;
}
}
part[p] = sum;
__syncthreads();
if (p == 0) {
for(i = 0; i < BLOCK; ++i) bias_updates[filter] += part[i];
}
}