#include "includes.h"
__global__ void backward_avgpool_layer_kernel(int n, int w, int h, int c, float *in_delta, float *out_delta)
{
int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
if(id >= n) return;

int k = id % c;
id /= c;
int b = id;

int i;
int out_index = (k + c*b);
for(i = 0; i < w*h; ++i){
int in_index = i + h*w*(k + b*c);
in_delta[in_index] += out_delta[out_index] / (w*h);
}
}