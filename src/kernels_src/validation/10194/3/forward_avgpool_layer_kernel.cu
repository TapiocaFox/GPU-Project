#include "includes.h"
__global__ void forward_avgpool_layer_kernel(int n, int w, int h, int c, float *input, float *output)
{
int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
if(id >= n) return;

int k = id % c;
id /= c;
int b = id;

int i;
int out_index = (k + c*b);
output[out_index] = 0;
for(i = 0; i < w*h; ++i){
int in_index = i + h*w*(k + b*c);
output[out_index] += input[in_index];
}
output[out_index] /= w*h;
}