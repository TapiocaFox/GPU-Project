#include "includes.h"
__global__ void binarize_weights_kernel(float *weights, int n, int size, float *binary)
{
int f = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
if (f >= n) return;
int i = 0;
float mean = 0;
for(i = 0; i < size; ++i){
mean += abs(weights[f*size + i]);
}
mean = mean / size;
for(i = 0; i < size; ++i){
binary[f*size + i] = (weights[f*size + i] > 0) ? mean : -mean;
//binary[f*size + i] = weights[f*size + i];
}
}