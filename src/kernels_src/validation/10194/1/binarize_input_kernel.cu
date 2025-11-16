#include "includes.h"
__global__ void binarize_input_kernel(float *input, int n, int size, float *binary)
{
int s = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
if (s >= size) return;
int i = 0;
float mean = 0;
for(i = 0; i < n; ++i){
mean += abs(input[i*size + s]);
}
mean = mean / n;
for(i = 0; i < n; ++i){
binary[i*size + s] = (input[i*size + s] > 0) ? mean : -mean;
}
}