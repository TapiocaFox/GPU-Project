#include "includes.h"
__global__ void smooth_l1_kernel(int n, float *pred, float *truth, float *delta, float *error)
{
int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
if(i < n){
float diff = truth[i] - pred[i];
float abs_val = abs(diff);
if(abs_val < 1) {
error[i] = diff * diff;
delta[i] = diff;
}
else {
error[i] = 2*abs_val - 1;
delta[i] = (diff < 0) ? -1 : 1;
}
}
}