#include "includes.h"
__global__ void adam_kernel(int N, float *x, float *m, float *v, float B1, float B2, float rate, float eps, int t)
{
int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
if (index >= N) return;

x[index] = x[index] - (rate * sqrt(1.-pow(B2, t)) / (1.-pow(B1, t)) * m[index] / (sqrt(v[index]) + eps));
//if(index == 0) printf("%f %f %f %f\n", m[index], v[index], (rate * sqrt(1.-pow(B2, t)) / (1.-pow(B1, t)) * m[index] / (sqrt(v[index]) + eps)));
}