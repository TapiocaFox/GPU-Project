#include "includes.h"
__global__ void flatten_kernel(int N, float *x, int spatial, int layers, int batch, int forward, float *out)
{
int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
if(i >= N) return;
int in_s = i%spatial;
i = i/spatial;
int in_c = i%layers;
i = i/layers;
int b = i;

int i1 = b*layers*spatial + in_c*spatial + in_s;
int i2 = b*layers*spatial + in_s*layers +  in_c;

if (forward) out[i2] = x[i1];
else out[i1] = x[i2];
}