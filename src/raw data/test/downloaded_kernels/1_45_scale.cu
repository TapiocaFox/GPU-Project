#include "includes.h"
__global__ void scale(float knot_max, int nx, int nsamples, float * x, int pitch_x)
{
int
col_idx = blockDim.x * blockIdx.x + threadIdx.x;

if(col_idx >= nx) return;

float
min, max,
* col = x + col_idx * pitch_x;

// find the min and the max
min = max = col[0];
for(int i = 1; i < nsamples; i++) {
if(col[i] < min) min = col[i];
if(col[i] > max) max = col[i];
}

float delta = max - min;
for(int i = 0; i < nsamples; i++)
col[i] = (knot_max * (col[i] - min)) / delta;
}