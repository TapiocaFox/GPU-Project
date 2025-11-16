#include "includes.h"
#define FALSE 0
#define TRUE !FALSE

#define NUMTHREADS 16
#define THREADWORK 32








__device__ int dIsSignificant(float signif, int df) {
float tcutoffs[49] = {
// cuttoffs for degrees of freedom <= 30
637.000, 31.600, 2.920, 8.610, 6.869, 5.959, 5.408, 5.041, 4.781,
4.587, 4.437, 4.318, 4.221, 4.140, 4.073, 4.015, 3.965, 3.922,
3.883, 3.850, 3.819, 3.792, 3.768, 3.745, 3.725, 3.707, 3.690,
3.674, 3.659, 3.646,
// cuttoffs for even degrees of freedom > 30 but <= 50
3.622, 3.601, 3.582, 3.566, 3.551, 3.538, 3.526, 3.515, 3.505, 3.496,
// 55 <= df <= 70 by 5s
3.476, 3.460, 3.447, 3.435,
3.416, // 80
3.390, // 100
3.357, // 150
3.340, // 200
3.290  // > 200
};

size_t index = 0;
if(df <= 0) return 0;
else if(df <= 30) index = df - 1;
else if(df <= 50) index = 30 + (df + (df%2) - 32) / 2;
else if(df <= 70) {
if(df <= 55) index = 40;
else if(df <= 60) index = 41;
else if(df <= 65) index = 42;
else if(df <= 70) index = 43;
}
else if(df <= 80) index = 44;
else if(df <= 100) index = 45;
else if(df <= 150) index = 46;
else if(df <= 200) index = 47;
else if(df > 200) index = 48;

if(fabsf(signif) < tcutoffs[index]) return FALSE;

return TRUE;
}
__global__ void dUpdateSignif(const float * gpuData, size_t n, float * gpuResults)
{
size_t
i, start, inrow, outrow,
bx = blockIdx.x, tx = threadIdx.x;
float
radicand, cor, npairs, tscore;

start = bx * NUMTHREADS * THREADWORK + tx * THREADWORK;

for(i = 0; i < THREADWORK; i++) {
if(start+i > n) break;

inrow = (start+i)*5;
outrow = (start+i)*6;

cor = gpuData[inrow+3];
npairs = gpuData[inrow+4];

if(cor >= 0.999)
tscore = 10000.0;
else {
radicand = (npairs - 2.f) / (1.f - cor * cor);
tscore = cor * sqrtf(radicand);
}
if(dIsSignificant(tscore, (int)npairs)) {
gpuResults[outrow]   = gpuData[inrow];
gpuResults[outrow+1] = gpuData[inrow+1];
gpuResults[outrow+2] = gpuData[inrow+2];
gpuResults[outrow+3] = cor;
gpuResults[outrow+4] = tscore;
gpuResults[outrow+5] = npairs;
} else {
gpuResults[outrow] = -1.f;
}
}
}