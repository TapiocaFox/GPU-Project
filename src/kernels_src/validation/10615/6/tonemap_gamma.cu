#include "includes.h"
__device__ float gamma_correction(float f_stop, float gamma, float val)
{
return powf((val*powf(2,f_stop)),(1.0/gamma));
}
__global__ void tonemap_gamma(float* imageIn, float* imageOut, int width, int height, int channels, int depth, float f_stop, float gamma)
{
int Row = blockDim.y * blockIdx.y + threadIdx.y;
int Col = blockDim.x * blockIdx.x + threadIdx.x;

if(Row < height && Col < width) {
imageOut[(Row*width+Col)*3+BLUE] = gamma_correction(f_stop, gamma, imageIn[(Row*width+Col)*3+BLUE]);
imageOut[(Row*width+Col)*3+GREEN] = gamma_correction(f_stop, gamma, imageIn[(Row*width+Col)*3+GREEN]);
imageOut[(Row*width+Col)*3+RED] = gamma_correction(f_stop, gamma, imageIn[(Row*width+Col)*3+RED]);
}
}