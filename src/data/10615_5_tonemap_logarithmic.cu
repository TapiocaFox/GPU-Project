#include "includes.h"
__device__ float logarithmic_mapping(float k, float q, float val_pixel){
return (log10(1 + q * val_pixel))/(log10(1 + k * maxLum));
}
__global__ void tonemap_logarithmic(float* imageIn, float* imageOut, int width, int height, int channels, int depth, float q, float k){
//printf("maxLum : %f\n", maxLum);
int Row = blockDim.y * blockIdx.y + threadIdx.y;
int Col = blockDim.x * blockIdx.x + threadIdx.x;

if(Row < height && Col < width) {
imageOut[(Row*width+Col)*3+BLUE] = logarithmic_mapping(k, q, imageIn[(Row*width+Col)*3+BLUE]);
imageOut[(Row*width+Col)*3+GREEN] = logarithmic_mapping(k, q, imageIn[(Row*width+Col)*3+GREEN]);
imageOut[(Row*width+Col)*3+RED] = logarithmic_mapping(k, q, imageIn[(Row*width+Col)*3+RED]);
}
}