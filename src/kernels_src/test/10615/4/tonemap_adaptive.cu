#include "includes.h"
__device__ float adaptive_mapping(float k, float q, float val_pixel){
return 	(k*log(1 + val_pixel))/((100*log10(1 + maxLum)) * ( powf((log(2+8*(val_pixel/maxLum))), (log(q)/log(0.5)) ) )	);
}
__global__ void tonemap_adaptive(float* imageIn, float* imageOut, int width, int height, int channels, int depth, float q, float k){
//printf("maxLum : %f\n", maxLum);
int Row = blockDim.y * blockIdx.y + threadIdx.y;
int Col = blockDim.x * blockIdx.x + threadIdx.x;

if(Row < height && Col < width) {
imageOut[(Row*width+Col)*3+BLUE] = adaptive_mapping(k, q, imageIn[(Row*width+Col)*3+BLUE]);
imageOut[(Row*width+Col)*3+GREEN] = adaptive_mapping(k, q, imageIn[(Row*width+Col)*3+GREEN]);
imageOut[(Row*width+Col)*3+RED] = adaptive_mapping(k, q, imageIn[(Row*width+Col)*3+RED]);
}
}