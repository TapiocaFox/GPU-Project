#include "includes.h"
__global__ void rgb2gray(unsigned char* d_Pin, unsigned char* d_Pout, int width, int height) {
int Row = blockIdx.y*blockDim.y + threadIdx.y;
int Col = blockIdx.x*blockDim.x + threadIdx.x;

if((Row < height) && (Col < width)) {
d_Pout[Row*width+Col] = d_Pin[(Row*width+Col)*3+BLUE]*0.114 + d_Pin[(Row*width+Col)*3+GREEN]*0.587 + d_Pin[(Row*width+Col)*3+RED]*0.299;

}
}