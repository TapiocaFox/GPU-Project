#include "includes.h"
__global__ void d_sobelFilter(unsigned char* imageIn, unsigned char* imageOut, int width, int height, int maskWidth, char* M) {
int Row = blockDim.y * blockIdx.y + threadIdx.y;
int Col = blockDim.x * blockIdx.x + threadIdx.x;

int nrow = Row - (maskWidth/2);
int ncol = Col - (maskWidth/2);
int res = 0;

if(Row < height && Col < width) {
for(int i=0; i<maskWidth; i++) {
for(int j=0; j<maskWidth; j++) {
if((nrow + i >= 0 && nrow + i < height) && (ncol + j >= 0 && ncol + j < width)) {
res += imageIn[(nrow + i)*width + (ncol + j)] * M[i*maskWidth + j];
}
}
}
if(res < 0)
res = 0;
else
if(res > 255)
res = 255;
imageOut[Row*width+Col] = (unsigned char)res;
}
}