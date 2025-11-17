#include "includes.h"
__global__ void grayscaleVer2D(uchar3* input, uchar3* output, int imageWidth, int imageHeight){
int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
int tid_y = threadIdx.y + blockIdx.y * blockDim.y;
if(tid_x > imageWidth || tid_y > imageHeight) return;
int tid = (int)(tid_x + tid_y * imageWidth);
output[tid].x = (input[tid].x + input[tid].y + input[tid].z) / 3;
output[tid].z = output[tid].y = output[tid].x;
}