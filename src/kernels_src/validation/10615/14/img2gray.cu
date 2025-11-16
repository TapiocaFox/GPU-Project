#include "includes.h"
__global__ void img2gray(unsigned char *imageInput, int width, int height, unsigned char *imageOutput){
int row = blockIdx.y*blockDim.y+threadIdx.y;
int col = blockIdx.x*blockDim.x+threadIdx.x;

if((row < height) && (col < width)){
imageOutput[row*width+col] = imageInput[(row*width+col)*3+RED]*0.299 + imageInput[(row*width+col)*3+GREEN]*0.587
+ imageInput[(row*width+col)*3+BLUE]*0.114;
}
}