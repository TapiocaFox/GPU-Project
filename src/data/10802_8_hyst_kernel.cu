#include "includes.h"
__global__ void hyst_kernel(unsigned char *data, unsigned char *out, int rows, int cols) {
// Establish our high and low thresholds as floats
float lowThresh  = 10;
float highThresh = 70;

// These variables are offset by one to avoid seg. fault errors
// As such, this kernel ignores the outside ring of pixels
const int row = blockIdx.y * blockDim.y + threadIdx.y + 1;
const int col = blockIdx.x * blockDim.x + threadIdx.x + 1;
const int pos = row * cols + col;

const unsigned char EDGE = 255;

unsigned char magnitude = data[pos];

if(magnitude >= highThresh)
out[pos] = EDGE;
else if(magnitude <= lowThresh)
out[pos] = 0;
else {
float med = (highThresh + lowThresh) / 2;

if(magnitude >= med)
out[pos] = EDGE;
else
out[pos] = 0;
}
}