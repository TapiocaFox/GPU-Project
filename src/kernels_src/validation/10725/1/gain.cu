#include "includes.h"
__global__ void gain(int width, int height, float rGain, float gGain, float bGain, float* input, float* output)
{
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;

if ((x < width) && (y < height))
{
int index = (y * width + x) * 4;
output[index + 0] = input[index + 0] * rGain;
output[index + 1] = input[index + 1] * gGain;
output[index + 2] = input[index + 2] * bGain;
output[index + 3] = input[index + 3];
}
}