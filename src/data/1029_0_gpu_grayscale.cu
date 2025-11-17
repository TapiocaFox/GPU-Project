#include "includes.h"


#define BLOCK_SIZE  16
#define HEADER_SIZE 122
#define BLOCK_SIZE_SH 18

typedef unsigned char BYTE;

/**
* Structure that represents a BMP image.
*/
typedef struct
{
int   width;
int   height;
float *data;
} BMPImage;

typedef struct timeval tval;

BYTE g_info[HEADER_SIZE]; // Reference header

/**
* Reads a BMP 24bpp file and returns a BMPImage structure.
* Thanks to https://stackoverflow.com/a/9296467
*/
__global__ void gpu_grayscale(int width, int height, float *image, float *image_out)
{
////////////////
// TO-DO #4.2 /////////////////////////////////////////////
// Implement the GPU version of the grayscale conversion //
///////////////////////////////////////////////////////////
const int h = blockIdx.y*blockDim.y + threadIdx.y;
const int w = blockIdx.x*blockDim.x + threadIdx.x;

int offset_out = h * width;
int offset = offset_out * 3;

if(h < height && w < width)
{
float *pixel = &image[offset + w * 3];
image_out[offset_out + w] = pixel[0] * 0.0722f + // B
pixel[1] * 0.7152f + // G
pixel[2] * 0.2126f;  // R
}

}