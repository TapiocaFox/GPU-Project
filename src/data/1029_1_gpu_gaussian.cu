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
__device__ float gpu_applyFilter(float *image, int stride, float *matrix, int filter_dim)
{
////////////////
// TO-DO #5.2 ////////////////////////////////////////////////
// Implement the GPU version of cpu_applyFilter()           //
//                                                          //
// Does it make sense to have a separate gpu_applyFilter()? //
//////////////////////////////////////////////////////////////
float pixel = 0.0f;

for (int h = 0; h < filter_dim; h++)
{
int offset        = h * stride;
int offset_kernel = h * filter_dim;

for (int w = 0; w < filter_dim; w++)
{
pixel += image[offset + w] * matrix[offset_kernel + w];
}
}

return pixel;
}
__global__ void gpu_gaussian(int width, int height, float *image, float *image_out)
{
float gaussian[9] = { 1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f,
2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f,
1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f };

int index_x = blockIdx.x * blockDim.x + threadIdx.x;
int index_y = blockIdx.y * blockDim.y + threadIdx.y;

__shared__ float sh_block[BLOCK_SIZE_SH * BLOCK_SIZE_SH];


if (index_x < (width - 2) && index_y < (height - 2))
{
int offset_t = index_y * width + index_x;
int offset   = (index_y + 1) * width + (index_x + 1);
int offset_shared = threadIdx.y * BLOCK_SIZE_SH + threadIdx.x;

sh_block[offset_shared] = image[offset_t];
__syncthreads();

if((threadIdx.y == BLOCK_SIZE - 1))
{
sh_block[offset_shared + BLOCK_SIZE_SH] = image[offset_t + width];
sh_block[offset_shared + BLOCK_SIZE_SH*2] = image[offset_t + 2*width];
}
__syncthreads();

if((threadIdx.x == BLOCK_SIZE - 1))
{
sh_block[offset_shared + 1] = image[offset_t + 1];
sh_block[offset_shared + 2] = image[offset_t + 2];
}
__syncthreads();

if((threadIdx.y == BLOCK_SIZE - 1) && (threadIdx.x == BLOCK_SIZE - 1))
{
sh_block[offset_shared + BLOCK_SIZE_SH + 1] = image[offset_t + width + 1];
sh_block[offset_shared + BLOCK_SIZE_SH*2 + 1] = image[offset_t + width*2 + 1];
sh_block[offset_shared + BLOCK_SIZE_SH + 2] = image[offset_t + width + 2];
sh_block[offset_shared + BLOCK_SIZE_SH*2 + 2] = image[offset_t + width*2 + 2];
}
__syncthreads();

image_out[offset] = gpu_applyFilter(&sh_block[offset_shared],
BLOCK_SIZE_SH, gaussian, 3);
}
}