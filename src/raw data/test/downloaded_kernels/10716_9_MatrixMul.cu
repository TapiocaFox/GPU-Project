#include "includes.h"
__global__ void MatrixMul( float *Md , float *Nd , float *Pd , const int WIDTH )
{
// calculate thread id
unsigned int col = TILE_WIDTH*blockIdx.x + threadIdx.x;
unsigned int row = TILE_WIDTH*blockIdx.y + threadIdx.y;

for (int k = 0 ; k<WIDTH ; k++ )
{
Pd[row*WIDTH + col]+= Md[row * WIDTH + k ] * Nd[ k * WIDTH + col];
}
}