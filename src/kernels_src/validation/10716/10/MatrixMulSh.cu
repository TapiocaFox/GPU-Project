#include "includes.h"
__global__ void MatrixMulSh( float *Md , float *Nd , float *Pd , const int WIDTH )
{

//Taking shared array to break the MAtrix in Tile widht and fatch them in that array per ele
__shared__ float Mds [TILE_WIDTH][TILE_WIDTH];
__shared__ float Nds [TILE_WIDTH][TILE_WIDTH];

// calculate thread id
unsigned int col = TILE_WIDTH*blockIdx.x + threadIdx.x;
unsigned int row = TILE_WIDTH*blockIdx.y + threadIdx.y;

for (int m = 0 ; m<WIDTH/TILE_WIDTH ; m++ ) // m indicate number of phase
{
Mds[threadIdx.y][threadIdx.x] =  Md[row*WIDTH + (m*TILE_WIDTH + threadIdx.x)];
Nds[threadIdx.y][threadIdx.x] =  Nd[ ( m*TILE_WIDTH + threadIdx.y) * WIDTH + col];
__syncthreads() ; // for syncronizeing the threads

// Do for tile
for ( int k = 0; k<TILE_WIDTH ; k++ )
Pd[row*WIDTH + col]+= Mds[threadIdx.x][k] * Nds[k][threadIdx.y];
__syncthreads() ; // for syncronizeing the threads

}
}