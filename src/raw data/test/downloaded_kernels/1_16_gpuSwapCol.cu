#include "includes.h"
#define NTHREADS 512





// Updates the column norms by subtracting the Hadamard-square of the
// Householder vector.
//
// N.B.:  Overflow incurred in computing the square should already have
// been detected in the original norm construction.

__global__ void gpuSwapCol(int rows, float * dArray, int coli, int * dColj, int * dPivot)
{
int rowIndex = blockIdx.x * blockDim.x + threadIdx.x;

if(rowIndex >= rows)
return;

int colj = coli + (*dColj);
float fholder;

fholder = dArray[rowIndex+coli*rows];
dArray[rowIndex+coli*rows] = dArray[rowIndex+colj*rows];
dArray[rowIndex+colj*rows] = fholder;

if((blockIdx.x == 0) && (threadIdx.x == 0)) {
int iholder = dPivot[coli];
dPivot[coli] = dPivot[colj];
dPivot[colj] = iholder;
}
}