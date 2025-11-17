#include "includes.h"
#define NTHREADS 512





// Updates the column norms by subtracting the Hadamard-square of the
// Householder vector.
//
// N.B.:  Overflow incurred in computing the square should already have
// been detected in the original norm construction.

__global__ void getColNorms(int rows, int cols, float * da, int lda, float * colNorms)
{
int colIndex = threadIdx.x + blockIdx.x * blockDim.x;
float
sum = 0.f, term,
* col;

if(colIndex >= cols)
return;

col = da + colIndex * lda;

// debug printing
// printf("printing column %d\n", colIndex);
// for(int i = 0; i < rows; i++)
// printf("%f, ", col[i]);
// puts("");
// end debug printing

for(int i = 0; i < rows; i++) {
term = col[i];
term *= term;
sum += term;
}

// debug printing
// printf("norm %f\n", norm);
// end debug printing

colNorms[colIndex] = sum;
}