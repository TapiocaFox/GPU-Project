#include "includes.h"
#define max(a, b) ((a > b)?a:b)

#define THREADSPERDIM   16

#define FALSE 0
#define TRUE !FALSE

// mX has order rows x cols
// vectY has length rows

// mX has order rows x cols
// vectY has length rows

__global__ void getRestricted(int countx, int county, int rows, int cols, float * mX, int mXdim, float * vY, int vYdim, float * mQ, int mQdim, float * mR, int mRdim, float * vectB, int vectBdim) {

int
m = blockIdx.x * THREADSPERDIM + threadIdx.x, n,
i, j, k;
float
sum, invnorm,
* X, * Y, * Q, * R, * B,
* coli, * colj,
* colQ, * colX;

if(m >= county) return;
if(m == 1) n = 0;
else n = 1;

X = mX + (m * mXdim);
// initialize the intercepts
for(i = 0; i < rows; i++)
X[i] = 1.f;

Y = vY + (m * countx + n) * vYdim;
B = vectB + m * vectBdim;
Q = mQ + m * mQdim;
R = mR + m * mRdim;

// initialize Q with X ...
for(i = 0; i < rows; i++) {
for(j = 0; j < cols; j++)
Q[i+j*rows] = X[i+j*rows];
}

// gramm-schmidt process to find Q
for(j = 0; j < cols; j++) {
colj = Q+rows*j;
for(i = 0; i < j; i++) {
coli = Q+rows*i;
sum = 0.f;
for(k = 0; k < rows; k++)
sum += coli[k] * colj[k];
for(k = 0; k < rows; k++)
colj[k] -= sum * coli[k];
}
sum = 0.f;
for(i = 0; i < rows; i++)
sum += colj[i] * colj[i];
invnorm = 1.f / sqrtf(sum);
for(i = 0; i < rows; i++)
colj[i] *= invnorm;
}
for(i = cols-1; i > -1; i--) {
colQ = Q+i*rows;
// matmult Q * X -> R
for(j = 0; j < cols; j++) {
colX = X+j*rows;
sum = 0.f;
for(k = 0; k < rows; k++)
sum += colQ[k] * colX[k];
R[i+j*cols] = sum;
}
sum = 0.f;
// compute the vector Q^t * Y -> B
for(j = 0; j < rows; j++)
sum += colQ[j] * Y[j];
// back substitution to find the x for Rx = B
for(j = cols-1; j > i; j--)
sum -= R[i+j*cols] * B[j];

B[i] = sum / R[i+i*cols];
}
}