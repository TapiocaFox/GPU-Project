#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;

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

int main(int argc, char **argv) {
    cudaSetDevice(0);
    
    // Grid: (5, 5), Block: (32, 32)
    dim3 gridBlock(5, 5);
    dim3 threadBlock(32, 32);
    
    // Allocate device memory
    int countx = 5;
    int county = 5;
    int rows = 100;
    int cols = 10;
    int mXdim = rows;
    int vYdim = rows;
    int mQdim = rows * cols;
    int mRdim = cols * cols;
    int vectBdim = cols;
    
    float *d_mX = NULL;
    float *d_vY = NULL;
    float *d_mQ = NULL;
    float *d_mR = NULL;
    float *d_vectB = NULL;
    
    cudaMalloc(&d_mX, county * mXdim * sizeof(float));
    cudaMalloc(&d_vY, county * countx * vYdim * sizeof(float));
    cudaMalloc(&d_mQ, county * mQdim * sizeof(float));
    cudaMalloc(&d_mR, county * mRdim * sizeof(float));
    cudaMalloc(&d_vectB, county * vectBdim * sizeof(float));
    
    // Warmup
    cudaFree(0);
    getRestricted<<<gridBlock, threadBlock>>>(countx, county, rows, cols, d_mX, mXdim, d_vY, vYdim, d_mQ, mQdim, d_mR, mRdim, d_vectB, vectBdim);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < 10; ++i) {
        getRestricted<<<gridBlock, threadBlock>>>(countx, county, rows, cols, d_mX, mXdim, d_vY, vYdim, d_mQ, mQdim, d_mR, mRdim, d_vectB, vectBdim);
    }
    cudaDeviceSynchronize();
    
    // Measure execution time
    auto start = steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        getRestricted<<<gridBlock, threadBlock>>>(countx, county, rows, cols, d_mX, mXdim, d_vY, vYdim, d_mQ, mQdim, d_mR, mRdim, d_vectB, vectBdim);
    }
    cudaDeviceSynchronize();
    auto end = steady_clock::now();
    
    auto usecs = duration_cast<duration<float, microseconds::period>>(end - start);
    float avg_time_us = usecs.count() / 1000.0f;
    float avg_time_ms = avg_time_us / 1000.0f;
    
    cout << "[" << avg_time_us << "," << avg_time_ms << "]" << endl;
    
    cudaFree(d_mX);
    cudaFree(d_vY);
    cudaFree(d_mQ);
    cudaFree(d_mR);
    cudaFree(d_vectB);
    
    return 0;
}
