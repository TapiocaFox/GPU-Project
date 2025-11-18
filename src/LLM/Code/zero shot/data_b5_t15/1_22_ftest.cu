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

__global__ void ftest(int diagFlag, int p, int rows, int colsx, int colsy, int rCols, int unrCols, float * obs, int obsDim, float * rCoeffs, int rCoeffsDim, float * unrCoeffs, int unrCoeffsDim, float * rdata, int rdataDim, float * unrdata, int unrdataDim, float * dfStats) // float * dpValues)
{
int
j = blockIdx.x * THREADSPERDIM + threadIdx.x,
i = blockIdx.y * THREADSPERDIM + threadIdx.y,
idx = i*colsx + j, k, m;
float
kobs, fp = (float) p, frows = (float) rows,
rSsq, unrSsq,
rEst, unrEst,
score = 0.f,
* tObs, * tRCoeffs, * tUnrCoeffs,
* tRdata, * tUnrdata;

if((i >= colsy) || (j >= colsx)) return;
if((!diagFlag) && (i == j)) {
dfStats[idx] = 0.f;
// dpValues[idx] = 0.f;
return;
}

tObs = obs + (i*colsx+j)*obsDim;

tRCoeffs = rCoeffs + i*rCoeffsDim;
tRdata = rdata + i*rdataDim;

tUnrCoeffs = unrCoeffs + (i*colsx+j)*unrCoeffsDim;
tUnrdata = unrdata + (i*colsx+j)*unrdataDim;

rSsq = unrSsq = 0.f;
for(k = 0; k < rows; k++) {
unrEst = rEst = 0.f;
kobs = tObs[k];
for(m = 0; m < rCols; m++)
rEst += tRCoeffs[m] * tRdata[k+m*rows];
for(m = 0; m < unrCols; m++)
unrEst += tUnrCoeffs[m] * tUnrdata[k+m*rows];
rSsq   += (kobs - rEst) * (kobs - rEst);
unrSsq += (kobs - unrEst) * (kobs - unrEst);

}
score = ((rSsq - unrSsq)*(frows-2.f*fp-1.f)) / (fp*unrSsq);

if(!isfinite(score))
score = 0.f;

dfStats[idx] = score;
}

int main(int argc, char **argv) {
    cudaSetDevice(0);
    
    // Grid: (5, 5), Block: (32, 32)
    dim3 gridBlock(5, 5);
    dim3 threadBlock(32, 32);
    
    // Allocate device memory
    int diagFlag = 1;
    int p = 5;
    int rows = 100;
    int colsx = 5;
    int colsy = 5;
    int rCols = 5;
    int unrCols = 10;
    int obsDim = rows;
    int rCoeffsDim = rCols;
    int unrCoeffsDim = unrCols;
    int rdataDim = rows * rCols;
    int unrdataDim = rows * unrCols;
    
    float *d_obs = NULL;
    float *d_rCoeffs = NULL;
    float *d_unrCoeffs = NULL;
    float *d_rdata = NULL;
    float *d_unrdata = NULL;
    float *d_dfStats = NULL;
    
    cudaMalloc(&d_obs, colsy * colsx * obsDim * sizeof(float));
    cudaMalloc(&d_rCoeffs, colsy * rCoeffsDim * sizeof(float));
    cudaMalloc(&d_unrCoeffs, colsy * colsx * unrCoeffsDim * sizeof(float));
    cudaMalloc(&d_rdata, colsy * rdataDim * sizeof(float));
    cudaMalloc(&d_unrdata, colsy * colsx * unrdataDim * sizeof(float));
    cudaMalloc(&d_dfStats, colsy * colsx * sizeof(float));
    
    // Warmup
    cudaFree(0);
    ftest<<<gridBlock, threadBlock>>>(diagFlag, p, rows, colsx, colsy, rCols, unrCols, d_obs, obsDim, d_rCoeffs, rCoeffsDim, d_unrCoeffs, unrCoeffsDim, d_rdata, rdataDim, d_unrdata, unrdataDim, d_dfStats);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < 10; ++i) {
        ftest<<<gridBlock, threadBlock>>>(diagFlag, p, rows, colsx, colsy, rCols, unrCols, d_obs, obsDim, d_rCoeffs, rCoeffsDim, d_unrCoeffs, unrCoeffsDim, d_rdata, rdataDim, d_unrdata, unrdataDim, d_dfStats);
    }
    cudaDeviceSynchronize();
    
    // Measure execution time
    auto start = steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        ftest<<<gridBlock, threadBlock>>>(diagFlag, p, rows, colsx, colsy, rCols, unrCols, d_obs, obsDim, d_rCoeffs, rCoeffsDim, d_unrCoeffs, unrCoeffsDim, d_rdata, rdataDim, d_unrdata, unrdataDim, d_dfStats);
    }
    cudaDeviceSynchronize();
    auto end = steady_clock::now();
    
    auto usecs = duration_cast<duration<float, microseconds::period>>(end - start);
    float avg_time_us = usecs.count() / 1000.0f;
    float avg_time_ms = avg_time_us / 1000.0f;
    
    cout << "[" << avg_time_us << "," << avg_time_ms << "]" << endl;
    
    cudaFree(d_obs);
    cudaFree(d_rCoeffs);
    cudaFree(d_unrCoeffs);
    cudaFree(d_rdata);
    cudaFree(d_unrdata);
    cudaFree(d_dfStats);
    
    return 0;
}
