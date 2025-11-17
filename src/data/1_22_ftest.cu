#include "includes.h"
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