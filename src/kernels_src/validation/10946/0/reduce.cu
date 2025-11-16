#include "includes.h"

using namespace std;
double iStart1, iStart2, iStart3a, iStart3b, iStart4a, iStart4b, iStart4c, iStart5;
double iElaps1=0, iElaps2=0, iElaps3a=0, iElaps3b=0, iElaps4=0, iElaps5=0;
// Hold configurations for Kmeans
struct Info {
int     numPoints;
int     dim;
int     numCentroids;
int     numRepeats;
int    *belongs;
float **points;
float **centroids;
int     thresholdLoops;
float   thresholdFraction;
int     threadPerBlock;
};

// ************* Utils ************* //

__global__ static void reduce(int *g_idata, int l1, int l2) {
extern __shared__ unsigned int sdata[];
unsigned int tid = threadIdx.x;

if (tid < l1) {
sdata[tid] = g_idata[tid];
} else {
sdata[tid] = 0;
}
__syncthreads();

// Parallel Reduction (l2 must be power of 2)
for (unsigned int s = l2 / 2; s > 0; s >>= 1) {
if (tid < s)     {
sdata[tid] += sdata[tid + s];
}
__syncthreads();
}

if (tid == 0) {
g_idata[0] = sdata[0];
}
}