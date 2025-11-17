#include "includes.h"
#pragma comment(lib,"cublas.lib")




using namespace std;

//==============================Function Prototypes================================
double getRand();

__global__ void deltaCalcOutput(float *OutActivation, float *Outputdelta, float *targets){
int n = blockIdx.x*blockDim.x + threadIdx.x;
Outputdelta[n] = (targets[n] - OutActivation[n]) * (1 / (1 + exp(-OutActivation[n]))*(1 - 1 / (1 + exp(-OutActivation[n]))));
}