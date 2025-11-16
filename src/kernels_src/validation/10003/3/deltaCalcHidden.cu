#include "includes.h"
#pragma comment(lib,"cublas.lib")




using namespace std;

//==============================Function Prototypes================================
double getRand();

__global__ void deltaCalcHidden(float *Activation,float *delta){
int n = blockIdx.x*blockDim.x + threadIdx.x;
delta[n] = delta[n] * (1 / (1 + exp(-Activation[n]))*(1 - 1 / (1 + exp(-Activation[n]))));
}