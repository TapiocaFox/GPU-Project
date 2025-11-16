#include "includes.h"
__global__ void gpuSum(int *prices,int *sumpricesout,int days,int seconds,int N)
{
int currentday = blockIdx.x*blockDim.x + threadIdx.x;
if(currentday<days)
{
int start = currentday * seconds;
int end = start+seconds;

int totprice=0;
for(int j=start;j<end;++j)
totprice+=prices[j];

sumpricesout[currentday] = totprice;
}
}