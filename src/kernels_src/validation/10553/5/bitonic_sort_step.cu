#include "includes.h"
#define N 50
#define NewN 100

#define LifeN 500
#define numofthreads 512
int numofeles=0,capacity;

struct chromosome
{
long long weight=0, value=0;
bool chromo[100003];
};
chromosome chromoele[N],*cudaChromo,*cudaNewpopulation,newpopulation[NewN],res,x[2];
int weight[100001],value[100001],*devValue,*devWeight,*devnumeles;
__global__ void bitonic_sort_step(chromosome *cudanewpopulation, int j, int k)
{
unsigned int i, ixj; /* Sorting partners: i and ixj */
i = threadIdx.x + blockDim.x * blockIdx.x;
ixj = i^j;
printf("                    %d                        \n", i);

/* The threads with the lowest ids sort the array. */
if ((ixj) > i) {
if ((i&k) != 0) {
/* Sort ascending */
if (cudanewpopulation[i].value < cudanewpopulation[ixj].value) {
/* exchange(i,ixj); */
chromosome temp = cudanewpopulation[i];
cudanewpopulation[i] = cudanewpopulation[ixj];
cudanewpopulation[ixj] = temp;

}
}
}

}