#include "includes.h"
__global__ void hello()
{
printf("Hello world! I'm thread %d\n", threadIdx.x);
}