#include "includes.h"
__global__ void hello()
{
printf("Hello world! blcokid: %d\nthreadid:%d\n", blockIdx.x, threadIdx.x);
}