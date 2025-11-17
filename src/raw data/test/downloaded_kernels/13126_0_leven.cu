#include "includes.h"
extern "C"

__global__ void leven(char* a, char* b, char* costs, int size) {
int i = blockIdx.x * blockDim.x + threadIdx.x;
if(i > 0 && i < size) {

costs[0] = i;
int nw = i - 1;
for(int j = 1; j <= size; j++) {
int firstMin = costs[j] < costs[j-1] ? costs[j] : costs[j-1];
// This line is hard to read due to the lack of min() function
int secondMin = 1 + firstMin < a[i - 1] == b[j - 1] ? nw : nw + 1 ? 1 + firstMin : a[i - 1] == b[j - 1] ? nw : nw + 1;
int cj = secondMin;
nw = costs[j];
costs[j] = cj;
}
}

}