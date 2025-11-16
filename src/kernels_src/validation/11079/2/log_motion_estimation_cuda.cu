#include "includes.h"
__global__ void log_motion_estimation_cuda(uint8 *current, uint8 *previous, int *vectors_x, int *vectors_y, int *M_B, int *N_B, int *B, int *M, int *N) {
//obtain idx;
int id = blockIdx.x * blockDim.x + threadIdx.x;
if (id >= ((*M_B) * (*N_B))) return;
int x, y;

x = id / (*M_B);
y = id % (*M_B);


int dd;
for (dd = 4; dd > 1; dd--) {  //--> gives 4 3 2
int step = 0;
if (dd == 4) {
//d=4;
step = 4;
} else if (dd == 3) {
step = 2;
} else if (dd == 2) {
step = 1;
} else {
continue;
}

int min = 255 * (*B) * (*B);
int bestx, besty, i, j, k, l;
for (i = -step; i < step + 1; i += step)      /* For all candidate blocks */
for (j = -step; j < step + 1; j += step) {
int dist = 0;
for (k = 0; k < (*B); k++)        /* For all pixels in the block */
for (l = 0; l < (*B); l++) {
int tmp9 = vectors_x[x * (*M_B) + y];
int p1, p2;
p1 = current[((*B) * x + k) * (*M) + (*B) * y + l];
if (((*B) * x + tmp9 + i + k) < 0 || ((*B) * x + tmp9 + i + k) > ((*N) - 1) ||
((*B) * y + tmp9 + j + l) < 0 || ((*B) * y + tmp9 + j + l) > ((*M) - 1)) {
p2 = 0;
} else {
p2 = previous[((*B) * x + tmp9 + i + k) * (*M) + (*B) * y + tmp9 + j + l];
}

dist += abs(p1 - p2);
}
if (dist < min) {
min = dist;
bestx = i;
besty = j;
}
}

int at = x * (*M_B) + y;

vectors_x[at] += bestx;

vectors_y[at] += besty;

}
}