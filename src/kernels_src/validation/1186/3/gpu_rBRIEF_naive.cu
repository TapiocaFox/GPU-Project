#include "includes.h"
__device__ __forceinline__ int conflict_free_index(int local_id, int real_idx)
{
return real_idx * 128 + local_id;
}
__global__ void gpu_rBRIEF_naive(float4* workload, int* output, int4* pattern, int4* train_bin_vec, int K, int P, int I)
{
// 0) Memory Setup
extern __shared__ float shared_patchBank[];
int* minVal;
int4   private_pattern[32];
int4   thisBuff, nextBuff;
int train_vec_x, train_vec_y, train_vec_z, train_vec_w;

// coordinate initialize in Private Registers
int coord[96] = { -0, -0, -0, -0, -0, 0, 1, 2, 3, 4,
-0, -4, -3, -2, -1, 0, 1, 2, 3, 4,
-0, -4, -3, -2, -1, 0, 1, 2, 3, 4,
-0, -4, -3, -2, -1, 0, 1, 2, 3, 4,
-0, -4, -3, -2, -1, 0, 1, 2, 3, 4,
-0, -4, -3, -2, -1, 0, 1, 2, 3, 4,
-0, -4, -3, -2, -1, 0, 1, 2, 3, 4,
-0, -4, -3, -2, -1, 0, 1, 2, 3, 4,
-0, -4, -3, -2, -1, 0, 1, 2, 3, 4,
-0, -4, -3, -2, -1, 0};

// 1) Setup thread ids
int local_id = threadIdx.x;
//int global_id = blockIdx.x * gridDim.x + local_id;

// 2) Load Sampling Pattern into Private Registers
#pragma unroll
for (int i = 0; i < 32; i++)
private_pattern[i] = pattern[i];

// 3) Load my patch into dedicated bank
for (int img = blockIdx.x; img < I; img+=gridDim.x) {

float4* patches;
int * res;
patches = &(workload[img * 3072]); // 128 patches of 24 float4 each
res     = &(output[img * 128]);    // 128 binary vector per image
#ifdef rBRIEFDEBUG
if (threadIdx.x == 0)
printf("Working on img: %d\n", img);
#endif

float4 thisNum;//= make_float4(0.0,0.0,0.0,0.0);
#pragma unroll
for (int i = 1; i < 24; i++) {
thisNum = patches[i * 128 + local_id];
shared_patchBank[conflict_free_index(local_id, i*4 + 0)] = thisNum.x;
shared_patchBank[conflict_free_index(local_id, i*4 + 1)] = thisNum.y;
shared_patchBank[conflict_free_index(local_id, i*4 + 2)] = thisNum.z;
shared_patchBank[conflict_free_index(local_id, i*4 + 3)] = thisNum.w;
}

#ifdef rBRIEFDEBUG
if (threadIdx.x == 0)
printf("Patch is loaded into private registers\n");
#endif

// 4) 1 thread works on 1 patch at a time
float m01 = 0.0;
float m10 = 0.0;
float intensity;
float theta;
#pragma unroll
for (int i = 5; i < 96; i++) {
intensity = shared_patchBank[conflict_free_index(local_id, i)];
m01       = __fmaf_rd(coord[i / 10], intensity, m01);
m10       = __fmaf_rd(coord[i], intensity, m10);
}

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
theta = atan2f(m01, m10); // BOTTLE NECK
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

#ifdef rBRIEFDEBUG
if (threadIdx.x == 0)
printf("m01: %f m10: %f theta: %f\n", m01, m10, theta);
#endif

// 5) Calculate the sin and cos of theta
float sin, cos;
sincosf(theta, &sin, &cos);
#ifdef rBRIEFDEBUG
if (threadIdx.x == 0)
printf("sin: %f cos: %f\n",sin, cos);
#endif

// 6) Sample the patch and return its binary vector
float Ia, Ib;
int ax, ay, bx, by;
unsigned int idxa, idxb;
int rotated_ax, rotated_ay, rotated_bx, rotated_by;
unsigned int binVector = 0;
int result;
#pragma unroll
for (int i = 0; i < 32; ++i) {
ax = private_pattern[i].x;
ay = private_pattern[i].y;
bx = private_pattern[i].z;
by = private_pattern[i].w;

rotated_ax = (int) (cos * ax - sin * ay);
rotated_ay = (int) (-10 * (sin * ay + cos * ay));
rotated_bx = (int) (cos * bx - sin * by);
rotated_by = (int) (-10 * (sin * by + cos * by));

idxa = __sad(rotated_ax, rotated_ay, 0) % 96;
idxb = __sad(rotated_bx, rotated_by, 0) % 96;

Ia = shared_patchBank[conflict_free_index(local_id, idxa)];
Ib = shared_patchBank[conflict_free_index(local_id, idxb)];

result = ((int) Ia > Ib) << i;
binVector |= result;
}

#ifdef rBRIEFDEBUG
if (threadIdx.x == 0) {
printf("%d", binVector);
printf("My Binary vector is: ");
while (binVector) {
if (binVector & 1)
printf("1");
else
printf("0");

binVector >>= 1;
}
printf("\n");
}
#endif

// 7) Preload binary vector from Global Memory and perform Hamming distance calculation
nextBuff = train_bin_vec[0];
float tmp = shared_patchBank[0]; // Borrow one value of shared memory
minVal = (int*) &(shared_patchBank[0]);
for (int i = 1; i < 32; i++) {
thisBuff = nextBuff;
nextBuff = train_bin_vec[i];

*minVal = 32;
train_vec_x = thisBuff.x;
train_vec_x ^= binVector;
train_vec_x = __popc(train_vec_x);
atomicMin(minVal, train_vec_x);
if(train_vec_x == *minVal)
res[i*4 + 0] = local_id;
__syncthreads();

*minVal = 32;
train_vec_y = thisBuff.y;
train_vec_y ^= binVector;
train_vec_y = __popc(train_vec_y);
atomicMin(minVal, train_vec_y);
if(train_vec_y == *minVal)
res[i*4 + 1] = local_id;
__syncthreads();

*minVal = 32;
train_vec_z = thisBuff.z;
train_vec_z ^= binVector;
train_vec_z = __popc(train_vec_z);
atomicMin(minVal, train_vec_z);
if(train_vec_z == *minVal)
res[i*4 + 2] = local_id;
__syncthreads();

*minVal = 32;
train_vec_w = thisBuff.w;
train_vec_w ^= binVector;
train_vec_w = __popc(train_vec_w);
atomicMin(minVal, train_vec_w);
if(train_vec_w == *minVal)
res[i*4 + 3] = local_id;
__syncthreads();
}
shared_patchBank[0] = tmp; // return the shared memory back to normal
}
}