#include "includes.h"





using namespace std;

#define MEDIAN_DIMENSION  3 // For matrix of 3 x 3. We can Use 5 x 5 , 7 x 7 , 9 x 9......
#define MEDIAN_LENGTH 9   // Shoul be  MEDIAN_DIMENSION x MEDIAN_DIMENSION = 3 x 3

#define BLOCK_WIDTH 16  // Should be 8 If matrix is of larger then of 5 x 5 elese error occur as " uses too much shared data "  at surround[BLOCK_WIDTH*BLOCK_HEIGHT][MEDIAN_LENGTH]
#define BLOCK_HEIGHT 16// Should be 8 If matrix is of larger then of 5 x 5 elese error occur as " uses too much shared data "  at surround[BLOCK_WIDTH*BLOCK_HEIGHT][MEDIAN_LENGTH]


__global__ void MedianFilter_gpu(unsigned short *Device_ImageData, int Image_Width, int Image_Height) {

__shared__ unsigned short surround[BLOCK_WIDTH*BLOCK_HEIGHT][MEDIAN_LENGTH];

int iterator;
const int Half_Of_MEDIAN_LENGTH = (MEDIAN_LENGTH / 2) + 1;
int StartPoint = MEDIAN_DIMENSION / 2;
int EndPoint = StartPoint + 1;

const int x = blockDim.x * blockIdx.x + threadIdx.x;
const int y = blockDim.y * blockIdx.y + threadIdx.y;

const int tid = threadIdx.y*blockDim.y + threadIdx.x;

if (x >= Image_Width || y >= Image_Height)
return;

//Fill surround with pixel value of Image in Matrix Pettern of MEDIAN_DIMENSION x MEDIAN_DIMENSION
if (x == 0 || x == Image_Width - StartPoint || y == 0
|| y == Image_Height - StartPoint) {
}
else {
iterator = 0;
for (int r = x - StartPoint; r < x + (EndPoint); r++) {
for (int c = y - StartPoint; c < y + (EndPoint); c++) {
surround[tid][iterator] = *(Device_ImageData + (c*Image_Width) + r);
iterator++;
}
}
//Sort the Surround Array to Find Median. Use Bubble Short  if Matrix oF 3 x 3 Matrix
//You can use Insertion commented below to Short Bigger Dimension Matrix

////      bubble short //

for (int i = 0; i<Half_Of_MEDIAN_LENGTH; ++i)
{
// Find position of minimum element
int min = i;
for (int l = i + 1; l<MEDIAN_LENGTH; ++l)
if (surround[tid][l] <surround[tid][min])
min = l;
// Put found minimum element in its place
unsigned short  temp = surround[tid][i];
surround[tid][i] = surround[tid][min];
surround[tid][min] = temp;
}//bubble short  end

//////insertion sort start   //

/*int t,j,i;
for ( i = 1 ; i< MEDIAN_LENGTH ; i++) {
j = i;
while ( j > 0 && surround[tid][j] < surround[tid][j-1]) {
t= surround[tid][j];
surround[tid][j]= surround[tid][j-1];
surround[tid][j-1] = t;
j--;
}
}*/

////insertion sort end



*(Device_ImageData + (y*Image_Width) + x) = surround[tid][Half_Of_MEDIAN_LENGTH - 1];   // it will give value of surround[tid][4] as Median Value if use 3 x 3 matrix
__syncthreads();
}
}