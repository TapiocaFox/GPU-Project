#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;

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

*(Device_ImageData + (y*Image_Width) + x) = surround[tid][Half_Of_MEDIAN_LENGTH - 1];   // it will give value of surround[tid][4] as Median Value if use 3 x 3 matrix
__syncthreads();
}
}

int main(int argc, char **argv) {
    cudaSetDevice(0);
    
    // Grid: (5, 5), Block: (32, 32)
    dim3 gridBlock(5, 5);
    dim3 threadBlock(32, 32);
    
    // Allocate device memory
    int Image_Width = 160;
    int Image_Height = 160;
    unsigned short *d_Device_ImageData = NULL;
    cudaMalloc(&d_Device_ImageData, Image_Width * Image_Height * sizeof(unsigned short));
    
    // Warmup
    cudaFree(0);
    MedianFilter_gpu<<<gridBlock, threadBlock>>>(d_Device_ImageData, Image_Width, Image_Height);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < 10; ++i) {
        MedianFilter_gpu<<<gridBlock, threadBlock>>>(d_Device_ImageData, Image_Width, Image_Height);
    }
    cudaDeviceSynchronize();
    
    // Measure execution time
    auto start = steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        MedianFilter_gpu<<<gridBlock, threadBlock>>>(d_Device_ImageData, Image_Width, Image_Height);
    }
    cudaDeviceSynchronize();
    auto end = steady_clock::now();
    
    auto usecs = duration_cast<duration<float, microseconds::period>>(end - start);
    float avg_time_us = usecs.count() / 1000.0f;
    float avg_time_ms = avg_time_us / 1000.0f;
    
    cout << "[" << avg_time_us << "," << avg_time_ms << "]" << endl;
    
    cudaFree(d_Device_ImageData);
    
    return 0;
}
