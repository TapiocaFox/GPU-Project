#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;

__global__ void non_max_supp_kernel(unsigned char *data, unsigned char *out, unsigned char *theta, int rows, int cols) {

extern __shared__ int l_mem[];
int* l_data = l_mem;

// These variables are offset by one to avoid seg. fault errors
// As such, this kernel ignores the outside ring of pixels
const int L_SIZE = blockDim.x;
const int g_row = blockIdx.y * blockDim.y + threadIdx.y + 1;
const int g_col = blockIdx.x * blockDim.x + threadIdx.x + 1;
const int l_row = threadIdx.y + 1;
const int l_col = threadIdx.x + 1;

const int pos = g_row * cols + g_col;

// copy to l_data
l_data[l_row * (L_SIZE + 2) + l_col] = data[pos];

// top most row
if(l_row == 1) {
l_data[0 * (L_SIZE + 2) + l_col] = data[pos - cols];
// top left
if(l_col == 1)
l_data[0 * (L_SIZE + 2) + 0] = data[pos - cols - 1];

// top right
else if(l_col == L_SIZE)
l_data[0 * (L_SIZE + 2) + (L_SIZE + 1)] = data[pos - cols + 1];
}
// bottom most row
else if(l_row == L_SIZE) {
l_data[(L_SIZE + 1) * (L_SIZE + 2) + l_col] = data[pos + cols];
// bottom left
if(l_col == 1)
l_data[(L_SIZE + 1) * (L_SIZE + 2) + 0] = data[pos + cols - 1];

// bottom right
else if(l_col == L_SIZE)
l_data[(L_SIZE + 1) * (L_SIZE + 2) + (L_SIZE + 1)] = data[pos + cols + 1];
}

if(l_col == 1)
l_data[l_row * (L_SIZE + 2) + 0] = data[pos - 1];
else if(l_col == L_SIZE)
l_data[l_row * (L_SIZE + 2) + (L_SIZE + 1)] = data[pos + 1];

__syncthreads();

unsigned char my_magnitude = l_data[l_row * (L_SIZE + 2) + l_col];

// The following variables are used to address the matrices more easily
switch(theta[pos]) {
// A gradient angle of 0 degrees = an edge that is North/South
// Check neighbors to the East and West
case 0:
// supress me if my neighbor has larger magnitude
if(my_magnitude <= l_data[l_row * (L_SIZE + 2) + l_col + 1] || // east
my_magnitude <= l_data[l_row * (L_SIZE + 2) + l_col - 1]) // west
{
out[pos] = 0;
}
// otherwise, copy my value to the output buffer
else {
out[pos] = my_magnitude;
}
break;

// A gradient angle of 45 degrees = an edge that is NW/SE
// Check neighbors to the NE and SW
case 45:
// supress me if my neighbor has larger magnitude
if(my_magnitude <= l_data[(l_row - 1) * (L_SIZE + 2) + l_col + 1] || // north east
my_magnitude <= l_data[(l_row + 1) * (L_SIZE + 2) + l_col - 1]) // south west
{
out[pos] = 0;
}
// otherwise, copy my value to the output buffer
else {
out[pos] = my_magnitude;
}
break;

// A gradient angle of 90 degrees = an edge that is E/W
// Check neighbors to the North and South
case 90:
// supress me if my neighbor has larger magnitude
if(my_magnitude <= l_data[(l_row - 1) * (L_SIZE + 2) + l_col] || // north
my_magnitude <= l_data[(l_row + 1) * (L_SIZE + 2) + l_col]) // south
{
out[pos] = 0;
}
// otherwise, copy my value to the output buffer
else {
out[pos] = my_magnitude;
}
break;

// A gradient angle of 135 degrees = an edge that is NE/SW
// Check neighbors to the NW and SE
case 135:
// supress me if my neighbor has larger magnitude
if(my_magnitude <= l_data[(l_row - 1) * (L_SIZE + 2) + l_col - 1] || // north west
my_magnitude <= l_data[(l_row + 1) * (L_SIZE + 2) + l_col + 1]) // south east
{
out[pos] = 0;
}
// otherwise, copy my value to the output buffer
else {
out[pos] = my_magnitude;
}
break;

default: out[pos] = my_magnitude; break;
}
}

int main(int argc, char **argv) {
    cudaSetDevice(0);
    
    // Grid: (5, 5), Block: (32, 32)
    dim3 gridBlock(5, 5);
    dim3 threadBlock(32, 32);
    
    // Allocate device memory
    int rows = 160;
    int cols = 160;
    int L_SIZE = 32;
    unsigned char *d_data = NULL;
    unsigned char *d_out = NULL;
    unsigned char *d_theta = NULL;
    cudaMalloc(&d_data, rows * cols * sizeof(unsigned char));
    cudaMalloc(&d_out, rows * cols * sizeof(unsigned char));
    cudaMalloc(&d_theta, rows * cols * sizeof(unsigned char));
    
    // Warmup
    cudaFree(0);
    non_max_supp_kernel<<<gridBlock, threadBlock, (L_SIZE + 2) * (L_SIZE + 2) * sizeof(int)>>>(d_data, d_out, d_theta, rows, cols);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < 10; ++i) {
        non_max_supp_kernel<<<gridBlock, threadBlock, (L_SIZE + 2) * (L_SIZE + 2) * sizeof(int)>>>(d_data, d_out, d_theta, rows, cols);
    }
    cudaDeviceSynchronize();
    
    // Measure execution time
    auto start = steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        non_max_supp_kernel<<<gridBlock, threadBlock, (L_SIZE + 2) * (L_SIZE + 2) * sizeof(int)>>>(d_data, d_out, d_theta, rows, cols);
    }
    cudaDeviceSynchronize();
    auto end = steady_clock::now();
    
    auto usecs = duration_cast<duration<float, microseconds::period>>(end - start);
    float avg_time_us = usecs.count() / 1000.0f;
    float avg_time_ms = avg_time_us / 1000.0f;
    
    cout << "[" << avg_time_us << "," << avg_time_ms << "]" << endl;
    
    cudaFree(d_data);
    cudaFree(d_out);
    cudaFree(d_theta);
    
    return 0;
}
