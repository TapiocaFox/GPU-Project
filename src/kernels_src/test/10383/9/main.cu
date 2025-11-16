#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include "remove_redness_from_coordinates.cu"
#include<chrono>
#include<iostream>
using namespace std;
using namespace std::chrono;
int blocks_[20][2] = {{8,8},{16,16},{24,24},{32,32},{1,64},{1,128},{1,192},{1,256},{1,320},{1,384},{1,448},{1,512},{1,576},{1,640},{1,704},{1,768},{1,832},{1,896},{1,960},{1,1024}};
int matrices_[7][2] = {{240,240},{496,496},{784,784},{1016,1016},{1232,1232},{1680,1680},{2024,2024}};
int main(int argc, char **argv) {
cudaSetDevice(0);
char* p;int matrix_len=strtol(argv[1], &p, 10);
for(int matrix_looper=0;matrix_looper<matrix_len;matrix_looper++){
for(int block_looper=0;block_looper<20;block_looper++){
int XSIZE=matrices_[matrix_looper][0],YSIZE=matrices_[matrix_looper][1],BLOCKX=blocks_[block_looper][0],BLOCKY=blocks_[block_looper][1];
const unsigned int *d_coordinates = NULL;
cudaMalloc(&d_coordinates, XSIZE*YSIZE);
unsigned char *d_r = NULL;
cudaMalloc(&d_r, XSIZE*YSIZE);
unsigned char *d_b = NULL;
cudaMalloc(&d_b, XSIZE*YSIZE);
unsigned char *d_g = NULL;
cudaMalloc(&d_g, XSIZE*YSIZE);
unsigned char *d_r_output = NULL;
cudaMalloc(&d_r_output, XSIZE*YSIZE);
int num_coordinates = 1;
int num_pixels_y = 1;
int num_pixels_x = 1;
int template_half_height = YSIZE;
int template_half_width = XSIZE;
int iXSIZE= XSIZE;
int iYSIZE= YSIZE;
while(iXSIZE%BLOCKX!=0)
{
iXSIZE++;
}
while(iYSIZE%BLOCKY!=0)
{
iYSIZE++;
}
dim3 gridBlock(iXSIZE/BLOCKX, iYSIZE/BLOCKY);
dim3 threadBlock(BLOCKX, BLOCKY);
cudaFree(0);
remove_redness_from_coordinates<<<gridBlock,threadBlock>>>(d_coordinates,d_r,d_b,d_g,d_r_output,num_coordinates,num_pixels_y,num_pixels_x,template_half_height,template_half_width);
cudaDeviceSynchronize();
for (int loop_counter = 0; loop_counter < 10; ++loop_counter) {
remove_redness_from_coordinates<<<gridBlock,threadBlock>>>(d_coordinates,d_r,d_b,d_g,d_r_output,num_coordinates,num_pixels_y,num_pixels_x,template_half_height,template_half_width);
}
auto start = steady_clock::now();
for (int loop_counter = 0; loop_counter < 1000; loop_counter++) {
remove_redness_from_coordinates<<<gridBlock,threadBlock>>>(d_coordinates,d_r,d_b,d_g,d_r_output,num_coordinates,num_pixels_y,num_pixels_x,template_half_height,template_half_width);
}
auto end = steady_clock::now();
auto usecs = duration_cast<duration<float, microseconds::period> >(end - start);
cout <<'['<<usecs.count()<<','<<'('<<BLOCKX<<','<<BLOCKY<<')' << ','<<'('<<XSIZE<<','<<YSIZE<<')'<<']' << endl;
}
}}