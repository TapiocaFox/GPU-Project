#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <iostream>
#include <vector_types.h>
using namespace std;
using namespace std::chrono;

__global__ void remove_redness_from_coordinates( const unsigned int*  d_coordinates, unsigned char* d_r, unsigned char* d_b, unsigned char* d_g, unsigned char* d_r_output, int    num_coordinates, int    num_pixels_y, int    num_pixels_x, int    template_half_height, int    template_half_width )
{
int ny              = num_pixels_y;
int nx              = num_pixels_x;
int global_index_1d = ( blockIdx.x * blockDim.x ) + threadIdx.x;

int imgSize = num_pixels_x * num_pixels_y;

if ( global_index_1d < num_coordinates )
{
unsigned int image_index_1d = d_coordinates[ imgSize - global_index_1d - 1 ];
ushort2 image_index_2d = make_ushort2(image_index_1d % num_pixels_x, image_index_1d / num_pixels_x);

for ( int y = image_index_2d.y - template_half_height; y <= image_index_2d.y + template_half_height; y++ )
{
for ( int x = image_index_2d.x - template_half_width; x <= image_index_2d.x + template_half_width; x++ )
{
int2 image_offset_index_2d         = make_int2( x, y );
int2 image_offset_index_2d_clamped = make_int2( min( nx - 1, max( 0, image_offset_index_2d.x ) ), min( ny - 1, max( 0, image_offset_index_2d.y ) ) );
int  image_offset_index_1d_clamped = ( nx * image_offset_index_2d_clamped.y ) + image_offset_index_2d_clamped.x;

unsigned char g_value = d_g[ image_offset_index_1d_clamped ];
unsigned char b_value = d_b[ image_offset_index_1d_clamped ];

unsigned int gb_average = ( g_value + b_value ) / 2;

d_r_output[ image_offset_index_1d_clamped ] = (unsigned char)gb_average;
}
}

}
}

int main(int argc, char **argv) {
    cudaSetDevice(0);
    
    // Grid: (5, 5), Block: (32, 32)
    dim3 gridBlock(5, 5);
    dim3 threadBlock(32, 32);
    
    // Allocate device memory
    int num_coordinates = 100;
    int num_pixels_y = 160;
    int num_pixels_x = 160;
    int template_half_height = 5;
    int template_half_width = 5;
    int imgSize = num_pixels_x * num_pixels_y;
    unsigned int *d_coordinates = NULL;
    unsigned char *d_r = NULL;
    unsigned char *d_b = NULL;
    unsigned char *d_g = NULL;
    unsigned char *d_r_output = NULL;
    cudaMalloc(&d_coordinates, imgSize * sizeof(unsigned int));
    cudaMalloc(&d_r, imgSize * sizeof(unsigned char));
    cudaMalloc(&d_b, imgSize * sizeof(unsigned char));
    cudaMalloc(&d_g, imgSize * sizeof(unsigned char));
    cudaMalloc(&d_r_output, imgSize * sizeof(unsigned char));
    
    // Warmup
    cudaFree(0);
    remove_redness_from_coordinates<<<gridBlock, threadBlock>>>(d_coordinates, d_r, d_b, d_g, d_r_output, num_coordinates, num_pixels_y, num_pixels_x, template_half_height, template_half_width);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < 10; ++i) {
        remove_redness_from_coordinates<<<gridBlock, threadBlock>>>(d_coordinates, d_r, d_b, d_g, d_r_output, num_coordinates, num_pixels_y, num_pixels_x, template_half_height, template_half_width);
    }
    cudaDeviceSynchronize();
    
    // Measure execution time
    auto start = steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        remove_redness_from_coordinates<<<gridBlock, threadBlock>>>(d_coordinates, d_r, d_b, d_g, d_r_output, num_coordinates, num_pixels_y, num_pixels_x, template_half_height, template_half_width);
    }
    cudaDeviceSynchronize();
    auto end = steady_clock::now();
    
    auto usecs = duration_cast<duration<float, microseconds::period>>(end - start);
    float avg_time_us = usecs.count() / 1000.0f;
    float avg_time_ms = avg_time_us / 1000.0f;
    
    cout << "[" << avg_time_us << "," << avg_time_ms << "]" << endl;
    
    cudaFree(d_coordinates);
    cudaFree(d_r);
    cudaFree(d_b);
    cudaFree(d_g);
    cudaFree(d_r_output);
    
    return 0;
}
