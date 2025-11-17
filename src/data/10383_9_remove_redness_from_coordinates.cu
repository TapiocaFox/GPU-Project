#include "includes.h"
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