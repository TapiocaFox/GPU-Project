#include "includes.h"
/* Non-local Means -- Cuda implementation
*
* Graikos Alexandros - 8128
*/

// Padded the mirrored pixels around our image
#define image(i,j) image[((i) + vertical_radius)*(n+2*horizontal_radius) + ((j)+horizontal_radius)]

// Weight function between pixel (i,j) <-> (k,l)
#define weights(i,j,k,l) weights[((i)*n + (j))*m*n + ((k)*n + (l))]

// Gaussian filter matrix
#define gaussian_matrix(i,j) gaussian_matrix[(i)*patchSize[1] + (j)]

// Shared memory patch matrix
#define shared_memory_patch(patch,i,j) patch[((i)+horizontal_radius)*(blockDim.y+2*horizontal_radius) + ((j)+vertical_radius)]

// Filter output
#define filtered_image(i,j) filtered_image[(i)*n + j]


// Start of shared memory
extern __shared__  float gaussian_matrix[];

/* int version of pow */
__device__ float compare_patches (int m, int n, float *patch1, float *patch2, int local_x, int local_y, int target_x, int target_y, float patch_sigma, int *patchSize , float filter_sigma) {

int vertical_radius = (patchSize[0] - 1) / 2;
int horizontal_radius = (patchSize[1] - 1) / 2;

int k, l;
float euclidian_distance = 0;

for (k=-vertical_radius; k<=vertical_radius; k++) {
for (l=-horizontal_radius; l<=horizontal_radius; l++) {

// Filter value assigned to patch (x,y)
float gaussian_filter = gaussian_matrix(k+vertical_radius,l+horizontal_radius);

// Compute euclidian distance (squared) between local_pixel
// and target_pixel
euclidian_distance += gaussian_filter*gaussian_filter*
powf( shared_memory_patch(patch1, local_x+k, local_y+l) -
shared_memory_patch(patch2, target_x+k, target_y+l), 2);
}

}

return expf( -euclidian_distance / (filter_sigma));
}
__device__ float compare_patches (int m, int n, float *patch1, float *patch2, int pixel_1_x, int pixel_1_y, int pixel_2_x, int pixel_2_y, float patch_sigma, int *patchSize , float filter_sigma);


/* Loads image patch(x,y) into shared memory matrix (patch) */
__device__ void load_patch(float *patch, const float *image, int m, int n,int patch_x, int patch_y, int *patchSize) {

int i = patch_x * blockDim.x + threadIdx.x;
int j = patch_y * blockDim.y + threadIdx.y;

int vertical_radius = (patchSize[0] - 1) / 2;
int horizontal_radius = (patchSize[1] -1 ) / 2;

// Copy thread assigned pixel to shared memory
shared_memory_patch(patch, threadIdx.x, threadIdx.y) = image(i,j);

// Copy the mirrored (padded) pixels into shared memory

// Left border
if (threadIdx.x < vertical_radius) {
shared_memory_patch(patch, -threadIdx.x-1, threadIdx.y) = image(i-2*threadIdx.x-1,j);

// Upper left diagonal
if (threadIdx.y < horizontal_radius) {
shared_memory_patch(patch, -threadIdx.x-1, -threadIdx.y-1) = image(i-2*threadIdx.x-1, j-2*threadIdx.y-1);
}
}

// Upper border
if (threadIdx.y < horizontal_radius) {
shared_memory_patch(patch, threadIdx.x, -threadIdx.y-1) = image(i,j-2*threadIdx.y-1);
}

// Bottom border
if (threadIdx.x >= blockDim.x - vertical_radius) {
shared_memory_patch(patch, 2*blockDim.x - (threadIdx.x+1), threadIdx.y) = image(i+1+2*(blockDim.x-threadIdx.x-1), j);

// Botoom left diagonal
if (threadIdx.y < horizontal_radius) {
shared_memory_patch(patch, 2*blockDim.x-(threadIdx.x+1), -threadIdx.y-1) =
image(i+1+2*(blockDim.x-threadIdx.x-1),j-2*threadIdx.y-1);
}
}

// Right border
if (threadIdx.y >= blockDim.y - horizontal_radius) {
shared_memory_patch(patch, threadIdx.x, 2*blockDim.y - (threadIdx.y+1)) = image(i,j+1+2*(blockDim.y-threadIdx.y-1));

// Upper right diagonal
if (threadIdx.x < vertical_radius) {
shared_memory_patch(patch, -threadIdx.x-1, 2*blockDim.y-(threadIdx.y+1)) =
image(i-2*threadIdx.x-1, j+1+2*(blockDim.y-threadIdx.y-1));
}
}

// Bottom right diagonal
if (threadIdx.x >= (blockDim.x - vertical_radius) && threadIdx.y >= (blockDim.y - horizontal_radius)) {
shared_memory_patch(patch, 2*blockDim.x-(threadIdx.x+1),2*blockDim.y-(threadIdx.y+1)) =
image(i+1+2*(blockDim.x-threadIdx.x-1), j+1+2*(blockDim.y-threadIdx.y-1));
}

__syncthreads();

}
__device__ int int_pow(int a, int b) {
int i, prod = 1;
for (i=0; i<b; i++) {
prod *= a;
}
return prod;
}
__global__ void nlm_kernel (float const *image, float *filtered_image,int m, int n, float patch_sigma, int *patchSize, float filter_sigma) {

int vertical_radius = (patchSize[0] - 1) / 2;
int horizontal_radius = (patchSize[1] - 1) / 2;

// Compute gaussian filter
if (threadIdx.x < patchSize[0] && threadIdx.y < patchSize[1]) {
gaussian_matrix(threadIdx.x, threadIdx.y) = exp(-(int_pow(threadIdx.x-horizontal_radius,2) +
int_pow(threadIdx.y-vertical_radius,2)) / (2*(patch_sigma*patch_sigma)));
}
__syncthreads();

// Pixel coordinates assigned to thread
int i = blockIdx.x * blockDim.x + threadIdx.x;
int j = blockIdx.y * blockDim.y + threadIdx.y;

// Addresses of shared memory matrices
float *target_patch = gaussian_matrix + patchSize[0]*patchSize[1];
float *local_patch = target_patch + (blockDim.x+2*vertical_radius)*(blockDim.y+2*horizontal_radius);

// Load local image patch assigned to block to shared memory
load_patch(local_patch, image, m, n, blockIdx.x, blockIdx.y, patchSize);

int k, l;
int patch_x, patch_y;

// Z value for current pixel
float Z = 0.f;
float filtered_pixel = 0.f;

// Load each target_patch from the image to shared memory and calculate
// weight between pixels in local patch and target patch
for (patch_x=0; patch_x<gridDim.x; patch_x++) {
for (patch_y=0; patch_y<gridDim.y; patch_y++) {
// Load image patch (x,y)
// !Do not load already present local_patch!
if (patch_x != blockIdx.x || patch_y != blockIdx.y) {
load_patch(target_patch, image, m, n, patch_x, patch_y, patchSize);
} else {
target_patch = local_patch;
}

// Calculate weights
for (k=0; k<blockDim.x; k++) {
for (l=0; l<blockDim.y; l++) {
float weight;

// Weight between ours pixel and target pixel's patch
// Consider case where local_patch equals target_patch
weight = compare_patches(m, n, local_patch, target_patch, threadIdx.x, threadIdx.y,
k, l, patch_sigma, patchSize, filter_sigma);

Z += weight;
// Add weight*pixel_value
filtered_pixel += weight*shared_memory_patch(target_patch,k,l);
}
}

// If we used local_patch for comparison reset target patch pointer
if (patch_x == blockIdx.x && patch_y == blockIdx.y) {
target_patch = gaussian_matrix + patchSize[0]*patchSize[1];
}

// Sync threads to load next patch
__syncthreads();
}
}

// Divide by Z to normalize weights
filtered_image(i,j) = filtered_pixel / Z;

return;
}