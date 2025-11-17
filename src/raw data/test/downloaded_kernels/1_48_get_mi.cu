#include "includes.h"
__global__ void get_mi(int nbins, int nsamples, int nx, float * x_bin_scores, int pitch_x_bin_scores, float * entropies_x, int ny, float * y_bin_scores, int pitch_y_bin_scores, float * entropies_y, float * mis, int pitch_mis)
{
int
col_x = blockDim.x * blockIdx.x + threadIdx.x,
col_y = blockDim.y * blockIdx.y + threadIdx.y;

if((col_x >= nx) || (col_y >= ny))
return;

float
prob, logp, mi = 0.f,
* x_bins = x_bin_scores + col_x * pitch_x_bin_scores,
* y_bins = y_bin_scores + col_y * pitch_y_bin_scores;

// calculate joint entropy
for(int i = 0; i < nbins; i++) {
for(int j = 0; j < nbins; j++) {
prob = 0.f;
for(int k = 0; k < nsamples; k++)
prob += x_bins[k * nbins + i] * y_bins[k * nbins + j];
prob /= (float)nsamples;

if(prob <= 0.f)
logp = 0.f;
else
logp = __log2f(prob);

mi += prob * logp;
}
}

// calculate mi from entropies
mi += entropies_x[col_x] + entropies_y[col_y];
(mis + col_y * pitch_mis)[col_x] = mi;
}