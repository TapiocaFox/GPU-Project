#include "includes.h"
__global__ void get_entropy(int nbins, int nsamples, int nx, float * bin_scores, int pitch_bin_scores, float * entropies)
{
int
col_x = blockDim.x * blockIdx.x + threadIdx.x;

if(col_x >= nx)
return;

float
* in_col = bin_scores + col_x * pitch_bin_scores,
entropy = 0.f, prob, logp;

for(int i = 0; i < nbins; i++) {
prob = 0.f;
for(int j = 0; j < nsamples; j++)
prob += in_col[j * nbins + i];
prob /= (double) nsamples;

if(prob <= 0.f)
logp = 0.f;
else
logp = __log2f(prob);

entropy += prob * logp;
}
entropies[col_x] = -entropy;
}