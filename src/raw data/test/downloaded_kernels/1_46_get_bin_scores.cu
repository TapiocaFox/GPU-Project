#include "includes.h"
__device__ float do_fraction(float numer, float denom) {
float result = 0.f;

if((numer == denom) && (numer != 0.f))
result = 1.f;
else if(denom != 0.f)
result = numer / denom;

return result;
}
__global__ void get_bin_scores(int nbins, int order, int nknots, float * knots, int nsamples, int nx, float * x, int pitch_x, float * bins, int pitch_bins)
{
int
col_x = blockDim.x * blockIdx.x + threadIdx.x;

if(col_x >= nx)
return;

float
ld, rd, z,
term1, term2,
* in_col = x + col_x * pitch_x,
* bin_col = bins + col_x * pitch_bins;
int i0;

for(int k = 0; k < nsamples; k++, bin_col += nbins) {
z = in_col[k];
i0 = (int)floorf(z) + order - 1;
if(i0 >= nbins)
i0 = nbins - 1;

bin_col[i0] = 1.f;
for(int i = 2; i <= order; i++) {
for(int j = i0 - i + 1; j <= i0; j++) {
rd = do_fraction(knots[j + i] - z, knots[j + i] - knots[j + 1]);

if((j < 0) || (j >= nbins) || (j >= nknots) || (j + i - 1 < 0) || (j > nknots))
term1 = 0.f;
else {
ld = do_fraction(z - knots[j],
knots[j + i - 1] - knots[j]);
term1 = ld * bin_col[j];
}

if((j + 1 < 0) || (j + 1 >= nbins) || (j + 1 >= nknots) || (j + i < 0) || (j + i >= nknots))
term2 = 0.f;
else {
rd = do_fraction(knots[j + i] - z,
knots[j + i] - knots[j + 1]);
term2 = rd * bin_col[j + 1];
}
bin_col[j] = term1 + term2;
}
}
}
}