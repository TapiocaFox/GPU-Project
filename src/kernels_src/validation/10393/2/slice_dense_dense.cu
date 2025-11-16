#include "includes.h"
/*
* Licensed to the Apache Software Foundation (ASF) under one
* or more contributor license agreements.  See the NOTICE file
* distributed with this work for additional information
* regarding copyright ownership.  The ASF licenses this file
* to you under the Apache License, Version 2.0 (the
* "License"); you may not use this file except in compliance
* with the License.  You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*/

/**********************************
When updating a kernel or adding a new one,
please compile the ptx file and commit it:
nvcc -ptx -arch=sm_30 SystemML.cu
***********************************/


/**
* Performs a slice operation where the input matrix is sparse and the output matrix is dense.
* This function avoids unnecessary sparse to dense conversion of the input matrix.
* Parallelization: rows of output matrix.
*
* @params inVal input val pointer
* @params inRowPtr input row pointer
* @params colInd input col index pointer
* @params ret dense output pointer
* @param rl row lower
* @param ru row upper
* @param cl column lower
* @param cu column upper
* @param retClen number of columns of output matrix
*/
extern "C"

/**
* Performs a slice operation where the input matrix is sparse and the output matrix is dense.
* This function avoids unnecessary sparse to dense conversion of the input matrix.
* Parallelization: subset of number of non-zeroes of input matrix.
*
* @params inVal input val pointer
* @params inRowPtr input row pointer
* @params colInd input col index pointer
* @params ret dense output pointer
* @param rl row lower
* @param ru row upper
* @param cl column lower
* @param cu column upper
* @param retClen number of columns of output matrix
*/
extern "C"

/**
* Performs a slice operation where the input matrix is dense and the output matrix is dense.
*
* @params in dense input pointer
* @params ret dense output pointer
* @param rl row lower
* @param ru row upper
* @param cl column lower
* @param cu column upper
* @param inClen number of columns of input matrix
* @param retRlen number of rows of output matrix
* @param retClen number of columns of output matrix
*/
extern "C"


/**
* Does a copy of upper to lower triangle of the given matrix
* @param ret the input and output array allocated on the GPU
* @param dim the number of rows of the square matrix ret
* @param N total number of elements of the matrix
*/
extern "C"

extern "C"
__global__ void slice_dense_dense(double* in, double* ret, int rl, int ru, int cl, int cu, int inClen, int retRlen, int retClen) {
int tid = blockIdx.x * blockDim.x + threadIdx.x;
int ix = tid / retClen;
int iy = tid % retClen;
if(ix < retRlen && iy < retClen) {
int inIndex = (ix + rl)*inClen + cl + iy;
ret[tid] = in[inIndex];
}
}