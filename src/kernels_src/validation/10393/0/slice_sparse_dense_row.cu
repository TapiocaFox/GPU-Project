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
__global__ void slice_sparse_dense_row(double* inVal, int* inRowPtr, int* colInd, double* ret, int rl, int ru, int cl, int cu, int retClen) {
int index = blockIdx.x * blockDim.x + threadIdx.x;
int rowIndex = index + rl;
if (rowIndex <= ru){
/*
* TODO: Alternative approach: use dynamic parallelism. We are skipping this for now to avoid
* the complexity of two-step separate compilation and linking process.
*
* extern "C"
* __global__ void slice_sparse_dense_row_helper(double* inVal, int* inRowPtr, int* colInd, double* ret,
*     int rl, int ru, int cl, int cu, int retClen, int start, int end, int index) {
*  int i = blockIdx.x * blockDim.x + threadIdx.x + start;
* 	// Only slice if the index falls into the given range
* 	if(i < end && cl <= colInd[i] && colInd[i] <= cu) {
* 		ret[ index*retClen + (colInd[i] - cl) ] = inVal[i];
* 	}
* }
*
* int size = inRowPtr[rowIndex+1] - inRowPtr[rowIndex];
* double numThreads = (double)min(size, MAX_NUM_THREADS_CHILD_KERNEL);
* slice_sparse_dense_row_helper<<< ceil(numThreads/ MAX_NUM_THREADS_CHILD_KERNEL), MAX_NUM_THREADS_CHILD_KERNEL>>>(inVal, inRowPtr, colInd, ret,
*			rl, ru, cl, cu, retClen, inRowPtr[rowIndex], inRowPtr[rowIndex+1], index);
*
* Two-step compilation and linking process in JCudaKernels's constructor:
* cuLinkAddFile(linkState, CUjitInputType.CU_JIT_INPUT_LIBRARY, "/usr/local/cuda/lib64/libcudadevrt.a", jitOptions);
*/
// Iterate over elements of the row 'rowIndex'.
for(int i = inRowPtr[rowIndex]; i < inRowPtr[rowIndex+1]; i++) {
// Only slice if the index falls into the given range
if(cl <= colInd[i] && colInd[i] <= cu) {
ret[ index*retClen + (colInd[i] - cl) ] = inVal[i];
}
}
}
}