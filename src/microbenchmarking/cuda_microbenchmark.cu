#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ITERATIONS 10000
#define WARP_SIZE 32

// Macro for CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// LATENCY MEASUREMENT KERNELS (Single Thread, No Parallelism)
// ============================================================================

// Kernel for measuring floating-point addition latency
// Following Figure 2: Counter loop with 256 instructions per iteration
__global__ void latency_fadd(float *input, float *output, int counter_iterations) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {  // Only one thread
        float result = input[0];
        clock_t total_time = 0;
        
        // Outer counter loop
        for (int counter = 0; counter < counter_iterations; counter++) {
            clock_t start = clock();
            
            // Inner loop: exactly 256 dependent instructions
            #pragma unroll 1
            for (int i = 0; i < 256; i++) {
                result = result + 1.0f;  // Dependent chain
            }
            
            clock_t end = clock();
            total_time += (end - start);
        }
        
        output[0] = result;
        output[1] = (float)total_time;
    }
}

// Kernel for measuring floating-point multiplication latency
// Following Figure 2: Counter loop with 256 instructions per iteration
__global__ void latency_fmul(float *input, float *output, int counter_iterations) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float result = input[0];
        clock_t total_time = 0;
        
        // Outer counter loop
        for (int counter = 0; counter < counter_iterations; counter++) {
            clock_t start = clock();
            
            // Inner loop: exactly 256 dependent instructions
            #pragma unroll 1
            for (int i = 0; i < 256; i++) {
                result = result * 1.000001f;  // Dependent chain
            }
            
            clock_t end = clock();
            total_time += (end - start);
        }
        
        output[0] = result;
        output[1] = (float)total_time;
    }
}

// Kernel for measuring FMA latency
// Following Figure 2: Counter loop with 256 instructions per iteration
__global__ void latency_fma(float *input, float *output, int counter_iterations) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float result = input[0];
        clock_t total_time = 0;
        
        // Outer counter loop
        for (int counter = 0; counter < counter_iterations; counter++) {
            clock_t start = clock();
            
            // Inner loop: exactly 256 dependent instructions
            #pragma unroll 1
            for (int i = 0; i < 256; i++) {
                result = result * 1.000001f + 0.5f;  // Dependent chain
            }
            
            clock_t end = clock();
            total_time += (end - start);
        }
        
        output[0] = result;
        output[1] = (float)total_time;
    }
}

// Kernel for measuring integer addition latency
// Following Figure 2: Counter loop with 256 instructions per iteration
__global__ void latency_iadd(int *input, int *output, int counter_iterations) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int result = input[0];
        clock_t total_time = 0;
        
        // Outer counter loop
        for (int counter = 0; counter < counter_iterations; counter++) {
            clock_t start = clock();
            
            // Inner loop: exactly 256 dependent instructions
            #pragma unroll 1
            for (int i = 0; i < 256; i++) {
                result = result + 1;  // Dependent chain
            }
            
            clock_t end = clock();
            total_time += (end - start);
        }
        
        output[0] = result;
        output[1] = (int)total_time;
    }
}

// Kernel for measuring integer multiplication latency
// Following Figure 2: Counter loop with 256 instructions per iteration
__global__ void latency_imul(int *input, int *output, int counter_iterations) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int result = input[0];
        clock_t total_time = 0;
        
        // Outer counter loop
        for (int counter = 0; counter < counter_iterations; counter++) {
            clock_t start = clock();
            
            // Inner loop: exactly 256 dependent instructions
            #pragma unroll 1
            for (int i = 0; i < 256; i++) {
                result = result * 3;  // Dependent chain
            }
            
            clock_t end = clock();
            total_time += (end - start);
        }
        
        output[0] = result;
        output[1] = (int)total_time;
    }
}

// Kernel for measuring global memory load latency
// Following Figure 2: Counter loop with 256 instructions per iteration
__global__ void latency_global_load(float *input, float *output, int counter_iterations) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        clock_t total_time = 0;
        float result = 0.0f;
        
        // Outer counter loop
        for (int counter = 0; counter < counter_iterations; counter++) {
            clock_t start = clock();
            
            // Inner loop: exactly 256 dependent loads
            #pragma unroll 1
            for (int i = 0; i < 256; i++) {
                result += input[i % 1024];  // Dependent loads
            }
            
            clock_t end = clock();
            total_time += (end - start);
        }
        
        output[0] = result;
        output[1] = (float)total_time;
    }
}

// Kernel for measuring shared memory load latency
// Following Figure 2: Counter loop with 256 instructions per iteration
__global__ void latency_shared_load(float *output, int counter_iterations) {
    __shared__ float shared_data[1024];
    
    if (threadIdx.x == 0) {
        // Initialize shared memory
        for (int i = 0; i < 1024; i++) {
            shared_data[i] = (float)i;
        }
        
        __syncthreads();
        
        clock_t total_time = 0;
        float result = 0.0f;
        
        // Outer counter loop
        for (int counter = 0; counter < counter_iterations; counter++) {
            clock_t start = clock();
            
            // Inner loop: exactly 256 dependent loads
            #pragma unroll 1
            for (int i = 0; i < 256; i++) {
                result += shared_data[i % 1024];
            }
            
            clock_t end = clock();
            total_time += (end - start);
        }
        
        output[0] = result;
        output[1] = (float)total_time;
    }
}

// ============================================================================
// THROUGHPUT MEASUREMENT KERNELS (Multiple Warps, High Occupancy)
// Different ILP levels to replicate Figure 3
// ============================================================================

// Kernel for measuring floating-point multiplication throughput with ILP=1
__global__ void throughput_fmul_ilp1(float *input, float *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    float a = input[tid];
    
    clock_t start = clock();
    
    // ILP=1: Only one variable, fully dependent operations
    #pragma unroll 1
    for (int i = 0; i < iterations; i++) {
        a = a * 1.000001f;  // Dependent chain
    }
    
    clock_t end = clock();
    
    output[tid] = a;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (float)(end - start);
    }
}

// Kernel for measuring floating-point multiplication throughput with ILP=2
__global__ void throughput_fmul_ilp2(float *input, float *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    float a = input[tid];
    float b = input[tid] + 1.0f;
    
    clock_t start = clock();
    
    // ILP=2: Two independent variables
    #pragma unroll 2
    for (int i = 0; i < iterations; i++) {
        a = a * 1.000001f;  // Independent from b
        b = b * 1.000002f;  // Independent from a
    }
    
    clock_t end = clock();
    
    output[tid] = a + b;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (float)(end - start);
    }
}

// Kernel for measuring floating-point multiplication throughput with ILP=3
__global__ void throughput_fmul_ilp3(float *input, float *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    float a = input[tid];
    float b = input[tid] + 1.0f;
    float c = input[tid] + 2.0f;
    
    clock_t start = clock();
    
    // ILP=3: Three independent variables
    #pragma unroll 3
    for (int i = 0; i < iterations; i++) {
        a = a * 1.000001f;  // Independent
        b = b * 1.000002f;  // Independent
        c = c * 1.000003f;  // Independent
    }
    
    clock_t end = clock();
    
    output[tid] = a + b + c;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (float)(end - start);
    }
}

// Original high-ILP kernel (for general throughput testing)
// Kernel for measuring floating-point multiplication throughput
__global__ void throughput_fmul(float *input, float *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    float a = input[tid];
    float b = 1.000001f;
    float c = 1.000002f;
    float d = 1.000003f;
    float e = 1.000004f;
    
    clock_t start = clock();
    
    // Independent operations to maximize ILP
    #pragma unroll 4
    for (int i = 0; i < iterations; i++) {
        a = a * b;
        c = c * d;
        e = e * b;
        a = a * c;
    }
    
    clock_t end = clock();
    
    output[tid] = a + c + e;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (float)(end - start);
    }
}

// Kernel for measuring FMA throughput
__global__ void throughput_fma(float *input, float *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    float a = input[tid];
    float b = 1.000001f;
    float c = 1.000002f;
    float d = 1.000003f;
    
    clock_t start = clock();
    
    #pragma unroll 4
    for (int i = 0; i < iterations; i++) {
        a = a * b + c;
        b = b * c + d;
        c = c * d + a;
        d = d * a + b;
    }
    
    clock_t end = clock();
    
    output[tid] = a + b + c + d;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (float)(end - start);
    }
}

// Kernel for measuring integer addition throughput
__global__ void throughput_iadd(int *input, int *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    int a = input[tid];
    int b = 1;
    int c = 2;
    int d = 3;
    
    clock_t start = clock();
    
    #pragma unroll 4
    for (int i = 0; i < iterations; i++) {
        a = a + b;
        c = c + d;
        a = a + c;
        b = b + d;
    }
    
    clock_t end = clock();
    
    output[tid] = a + b + c + d;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (int)(end - start);
    }
}

// ============================================================================
// PEAK WARPS MEASUREMENT KERNELS (Varying Number of Warps)
// ============================================================================

// Kernel for measuring peak warps for floating-point multiplication
__global__ void peakwarps_fmul(float *input, float *output, int iterations, int *timing) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    float a = input[tid];
    float b = 1.000001f;
    float c = 1.000002f;
    
    clock_t start = clock();
    
    #pragma unroll 4
    for (int i = 0; i < iterations; i++) {
        a = a * b;
        c = c * b;
        a = a * c;
    }
    
    clock_t end = clock();
    
    output[tid] = a + c;
    
    // Store timing for block 0 only
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *timing = (int)(end - start);
    }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

void measure_latency_fadd(float gpu_clock_mhz) {
    printf("\n=== Measuring FADD Latency ===\n");
    
    float *d_input, *d_output;
    float h_input = 1.0f;
    float h_output[2];
    
    const int counter_iterations = 100;  // Number of times to run 256 instructions
    const int instructions_per_counter = 256;  // Fixed at 256 as per Figure 2
    
    CUDA_CHECK(cudaMalloc(&d_input, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, 2 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, &h_input, sizeof(float), cudaMemcpyHostToDevice));
    
    // Warm-up
    latency_fadd<<<1, 1>>>(d_input, d_output, counter_iterations);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Actual measurement
    latency_fadd<<<1, 1>>>(d_input, d_output, counter_iterations);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_output, d_output, 2 * sizeof(float), cudaMemcpyDeviceToHost));
    
    float total_cycles = h_output[1];
    int total_instructions = counter_iterations * instructions_per_counter;
    float latency = total_cycles / total_instructions;
    
    printf("Counter iterations: %d\n", counter_iterations);
    printf("Instructions per counter: %d\n", instructions_per_counter);
    printf("Total instructions: %d\n", total_instructions);
    printf("Total cycles: %.2f\n", total_cycles);
    printf("Latency per instruction: %.2f cycles\n", latency);
    printf("Latency: %.2f ns\n", latency * 1000.0f / gpu_clock_mhz);
    
    cudaFree(d_input);
    cudaFree(d_output);
}

void measure_latency_fmul(float gpu_clock_mhz) {
    printf("\n=== Measuring FMUL Latency ===\n");
    
    float *d_input, *d_output;
    float h_input = 1.5f;
    float h_output[2];
    
    const int counter_iterations = 100;
    const int instructions_per_counter = 256;
    
    CUDA_CHECK(cudaMalloc(&d_input, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, 2 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, &h_input, sizeof(float), cudaMemcpyHostToDevice));
    
    latency_fmul<<<1, 1>>>(d_input, d_output, counter_iterations);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_output, d_output, 2 * sizeof(float), cudaMemcpyDeviceToHost));
    
    float total_cycles = h_output[1];
    int total_instructions = counter_iterations * instructions_per_counter;
    float latency = total_cycles / total_instructions;
    
    printf("Counter iterations: %d\n", counter_iterations);
    printf("Instructions per counter: %d\n", instructions_per_counter);
    printf("Total instructions: %d\n", total_instructions);
    printf("Total cycles: %.2f\n", total_cycles);
    printf("Latency per instruction: %.2f cycles\n", latency);
    printf("Latency: %.2f ns\n", latency * 1000.0f / gpu_clock_mhz);
    
    cudaFree(d_input);
    cudaFree(d_output);
}

void measure_latency_fma(float gpu_clock_mhz) {
    printf("\n=== Measuring FMA Latency ===\n");
    
    float *d_input, *d_output;
    float h_input = 1.5f;
    float h_output[2];
    
    const int counter_iterations = 100;
    const int instructions_per_counter = 256;
    
    CUDA_CHECK(cudaMalloc(&d_input, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, 2 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, &h_input, sizeof(float), cudaMemcpyHostToDevice));
    
    latency_fma<<<1, 1>>>(d_input, d_output, counter_iterations);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_output, d_output, 2 * sizeof(float), cudaMemcpyDeviceToHost));
    
    float total_cycles = h_output[1];
    int total_instructions = counter_iterations * instructions_per_counter;
    float latency = total_cycles / total_instructions;
    
    printf("Counter iterations: %d\n", counter_iterations);
    printf("Instructions per counter: %d\n", instructions_per_counter);
    printf("Total instructions: %d\n", total_instructions);
    printf("Total cycles: %.2f\n", total_cycles);
    printf("Latency per instruction: %.2f cycles\n", latency);
    printf("Latency: %.2f ns\n", latency * 1000.0f / gpu_clock_mhz);
    
    cudaFree(d_input);
    cudaFree(d_output);
}

void measure_latency_global_load(float gpu_clock_mhz) {
    printf("\n=== Measuring Global Load Latency ===\n");
    
    float *d_input, *d_output;
    float h_input[1024];
    float h_output[2];
    
    const int counter_iterations = 100;
    const int instructions_per_counter = 256;
    
    for (int i = 0; i < 1024; i++) h_input[i] = (float)i;
    
    CUDA_CHECK(cudaMalloc(&d_input, 1024 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, 2 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, 1024 * sizeof(float), cudaMemcpyHostToDevice));
    
    latency_global_load<<<1, 1>>>(d_input, d_output, counter_iterations);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_output, d_output, 2 * sizeof(float), cudaMemcpyDeviceToHost));
    
    float total_cycles = h_output[1];
    int total_instructions = counter_iterations * instructions_per_counter;
    float latency = total_cycles / total_instructions;
    
    printf("Counter iterations: %d\n", counter_iterations);
    printf("Instructions per counter: %d\n", instructions_per_counter);
    printf("Total instructions: %d\n", total_instructions);
    printf("Total cycles: %.2f\n", total_cycles);
    printf("Latency per instruction: %.2f cycles\n", latency);
    printf("Latency: %.2f ns\n", latency * 1000.0f / gpu_clock_mhz);
    
    cudaFree(d_input);
    cudaFree(d_output);
}

void measure_latency_shared_load(float gpu_clock_mhz) {
    printf("\n=== Measuring Shared Load Latency ===\n");
    
    float *d_output;
    float h_output[2];
    
    const int counter_iterations = 100;
    const int instructions_per_counter = 256;
    
    CUDA_CHECK(cudaMalloc(&d_output, 2 * sizeof(float)));
    
    latency_shared_load<<<1, 256>>>(d_output, counter_iterations);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_output, d_output, 2 * sizeof(float), cudaMemcpyDeviceToHost));
    
    float total_cycles = h_output[1];
    int total_instructions = counter_iterations * instructions_per_counter;
    float latency = total_cycles / total_instructions;
    
    printf("Counter iterations: %d\n", counter_iterations);
    printf("Instructions per counter: %d\n", instructions_per_counter);
    printf("Total instructions: %d\n", total_instructions);
    printf("Total cycles: %.2f\n", total_cycles);
    printf("Latency per instruction: %.2f cycles\n", latency);
    printf("Latency: %.2f ns\n", latency * 1000.0f / gpu_clock_mhz);
    
    cudaFree(d_output);
}

void measure_throughput_fmul(int num_warps, float gpu_clock_mhz) {
    printf("\n=== Measuring FMUL Throughput (Warps: %d) ===\n", num_warps);
    
    int num_threads = num_warps * WARP_SIZE;
    
    // Calculate blocks and threads per block
    // If warps <= 32, use 1 block. Otherwise, use multiple blocks.
    int threads_per_block = (num_threads <= 1024) ? num_threads : 1024;
    int blocks = (num_threads + threads_per_block - 1) / threads_per_block;
    
    float *d_input, *d_output;
    float *h_input = (float*)malloc(num_threads * sizeof(float));
    float *h_output = (float*)malloc((num_threads + 1) * sizeof(float));
    
    for (int i = 0; i < num_threads; i++) h_input[i] = 1.5f;
    
    CUDA_CHECK(cudaMalloc(&d_input, num_threads * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, (num_threads + 1) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, num_threads * sizeof(float), cudaMemcpyHostToDevice));
    
    // Warm-up
    throughput_fmul<<<blocks, threads_per_block>>>(d_input, d_output, ITERATIONS);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Actual measurement
    throughput_fmul<<<blocks, threads_per_block>>>(d_input, d_output, ITERATIONS);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_output, d_output, (num_threads + 1) * sizeof(float), cudaMemcpyDeviceToHost));
    
    float total_cycles = h_output[num_threads];
    float total_ops = (float)num_threads * ITERATIONS * 4;  // 4 ops per iteration
    float throughput = total_ops / total_cycles;
    
    printf("Blocks: %d, Threads per block: %d, Total threads: %d\n", 
           blocks, threads_per_block, num_threads);
    printf("Total cycles: %.2f\n", total_cycles);
    printf("Total operations: %.0f\n", total_ops);
    printf("Throughput: %.2f ops/cycle\n", throughput);
    printf("Throughput: %.2f GFLOPS\n", throughput * gpu_clock_mhz / 1000.0f);
    
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
}

void measure_peakwarps_fmul(float gpu_clock_mhz) {
    printf("\n=== Measuring Peak Warps for FMUL ===\n");
    printf("%-10s %-10s %-12s %-15s %-15s %-15s\n", 
           "Warps", "Blocks", "Threads", "Cycles", "Throughput", "GFLOPS");
    printf("--------------------------------------------------------------------------------\n");
    
    // Test from 1 to 64 warps (or higher to match Figure 3)
    // When warps > 32, use multiple blocks
    for (int num_warps = 1; num_warps <= 64; num_warps *= 2) {
        int num_threads = num_warps * WARP_SIZE;
        int threads_per_block = (num_threads <= 1024) ? num_threads : 1024;
        int blocks = (num_threads + threads_per_block - 1) / threads_per_block;
        
        float *d_input, *d_output;
        int *d_timing;
        float *h_input = (float*)malloc(num_threads * sizeof(float));
        float *h_output = (float*)malloc(num_threads * sizeof(float));
        int h_timing;
        
        for (int i = 0; i < num_threads; i++) h_input[i] = 1.5f;
        
        CUDA_CHECK(cudaMalloc(&d_input, num_threads * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output, num_threads * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_timing, sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_input, h_input, num_threads * sizeof(float), cudaMemcpyHostToDevice));
        
        // Warm-up
        peakwarps_fmul<<<blocks, threads_per_block>>>(d_input, d_output, ITERATIONS, d_timing);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Actual measurement
        peakwarps_fmul<<<blocks, threads_per_block>>>(d_input, d_output, ITERATIONS, d_timing);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaMemcpy(&h_timing, d_timing, sizeof(int), cudaMemcpyDeviceToHost));
        
        float total_cycles = (float)h_timing;
        float total_ops = (float)num_threads * ITERATIONS * 3;
        float throughput = total_ops / total_cycles;
        float gflops = throughput * gpu_clock_mhz / 1000.0f;
        
        printf("%-10d %-10d %-12d %-15.2f %-15.2f %-15.2f\n", 
               num_warps, blocks, num_threads, total_cycles, throughput, gflops);
        
        free(h_input);
        free(h_output);
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_timing);
    }
}

// New function to replicate Figure 3: Throughput vs Warps for different ILP levels
void measure_throughput_vs_ilp(float gpu_clock_mhz) {
    // Get GPU properties to determine limits
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
    int max_threads_per_block = prop.maxThreadsPerBlock;
    int max_warps_per_sm = max_threads_per_sm / WARP_SIZE;
    int max_warps_per_block = max_threads_per_block / WARP_SIZE;
    
    printf("\n\n### REPLICATING FIGURE 3: Throughput vs Warps for Different ILP ###\n");
    printf("GPU Limits:\n");
    printf("  Max threads/SM: %d (%d warps/SM)\n", max_threads_per_sm, max_warps_per_sm);
    printf("  Max threads/block: %d (%d warps/block)\n", max_threads_per_block, max_warps_per_block);
    printf("\nNote: Testing single-block (single-SM) behavior up to %d warps\n", max_warps_per_block);
    printf("Paper's K20 had 64 warps/SM and 32 warps/block max.\n\n");
    
    printf("\n--- ILP = 1 (Single dependent variable) ---\n");
    printf("%-10s %-10s %-15s %-15s\n", "Warps", "Threads", "Throughput", "GFLOPS");
    printf("--------------------------------------------------------\n");
    
    // Use only 1 block, limited by max_warps_per_block
    for (int num_warps = 1; num_warps <= max_warps_per_block; num_warps *= 2) {
        int num_threads = num_warps * WARP_SIZE;
        
        float *d_input, *d_output;
        float *h_input = (float*)malloc(num_threads * sizeof(float));
        float *h_output = (float*)malloc((num_threads + 1) * sizeof(float));
        
        for (int i = 0; i < num_threads; i++) h_input[i] = 1.5f;
        
        CUDA_CHECK(cudaMalloc(&d_input, num_threads * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output, (num_threads + 1) * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_input, h_input, num_threads * sizeof(float), cudaMemcpyHostToDevice));
        
        // Warm-up and actual measurement
        throughput_fmul_ilp1<<<1, num_threads>>>(d_input, d_output, ITERATIONS);
        CUDA_CHECK(cudaDeviceSynchronize());
        throughput_fmul_ilp1<<<1, num_threads>>>(d_input, d_output, ITERATIONS);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaMemcpy(h_output, d_output, (num_threads + 1) * sizeof(float), cudaMemcpyDeviceToHost));
        
        float total_cycles = h_output[num_threads];
        float total_ops = (float)num_threads * ITERATIONS;  // 1 op per iteration
        float throughput = total_ops / total_cycles;
        
        printf("%-10d %-10d %-15.2f %-15.2f\n", num_warps, num_threads, 
               throughput, throughput * gpu_clock_mhz / 1000.0f);
        
        free(h_input);
        free(h_output);
        cudaFree(d_input);
        cudaFree(d_output);
    }
    
    printf("\n--- ILP = 2 (Two independent variables) ---\n");
    printf("%-10s %-10s %-15s %-15s\n", "Warps", "Threads", "Throughput", "GFLOPS");
    printf("--------------------------------------------------------\n");
    
    for (int num_warps = 1; num_warps <= max_warps_per_block; num_warps *= 2) {
        int num_threads = num_warps * WARP_SIZE;
        
        float *d_input, *d_output;
        float *h_input = (float*)malloc(num_threads * sizeof(float));
        float *h_output = (float*)malloc((num_threads + 1) * sizeof(float));
        
        for (int i = 0; i < num_threads; i++) h_input[i] = 1.5f;
        
        CUDA_CHECK(cudaMalloc(&d_input, num_threads * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output, (num_threads + 1) * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_input, h_input, num_threads * sizeof(float), cudaMemcpyHostToDevice));
        
        throughput_fmul_ilp2<<<1, num_threads>>>(d_input, d_output, ITERATIONS);
        CUDA_CHECK(cudaDeviceSynchronize());
        throughput_fmul_ilp2<<<1, num_threads>>>(d_input, d_output, ITERATIONS);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaMemcpy(h_output, d_output, (num_threads + 1) * sizeof(float), cudaMemcpyDeviceToHost));
        
        float total_cycles = h_output[num_threads];
        float total_ops = (float)num_threads * ITERATIONS * 2;  // 2 ops per iteration
        float throughput = total_ops / total_cycles;
        
        printf("%-10d %-10d %-15.2f %-15.2f\n", num_warps, num_threads, 
               throughput, throughput * gpu_clock_mhz / 1000.0f);
        
        free(h_input);
        free(h_output);
        cudaFree(d_input);
        cudaFree(d_output);
    }
    
    printf("\n--- ILP = 3 (Three independent variables) ---\n");
    printf("%-10s %-10s %-15s %-15s\n", "Warps", "Threads", "Throughput", "GFLOPS");
    printf("--------------------------------------------------------\n");
    
    for (int num_warps = 1; num_warps <= max_warps_per_block; num_warps *= 2) {
        int num_threads = num_warps * WARP_SIZE;
        
        float *d_input, *d_output;
        float *h_input = (float*)malloc(num_threads * sizeof(float));
        float *h_output = (float*)malloc((num_threads + 1) * sizeof(float));
        
        for (int i = 0; i < num_threads; i++) h_input[i] = 1.5f;
        
        CUDA_CHECK(cudaMalloc(&d_input, num_threads * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output, (num_threads + 1) * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_input, h_input, num_threads * sizeof(float), cudaMemcpyHostToDevice));
        
        throughput_fmul_ilp3<<<1, num_threads>>>(d_input, d_output, ITERATIONS);
        CUDA_CHECK(cudaDeviceSynchronize());
        throughput_fmul_ilp3<<<1, num_threads>>>(d_input, d_output, ITERATIONS);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaMemcpy(h_output, d_output, (num_threads + 1) * sizeof(float), cudaMemcpyDeviceToHost));
        
        float total_cycles = h_output[num_threads];
        float total_ops = (float)num_threads * ITERATIONS * 3;  // 3 ops per iteration
        float throughput = total_ops / total_cycles;
        
        printf("%-10d %-10d %-15.2f %-15.2f\n", num_warps, num_threads, 
               throughput, throughput * gpu_clock_mhz / 1000.0f);
        
        free(h_input);
        free(h_output);
        cudaFree(d_input);
        cudaFree(d_output);
    }
}

// ============================================================================
// MAIN FUNCTION
// ============================================================================

int main() {
    // Get device properties
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    printf("========================================\n");
    printf("CUDA Microbenchmarking Tool\n");
    printf("Based on Figure 2 from the Paper\n");
    printf("========================================\n");
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Clock Rate: %.2f MHz\n", prop.clockRate / 1000.0f);
    printf("Number of SMs: %d\n", prop.multiProcessorCount);
    printf("Warp Size: %d\n", prop.warpSize);
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Max Threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("========================================\n");
    
    float gpu_clock_mhz = prop.clockRate / 1000.0f;
    
    // Latency measurements
    printf("\n\n### LATENCY MEASUREMENTS ###\n");
    measure_latency_fadd(gpu_clock_mhz);
    measure_latency_fmul(gpu_clock_mhz);
    measure_latency_fma(gpu_clock_mhz);
    measure_latency_global_load(gpu_clock_mhz);
    measure_latency_shared_load(gpu_clock_mhz);
    
    // Throughput measurements
    printf("\n\n### THROUGHPUT MEASUREMENTS ###\n");
    measure_throughput_fmul(4, gpu_clock_mhz);   // 4 warps
    measure_throughput_fmul(16, gpu_clock_mhz);  // 16 warps
    measure_throughput_fmul(32, gpu_clock_mhz);  // 32 warps
    measure_throughput_fmul(64, gpu_clock_mhz);  // 64 warps
    
    // Peak warps measurements
    printf("\n\n### PEAK WARPS MEASUREMENTS ###\n");
    measure_peakwarps_fmul(gpu_clock_mhz);
    
    // Replicate Figure 3: Throughput vs Warps for different ILP levels
    measure_throughput_vs_ilp(gpu_clock_mhz);
    
    printf("\n\n========================================\n");
    printf("Microbenchmarking Complete!\n");
    printf("========================================\n");
    
    return 0;
}
