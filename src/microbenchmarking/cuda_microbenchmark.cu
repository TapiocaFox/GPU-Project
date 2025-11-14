#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <string>
#include <cmath>

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

struct LatencyMeasurement {
    std::string name;
    float total_cycles;
    int total_instructions;
    float latency_cycles;
    float latency_ns;
};

struct ILPThroughputMeasurement {
    std::string instruction;
    int ilp;
    int warps;
    int threads;
    float total_cycles;
    float total_ops;
    float throughput_ops_per_cycle;
    float throughput_gflops;
};

struct ComputeDelayModel {
    std::string instruction;
    float latency_cycles;
    float m_i_c;          // concurrency threshold (m_i^c)
    float stall_term;     // S_zw / t_p_i^c
    float tolerance;
    bool has_stall_region;
};

struct InstructionSummary {
    std::string instruction;
    float latency;                    // l_i^c
    float throughput_ilp1;            // Throughput at ILP=1
    float throughput_ilp2;            // Throughput at ILP=2
    float throughput_ilp3;            // Throughput at ILP=3
    int peak_warps_ilp1;              // Peak warps at ILP=1
    int peak_warps_ilp2;              // Peak warps at ILP=2
    int peak_warps_ilp3;              // Peak warps at ILP=3
};

static float safe_divide(float numerator, float denominator, float default_value = 0.0f) {
    if (fabsf(denominator) < 1e-9f) {
        return default_value;
    }
    return numerator / denominator;
}

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

// Kernel for measuring floating-point addition throughput with ILP=1
__global__ void throughput_fadd_ilp1(float *input, float *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    float a = input[tid];
    
    clock_t start = clock();
    
    // ILP=1: Only one variable, fully dependent operations
    #pragma unroll 1
    for (int i = 0; i < iterations; i++) {
        a = a + 1.0f;  // Dependent chain
    }
    
    clock_t end = clock();
    
    output[tid] = a;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (float)(end - start);
    }
}

// Kernel for measuring floating-point addition throughput with ILP=2
__global__ void throughput_fadd_ilp2(float *input, float *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    float a = input[tid];
    float b = input[tid] + 1.0f;
    
    clock_t start = clock();
    
    // ILP=2: Two independent variables
    #pragma unroll 2
    for (int i = 0; i < iterations; i++) {
        a = a + 1.0f;  // Independent from b
        b = b + 1.0f;  // Independent from a
    }
    
    clock_t end = clock();
    
    output[tid] = a + b;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (float)(end - start);
    }
}

// Kernel for measuring floating-point addition throughput with ILP=3
__global__ void throughput_fadd_ilp3(float *input, float *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    float a = input[tid];
    float b = input[tid] + 1.0f;
    float c = input[tid] + 2.0f;
    
    clock_t start = clock();
    
    // ILP=3: Three independent variables
    #pragma unroll 3
    for (int i = 0; i < iterations; i++) {
        a = a + 1.0f;  // Independent
        b = b + 1.0f;  // Independent
        c = c + 1.0f;  // Independent
    }
    
    clock_t end = clock();
    
    output[tid] = a + b + c;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (float)(end - start);
    }
}

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

// Kernel for measuring FMA throughput with ILP=1
__global__ void throughput_fma_ilp1(float *input, float *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    float a = input[tid];
    float b = 1.000001f;
    float c = 1.000002f;
    
    clock_t start = clock();
    
    // ILP=1: Only one variable, fully dependent operations
    #pragma unroll 1
    for (int i = 0; i < iterations; i++) {
        a = a * b + c;  // Dependent chain
    }
    
    clock_t end = clock();
    
    output[tid] = a;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (float)(end - start);
    }
}

// Kernel for measuring FMA throughput with ILP=2
__global__ void throughput_fma_ilp2(float *input, float *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    float a = input[tid];
    float b = input[tid] + 1.0f;
    float c = 1.000001f;
    float d = 1.000002f;
    
    clock_t start = clock();
    
    // ILP=2: Two independent variables
    #pragma unroll 2
    for (int i = 0; i < iterations; i++) {
        a = a * c + d;  // Independent from b
        b = b * d + c;  // Independent from a
    }
    
    clock_t end = clock();
    
    output[tid] = a + b;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (float)(end - start);
    }
}

// Kernel for measuring FMA throughput with ILP=3
__global__ void throughput_fma_ilp3(float *input, float *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    float a = input[tid];
    float b = input[tid] + 1.0f;
    float c = input[tid] + 2.0f;
    float d = 1.000001f;
    float e = 1.000002f;
    
    clock_t start = clock();
    
    // ILP=3: Three independent variables
    #pragma unroll 3
    for (int i = 0; i < iterations; i++) {
        a = a * d + e;  // Independent
        b = b * e + d;  // Independent
        c = c * d + e;  // Independent
    }
    
    clock_t end = clock();
    
    output[tid] = a + b + c;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (float)(end - start);
    }
}

// Original high-ILP kernel (for general throughput testing)
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

LatencyMeasurement measure_latency_fadd(float gpu_clock_mhz) {
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
    
    LatencyMeasurement result;
    result.name = "FADD";
    result.total_cycles = total_cycles;
    result.total_instructions = total_instructions;
    result.latency_cycles = latency;
    result.latency_ns = latency * 1000.0f / gpu_clock_mhz;
    
    return result;
}

LatencyMeasurement measure_latency_fmul(float gpu_clock_mhz) {
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

    LatencyMeasurement result;
    result.name = "FMUL";
    result.total_cycles = total_cycles;
    result.total_instructions = total_instructions;
    result.latency_cycles = latency;
    result.latency_ns = latency * 1000.0f / gpu_clock_mhz;
    
    return result;
}

LatencyMeasurement measure_latency_fma(float gpu_clock_mhz) {
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
    
    LatencyMeasurement result;
    result.name = "FMA";
    result.total_cycles = total_cycles;
    result.total_instructions = total_instructions;
    result.latency_cycles = latency;
    result.latency_ns = latency * 1000.0f / gpu_clock_mhz;
    
    return result;
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
std::vector<ILPThroughputMeasurement> measure_throughput_vs_ilp(float gpu_clock_mhz) {
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
    
    std::vector<ILPThroughputMeasurement> results;
    results.reserve(max_warps_per_block * 3);  // rough estimate
    
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
        
        ILPThroughputMeasurement measurement;
        measurement.instruction = "FMUL";
        measurement.ilp = 1;
        measurement.warps = num_warps;
        measurement.threads = num_threads;
        measurement.total_cycles = total_cycles;
        measurement.total_ops = total_ops;
        measurement.throughput_ops_per_cycle = throughput;
        measurement.throughput_gflops = throughput * gpu_clock_mhz / 1000.0f;
        results.push_back(measurement);
        
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
        
        ILPThroughputMeasurement measurement;
        measurement.instruction = "FMUL";
        measurement.ilp = 2;
        measurement.warps = num_warps;
        measurement.threads = num_threads;
        measurement.total_cycles = total_cycles;
        measurement.total_ops = total_ops;
        measurement.throughput_ops_per_cycle = throughput;
        measurement.throughput_gflops = throughput * gpu_clock_mhz / 1000.0f;
        results.push_back(measurement);
        
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
        
        ILPThroughputMeasurement measurement;
        measurement.instruction = "FMUL";
        measurement.ilp = 3;
        measurement.warps = num_warps;
        measurement.threads = num_threads;
        measurement.total_cycles = total_cycles;
        measurement.total_ops = total_ops;
        measurement.throughput_ops_per_cycle = throughput;
        measurement.throughput_gflops = throughput * gpu_clock_mhz / 1000.0f;
        results.push_back(measurement);
        
        free(h_input);
        free(h_output);
        cudaFree(d_input);
        cudaFree(d_output);
    }
    
    return results;
}

ComputeDelayModel calibrate_compute_delay_model(
    const LatencyMeasurement &latency_result,
    const std::vector<ILPThroughputMeasurement> &throughput_results,
    float tolerance_rel = 0.02f) {
    
    ComputeDelayModel model{};
    model.instruction = latency_result.name;
    model.latency_cycles = latency_result.latency_cycles;
    model.m_i_c = 0.0f;
    model.stall_term = 0.0f;
    model.tolerance = tolerance_rel;
    model.has_stall_region = false;
    
    float max_base_concurrency = 0.0f;
    float stall_weighted_sum = 0.0f;
    float stall_weight = 0.0f;
    
    for (const auto &measurement : throughput_results) {
        if (measurement.instruction != latency_result.name) {
            continue;
        }
        
        float ilp = static_cast<float>(measurement.ilp);
        float tlp = static_cast<float>(measurement.warps);
        float concurrency = ilp * tlp;
        if (concurrency <= 0.0f) {
            continue;
        }
        
        float measured_delay = safe_divide(measurement.total_cycles, measurement.total_ops, 0.0f);
        if (measured_delay <= 0.0f) {
            continue;
        }
        
        float base_delay = latency_result.latency_cycles / concurrency;
        float diff = fabsf(measured_delay - base_delay);
        float rel_diff = diff / base_delay;
        
        if (rel_diff <= tolerance_rel) {
            if (concurrency > max_base_concurrency) {
                max_base_concurrency = concurrency;
            }
        } else if (measured_delay > base_delay) {
            float extra = measured_delay - base_delay;
            stall_weighted_sum += extra * measurement.total_ops;
            stall_weight += measurement.total_ops;
            model.has_stall_region = true;
        }
    }
    
    model.m_i_c = max_base_concurrency;
    if (stall_weight > 0.0f) {
        model.stall_term = stall_weighted_sum / stall_weight;
    }
    
    return model;
}

float compute_delay_cycles(const ComputeDelayModel &model, float ilp, float tlp) {
    float concurrency = ilp * tlp;
    if (concurrency <= 0.0f) {
        return 0.0f;
    }
    
    float base_delay = model.latency_cycles / concurrency;
    if (concurrency <= model.m_i_c + model.tolerance) {
        return base_delay;
    }
    
    float stall = model.has_stall_region ? model.stall_term : 0.0f;
    return base_delay + stall;
}

void report_compute_delay_model(const ComputeDelayModel &model,
                                const std::vector<ILPThroughputMeasurement> &throughput_results,
                                float gpu_clock_mhz) {
    printf("\n\n### INSTRUCTION DELAY MODEL (Compute) ###\n");
    printf("Instruction: %s\n", model.instruction.c_str());
    printf("Latency (l_i^c): %.4f cycles\n", model.latency_cycles);
    printf("Concurrency threshold (m_i^c): %.2f\n", model.m_i_c);
    if (model.has_stall_region) {
        printf("Scheduler stall term (S_zw / t_p_i^c): %.6f cycles\n", model.stall_term);
    } else {
        printf("Scheduler stall term (S_zw / t_p_i^c): not observed (no stall region)\n");
    }
    printf("Calibration tolerance: %.2f%%\n", model.tolerance * 100.0f);
    
    printf("\n%-6s %-6s %-12s %-12s %-12s %-15s %-15s %-15s %-15s\n",
           "ILP", "TLP", "Base Delay", "Stall", "Pred Delay", "Pred Cycles", "Meas Cycles", "Δ Delay", "Err %");
    
    const float eps = 1e-6f;
    
    for (const auto &measurement : throughput_results) {
        if (measurement.instruction != model.instruction) {
            continue;
        }
        
        float ilp = static_cast<float>(measurement.ilp);
        float tlp = static_cast<float>(measurement.warps);
        if (ilp <= eps || tlp <= eps) {
            continue;
        }
        
        float concurrency = ilp * tlp;
        float base_delay = model.latency_cycles / concurrency;
        float stall = (concurrency > model.m_i_c + model.tolerance && model.has_stall_region) ? model.stall_term : 0.0f;
        float predicted_delay = base_delay + stall;
        float predicted_cycles = predicted_delay * measurement.total_ops;
        float measured_cycles = measurement.total_cycles;
        float measured_delay = safe_divide(measured_cycles, measurement.total_ops, 0.0f);
        float delta_delay = predicted_delay - measured_delay;
        float error_pct = safe_divide(predicted_cycles - measured_cycles, measured_cycles, 0.0f) * 100.0f;
        
        printf("%-6d %-6d %-12.5f %-12.5f %-12.5f %-15.2f %-15.2f %-15.5f %-15.2f\n",
               measurement.ilp,
               measurement.warps,
               base_delay,
               stall,
               predicted_delay,
               predicted_cycles,
               measured_cycles,
               delta_delay,
               error_pct);
    }
}

// Generic function to measure throughput for a given instruction type and ILP
std::vector<ILPThroughputMeasurement> measure_instruction_throughput(
    const std::string &instruction_name,
    void (*kernel_ilp1)(float*, float*, int),
    void (*kernel_ilp2)(float*, float*, int),
    void (*kernel_ilp3)(float*, float*, int),
    int ops_per_iteration_ilp1,
    int ops_per_iteration_ilp2,
    int ops_per_iteration_ilp3,
    float gpu_clock_mhz,
    bool verbose = false) {
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int max_threads_per_block = prop.maxThreadsPerBlock;
    int max_warps_per_block = max_threads_per_block / WARP_SIZE;
    
    std::vector<ILPThroughputMeasurement> results;
    results.reserve(max_warps_per_block * 3);
    
    // Test different ILP levels
    for (int ilp = 1; ilp <= 3; ilp++) {
        void (*kernel_func)(float*, float*, int) = nullptr;
        int ops_per_iteration = 0;
        
        if (ilp == 1) {
            kernel_func = kernel_ilp1;
            ops_per_iteration = ops_per_iteration_ilp1;
        } else if (ilp == 2) {
            kernel_func = kernel_ilp2;
            ops_per_iteration = ops_per_iteration_ilp2;
        } else if (ilp == 3) {
            kernel_func = kernel_ilp3;
            ops_per_iteration = ops_per_iteration_ilp3;
        }
        
        if (kernel_func == nullptr) continue;
        
        if (verbose) {
            printf("\n--- %s ILP = %d ---\n", instruction_name.c_str(), ilp);
            printf("%-10s %-10s %-15s\n", "Warps", "Threads", "Throughput");
        }
        
        // Test different warp counts
        for (int num_warps = 1; num_warps <= max_warps_per_block; num_warps *= 2) {
            int num_threads = num_warps * WARP_SIZE;
            
            float *d_input, *d_output;
            float *h_input = (float*)malloc(num_threads * sizeof(float));
            float *h_output = (float*)malloc((num_threads + 1) * sizeof(float));
            
            for (int i = 0; i < num_threads; i++) h_input[i] = 1.5f;
            
            CUDA_CHECK(cudaMalloc(&d_input, num_threads * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_output, (num_threads + 1) * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_input, h_input, num_threads * sizeof(float), cudaMemcpyHostToDevice));
            
            // Warm-up
            kernel_func<<<1, num_threads>>>(d_input, d_output, ITERATIONS);
            CUDA_CHECK(cudaDeviceSynchronize());
            
            // Actual measurement
            kernel_func<<<1, num_threads>>>(d_input, d_output, ITERATIONS);
            CUDA_CHECK(cudaDeviceSynchronize());
            
            CUDA_CHECK(cudaMemcpy(h_output, d_output, (num_threads + 1) * sizeof(float), cudaMemcpyDeviceToHost));
            
            float total_cycles = h_output[num_threads];
            float total_ops = (float)num_threads * ITERATIONS * ops_per_iteration;
            float throughput = total_ops / total_cycles;
            
            if (verbose) {
                printf("%-10d %-10d %-15.2f\n", num_warps, num_threads, throughput);
            }
            
            ILPThroughputMeasurement measurement;
            measurement.instruction = instruction_name;
            measurement.ilp = ilp;
            measurement.warps = num_warps;
            measurement.threads = num_threads;
            measurement.total_cycles = total_cycles;
            measurement.total_ops = total_ops;
            measurement.throughput_ops_per_cycle = throughput;
            measurement.throughput_gflops = throughput * gpu_clock_mhz / 1000.0f;
            results.push_back(measurement);
            
            free(h_input);
            free(h_output);
            cudaFree(d_input);
            cudaFree(d_output);
        }
    }
    
    return results;
}

// Calculate peak warps for a given ILP level
// PeakWarps is the warps count where throughput reaches its maximum
int calculate_peak_warps(const std::vector<ILPThroughputMeasurement> &measurements,
                         const std::string &instruction_name,
                         int ilp) {
    float max_throughput = 0.0f;
    int peak_warps = 0;
    
    // Find the maximum throughput and its corresponding warps
    for (const auto &m : measurements) {
        if (m.instruction == instruction_name && m.ilp == ilp) {
            if (m.throughput_ops_per_cycle > max_throughput) {
                max_throughput = m.throughput_ops_per_cycle;
                peak_warps = m.warps;
            }
        }
    }
    
    // If multiple warps have the same max throughput, return the maximum warps
    // This handles the case where throughput plateaus
    for (const auto &m : measurements) {
        if (m.instruction == instruction_name && m.ilp == ilp) {
            // Allow small floating point differences (0.1%)
            if (fabsf(m.throughput_ops_per_cycle - max_throughput) / max_throughput < 0.001f) {
                if (m.warps > peak_warps) {
                    peak_warps = m.warps;
                }
            }
        }
    }
    
    return peak_warps;
}

// Get throughput at a specific ILP level (use maximum measured throughput)
float get_throughput_at_ilp(const std::vector<ILPThroughputMeasurement> &measurements,
                            const std::string &instruction_name,
                            int ilp) {
    float max_throughput = 0.0f;
    
    for (const auto &m : measurements) {
        if (m.instruction == instruction_name && m.ilp == ilp) {
            if (m.throughput_ops_per_cycle > max_throughput) {
                max_throughput = m.throughput_ops_per_cycle;
            }
        }
    }
    
    return max_throughput;
}

// Generate instruction summary table like paper Table II
InstructionSummary generate_instruction_summary(
    const LatencyMeasurement &latency,
    const std::vector<ILPThroughputMeasurement> &throughput_results) {
    
    InstructionSummary summary;
    summary.instruction = latency.name;
    summary.latency = latency.latency_cycles;
    summary.throughput_ilp1 = get_throughput_at_ilp(throughput_results, latency.name, 1);
    summary.throughput_ilp2 = get_throughput_at_ilp(throughput_results, latency.name, 2);
    summary.throughput_ilp3 = get_throughput_at_ilp(throughput_results, latency.name, 3);
    summary.peak_warps_ilp1 = calculate_peak_warps(throughput_results, latency.name, 1);
    summary.peak_warps_ilp2 = calculate_peak_warps(throughput_results, latency.name, 2);
    summary.peak_warps_ilp3 = calculate_peak_warps(throughput_results, latency.name, 3);
    
    return summary;
}

// Print table like paper Table II
void print_instruction_summary_table(const std::vector<InstructionSummary> &summaries) {
    printf("\n\n");
    printf("===========================================================================================================\n");
    printf("TABLE: Compute Instruction Summary\n");
    printf("===========================================================================================================\n");
    printf("%-15s %-10s %-15s %-15s %-15s %-15s %-15s %-15s\n",
           "Instruction", "Latency", "Throughput", "Throughput", "Throughput",
           "PeakWarps", "PeakWarps", "PeakWarps");
    printf("%-15s %-10s %-15s %-15s %-15s %-15s %-15s %-15s\n",
           "", "", "ILP=1", "ILP=2", "ILP=3", "ILP=1", "ILP=2", "ILP=3");
    printf("-----------------------------------------------------------------------------------------------------------\n");
    
    for (const auto &summary : summaries) {
        printf("%-15s %-10.2f %-15.2f %-15.2f %-15.2f %-15d %-15d %-15d\n",
               summary.instruction.c_str(),
               summary.latency,
               summary.throughput_ilp1,
               summary.throughput_ilp2,
               summary.throughput_ilp3,
               summary.peak_warps_ilp1,
               summary.peak_warps_ilp2,
               summary.peak_warps_ilp3);
    }
    
    printf("===========================================================================================================\n");
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
    LatencyMeasurement fadd_latency = measure_latency_fadd(gpu_clock_mhz);
    LatencyMeasurement fmul_latency = measure_latency_fmul(gpu_clock_mhz);
    LatencyMeasurement fma_latency = measure_latency_fma(gpu_clock_mhz);
    measure_latency_global_load(gpu_clock_mhz);
    measure_latency_shared_load(gpu_clock_mhz);
    
    // Measure throughput for all instruction types with different ILP levels
    printf("\n\n### THROUGHPUT MEASUREMENTS (All Instructions) ###\n");
    
    // FADD throughput measurements
    std::vector<ILPThroughputMeasurement> fadd_throughput = measure_instruction_throughput(
        "FADD",
        throughput_fadd_ilp1,
        throughput_fadd_ilp2,
        throughput_fadd_ilp3,
        1,  // ops per iteration ILP=1
        2,  // ops per iteration ILP=2
        3,  // ops per iteration ILP=3
        gpu_clock_mhz,
        false  // verbose
    );
    
    // FMUL throughput measurements
    std::vector<ILPThroughputMeasurement> fmul_throughput = measure_instruction_throughput(
        "FMUL",
        throughput_fmul_ilp1,
        throughput_fmul_ilp2,
        throughput_fmul_ilp3,
        1,  // ops per iteration ILP=1
        2,  // ops per iteration ILP=2
        3,  // ops per iteration ILP=3
        gpu_clock_mhz,
        false  // verbose
    );
    
    // FMA throughput measurements
    std::vector<ILPThroughputMeasurement> fma_throughput = measure_instruction_throughput(
        "FMA",
        throughput_fma_ilp1,
        throughput_fma_ilp2,
        throughput_fma_ilp3,
        1,  // ops per iteration ILP=1
        2,  // ops per iteration ILP=2
        3,  // ops per iteration ILP=3
        gpu_clock_mhz,
        false  // verbose
    );
    
    // Combine all throughput results
    std::vector<ILPThroughputMeasurement> all_throughput;
    all_throughput.insert(all_throughput.end(), fadd_throughput.begin(), fadd_throughput.end());
    all_throughput.insert(all_throughput.end(), fmul_throughput.begin(), fmul_throughput.end());
    all_throughput.insert(all_throughput.end(), fma_throughput.begin(), fma_throughput.end());
    
    // Generate instruction summaries
    std::vector<InstructionSummary> summaries;
    summaries.push_back(generate_instruction_summary(fadd_latency, fadd_throughput));
    summaries.push_back(generate_instruction_summary(fmul_latency, fmul_throughput));
    summaries.push_back(generate_instruction_summary(fma_latency, fma_throughput));
    
    // Print table like paper Table II
    print_instruction_summary_table(summaries);
    
    // Instruction delay synthesis (for FMUL as example)
    printf("\n\n### INSTRUCTION DELAY MODEL (Example: FMUL) ###\n");
    ComputeDelayModel fmul_model = calibrate_compute_delay_model(fmul_latency, fmul_throughput);
    report_compute_delay_model(fmul_model, fmul_throughput, gpu_clock_mhz);
    
    printf("\n\n========================================\n");
    printf("Microbenchmarking Complete!\n");
    printf("========================================\n");
    
    return 0;
}
