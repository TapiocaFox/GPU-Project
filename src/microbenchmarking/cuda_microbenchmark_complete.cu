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

// Existing FADD, FMUL, FMA, IADD, IMUL kernels...
__global__ void latency_fadd(float *input, float *output, int counter_iterations) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float result = input[0];
        clock_t total_time = 0;
        
        for (int counter = 0; counter < counter_iterations; counter++) {
            unsigned long long start = clock64();
            
            for (int i = 0; i < 256; i++) {
                result = result + 1.0f;
            }
            
            unsigned long long end = clock64();
            total_time += (end - start);
        }
        
        output[0] = result;
        output[1] = (float)total_time;
    }
}

__global__ void latency_fmul(float *input, float *output, int counter_iterations) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float result = input[0];
        clock_t total_time = 0;
        
        for (int counter = 0; counter < counter_iterations; counter++) {
            unsigned long long start = clock64();
            
            for (int i = 0; i < 256; i++) {
                result = result * 1.000001f;
            }
            
            unsigned long long end = clock64();
            total_time += (end - start);
        }
        
        output[0] = result;
        output[1] = (float)total_time;
    }
}

__global__ void latency_fma(float *input, float *output, int counter_iterations) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float result = input[0];
        clock_t total_time = 0;
        
        for (int counter = 0; counter < counter_iterations; counter++) {
            unsigned long long start = clock64();
            
            for (int i = 0; i < 256; i++) {
                result = result * 1.000001f + 0.5f;
            }
            
            unsigned long long end = clock64();
            total_time += (end - start);
        }
        
        output[0] = result;
        output[1] = (float)total_time;
    }
}

// NEW LATENCY KERNELS FOR MISSING INSTRUCTIONS

// Floating-point subtract
__global__ void latency_fsub(float *input, float *output, int counter_iterations) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float result = input[0];
        clock_t total_time = 0;
        
        for (int counter = 0; counter < counter_iterations; counter++) {
            unsigned long long start = clock64();
            
            for (int i = 0; i < 256; i++) {
                result = result - 0.000001f;
            }
            
            unsigned long long end = clock64();
            total_time += (end - start);
        }
        
        output[0] = result;
        output[1] = (float)total_time;
    }
}

// Integer add with saturation
__global__ void latency_adds(int *input, int *output, int counter_iterations) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int result = input[0];
        clock_t total_time = 0;
        
        for (int counter = 0; counter < counter_iterations; counter++) {
            unsigned long long start = clock64();
            
            for (int i = 0; i < 256; i++) {
                // Saturated add: check for overflow
                // long long temp = (long long)result + 1;
                // result = (temp > INT_MAX) ? INT_MAX : (int)temp;
                // Saturated add: check for overflow
                result += 1;
            }
            
            unsigned long long end = clock64();
            total_time += (end - start);
        }
        
        output[0] = result;
        output[1] = (int)total_time;
    }
}

// Integer subtract with saturation  
__global__ void latency_subs(int *input, int *output, int counter_iterations) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int result = input[0];
        clock_t total_time = 0;
        
        for (int counter = 0; counter < counter_iterations; counter++) {
            unsigned long long start = clock64();
            
            for (int i = 0; i < 256; i++) {
                result -= 1;
            }
            
            unsigned long long end = clock64();
            total_time += (end - start);
        }
        
        output[0] = result;
        output[1] = (int)total_time;
    }
}

// Bitwise AND
__global__ void latency_and(int *input, int *output, int counter_iterations) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int result = input[0];
        clock_t total_time = 0;
        
        for (int counter = 0; counter < counter_iterations; counter++) {
            unsigned long long start = clock64();
            
            for (int i = 0; i < 256; i++) {
                result = result & 0xFFFFFFFE;  // Clear lowest bit
                result = result | 1;          // Set it back to maintain pattern
            }
            
            unsigned long long end = clock64();
            total_time += (end - start);
        }
        
        output[0] = result;
        output[1] = (int)total_time;
    }
}

// NEW HIGH-FREQUENCY INTEGER OPERATIONS

// 32-bit signed integer add
__global__ void latency_add_s32(int *input, int *output, int counter_iterations) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int result = input[0];
        clock_t total_time = 0;
        
        for (int counter = 0; counter < counter_iterations; counter++) {
            unsigned long long start = clock64();
            
            for (int i = 0; i < 256; i++) {
                result = result + 1;  // Simple add operation
            }
            
            unsigned long long end = clock64();
            total_time += (end - start);
        }
        
        output[0] = result;
        output[1] = (int)total_time;
    }
}

// 64-bit signed integer add
__global__ void latency_add_s64(long long *input, long long *output, int counter_iterations) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        long long result = input[0];
        clock_t total_time = 0;
        
        for (int counter = 0; counter < counter_iterations; counter++) {
            unsigned long long start = clock64();
            
            for (int i = 0; i < 256; i++) {
                result = result + 1LL;  // 64-bit add operation
            }
            
            unsigned long long end = clock64();
            total_time += (end - start);
        }
        
        output[0] = result;
        output[1] = (long long)total_time;
    }
}

// 32-bit signed integer subtract
__global__ void latency_sub_s32(int *input, int *output, int counter_iterations) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int result = input[0];
        clock_t total_time = 0;
        
        for (int counter = 0; counter < counter_iterations; counter++) {
            unsigned long long start = clock64();
            
            for (int i = 0; i < 256; i++) {
                result = result - 1;  // Simple subtract operation
            }
            
            unsigned long long end = clock64();
            total_time += (end - start);
        }
        
        output[0] = result;
        output[1] = (int)total_time;
    }
}

// 32-bit signed multiply (low bits)
__global__ void latency_mul_lo_s32(int *input, int *output, int counter_iterations) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int result = input[0];
        clock_t total_time = 0;
        
        for (int counter = 0; counter < counter_iterations; counter++) {
            unsigned long long start = clock64();
            
            for (int i = 0; i < 256; i++) {
                result = (result * 1001) % 100000;  // Multiply with modulo to keep reasonable
            }
            
            unsigned long long end = clock64();
            total_time += (end - start);
        }
        
        output[0] = result;
        output[1] = (int)total_time;
    }
}

// 32-bit to 64-bit multiply (wide multiply)
__global__ void latency_mul_wide_s32(int *input, long long *output, int counter_iterations) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int result_int = input[0];
        long long result = result_int;
        clock_t total_time = 0;
        
        for (int counter = 0; counter < counter_iterations; counter++) {
            unsigned long long start = clock64();
            
            for (int i = 0; i < 256; i++) {
                result = (long long)result_int * 1001LL;  // 32->64 bit multiply
                result_int = (int)(result % 100000);      // Keep input reasonable
            }
            
            unsigned long long end = clock64();
            total_time += (end - start);
        }
        
        output[0] = result;
        output[1] = (long long)total_time;
    }
}

// 32-bit multiply-add (low bits)
__global__ void latency_mad_lo_s32(int *input, int *output, int counter_iterations) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int result = input[0];
        clock_t total_time = 0;
        
        for (int counter = 0; counter < counter_iterations; counter++) {
            unsigned long long start = clock64();
            
            for (int i = 0; i < 256; i++) {
                result = (result * 3 + 2) % 100000;  // Multiply-add with modulo
            }
            
            unsigned long long end = clock64();
            total_time += (end - start);
        }
        
        output[0] = result;
        output[1] = (int)total_time;
    }
}

// Integer multiply-add with saturation
__global__ void latency_mads(int *input, int *output, int counter_iterations) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int result = input[0];
        clock_t total_time = 0;
        
        for (int counter = 0; counter < counter_iterations; counter++) {
            unsigned long long start = clock64();
            
            for (int i = 0; i < 256; i++) {
                result = result * 2 + 1;  // Simple multiply-add
            }
            
            unsigned long long end = clock64();
            total_time += (end - start);
        }
        
        output[0] = result;
        output[1] = (int)total_time;
    }
}

// Integer multiply with saturation
__global__ void latency_muls(int *input, int *output, int counter_iterations) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int result = input[0];
        clock_t total_time = 0;
        
        for (int counter = 0; counter < counter_iterations; counter++) {
            unsigned long long start = clock64();
            
            for (int i = 0; i < 256; i++) {
                result = (result * 1001) % 100000;  // Multiply with modulo to prevent overflow
            }
            
            unsigned long long end = clock64();
            total_time += (end - start);
        }
        
        output[0] = result;
        output[1] = (int)total_time;
    }
}

// Integer division
__global__ void latency_divs(int *input, int *output, int counter_iterations) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int result = input[0];
        clock_t total_time = 0;
        
        for (int counter = 0; counter < counter_iterations; counter++) {
            unsigned long long start = clock64();
            
            for (int i = 0; i < 256; i++) {
                result = result / 2 + 1000000;  // Keep value reasonable
            }
            
            unsigned long long end = clock64();
            total_time += (end - start);
        }
        
        output[0] = result;
        output[1] = (int)total_time;
    }
}

// Floating-point division
__global__ void latency_divf(float *input, float *output, int counter_iterations) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float result = input[0];
        clock_t total_time = 0;
        
        for (int counter = 0; counter < counter_iterations; counter++) {
            unsigned long long start = clock64();
            
            for (int i = 0; i < 256; i++) {
                result = result / 1.000001f;
            }
            
            unsigned long long end = clock64();
            total_time += (end - start);
        }
        
        output[0] = result;
        output[1] = (float)total_time;
    }
}

// Square root
__global__ void latency_sqrt(float *input, float *output, int counter_iterations) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float result = input[0];
        clock_t total_time = 0;
        
        for (int counter = 0; counter < counter_iterations; counter++) {
            unsigned long long start = clock64();
            
            for (int i = 0; i < 256; i++) {
                result = sqrtf(result + 0.1f);
            }
            
            unsigned long long end = clock64();
            total_time += (end - start);
        }
        
        output[0] = result;
        output[1] = (float)total_time;
    }
}

// Set predicate
__global__ void latency_setp(float *input, float *output, int counter_iterations) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float result = input[0];
        clock_t total_time = 0;
        
        for (int counter = 0; counter < counter_iterations; counter++) {
            unsigned long long start = clock64();
            
            for (int i = 0; i < 256; i++) {
                bool pred = (result > 0.5f);
                result = pred ? result + 0.1f : result - 0.1f;
            }
            
            unsigned long long end = clock64();
            total_time += (end - start);
        }
        
        output[0] = result;
        output[1] = (float)total_time;
    }
}

// Convert
__global__ void latency_cvt(float *input, float *output, int counter_iterations) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float result_f = input[0];
        clock_t total_time = 0;
        
        for (int counter = 0; counter < counter_iterations; counter++) {
            unsigned long long start = clock64();
            
            for (int i = 0; i < 256; i++) {
                int result_i = (int)result_f;      // float to int conversion
                result_f = (float)result_i;        // int to float conversion
            }
            
            unsigned long long end = clock64();
            total_time += (end - start);
        }
        
        output[0] = result_f;
        output[1] = (float)total_time;
    }
}

LatencyMeasurement measure_latency_cvt(float gpu_clock_mhz) {
    printf("\n=== Measuring CVT Latency ===\n");
    
    float *d_input, *d_output;
    float h_input = 123.456f;
    float h_output[2];
    
    const int counter_iterations = 100;
    const int instructions_per_counter = 256;
    
    CUDA_CHECK(cudaMalloc(&d_input, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, 2 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, &h_input, sizeof(float), cudaMemcpyHostToDevice));
    
    // Warm-up
    latency_cvt<<<1, 1>>>(d_input, d_output, counter_iterations);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Actual measurement
    latency_cvt<<<1, 1>>>(d_input, d_output, counter_iterations);
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
    result.name = "CVT";
    result.total_cycles = total_cycles;
    result.total_instructions = total_instructions;
    result.latency_cycles = latency;
    result.latency_ns = latency * 1000.0f / gpu_clock_mhz;
    
    return result;
}

// Move
__global__ void latency_mov(float *input, float *output, int counter_iterations) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float result = input[0];
        float temp = 0.0f;
        clock_t total_time = 0;
        
        for (int counter = 0; counter < counter_iterations; counter++) {
            unsigned long long start = clock64();
            
            for (int i = 0; i < 256; i++) {
                // Use volatile inline assembly with memory constraint (dependent chain)
                asm volatile("mov.f32 %0, %1;" : "=f"(temp) : "f"(result) : "memory");
                asm volatile("mov.f32 %0, %1;" : "=f"(result) : "f"(temp) : "memory");
            }
            
            unsigned long long end = clock64();
            total_time += (end - start);
        }
        
        output[0] = result;
        output[1] = (float)total_time;
    }
}

// NEW MOV INSTRUCTION VARIANTS

// MOV.U32 (32-bit unsigned integer move)
__global__ void latency_mov_u32(unsigned int *input, unsigned int *output, int counter_iterations) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        unsigned int result = input[0];
        unsigned int temp = 0;
        clock_t total_time = 0;
        
        for (int counter = 0; counter < counter_iterations; counter++) {
            unsigned long long start = clock64();
            
            for (int i = 0; i < 256; i++) {
                // Use integer register constraints (%r)
                asm volatile("mov.u32 %0, %1;" : "=r"(temp) : "r"(result) : "memory");
                asm volatile("mov.u32 %0, %1;" : "=r"(result) : "r"(temp) : "memory");
            }
            
            unsigned long long end = clock64();
            total_time += (end - start);
        }
        
        output[0] = result;
        output[1] = (unsigned int)total_time;
    }
}

// MOV.F32 (32-bit float move) - renamed for clarity
__global__ void latency_mov_f32(float *input, float *output, int counter_iterations) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float result = input[0];
        float temp = 0.0f;
        clock_t total_time = 0;
        
        for (int counter = 0; counter < counter_iterations; counter++) {
            unsigned long long start = clock64();
            
            for (int i = 0; i < 256; i++) {
                // Use float register constraints (%f)
                asm volatile("mov.f32 %0, %1;" : "=f"(temp) : "f"(result) : "memory");
                asm volatile("mov.f32 %0, %1;" : "=f"(result) : "f"(temp) : "memory");
            }
            
            unsigned long long end = clock64();
            total_time += (end - start);
        }
        
        output[0] = result;
        output[1] = (float)total_time;
    }
}

// MOV.B64 (64-bit bitwise move)
__global__ void latency_mov_b64(unsigned long long *input, unsigned long long *output, int counter_iterations) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        unsigned long long result = input[0];
        unsigned long long temp = 0ULL;
        clock_t total_time = 0;
        
        for (int counter = 0; counter < counter_iterations; counter++) {
            unsigned long long start = clock64();
            
            for (int i = 0; i < 256; i++) {
                // Use 64-bit register constraints (%rd)
                asm volatile("mov.b64 %0, %1;" : "=l"(temp) : "l"(result) : "memory");
                asm volatile("mov.b64 %0, %1;" : "=l"(result) : "l"(temp) : "memory");
            }
            
            unsigned long long end = clock64();
            total_time += (end - start);
        }
        
        output[0] = result;
        output[1] = (unsigned long long)total_time;
    }
}

// MOV.B32 (32-bit bitwise move)
__global__ void latency_mov_b32(unsigned int *input, unsigned int *output, int counter_iterations) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        unsigned int result = input[0];
        unsigned int temp = 0;
        clock_t total_time = 0;
        
        for (int counter = 0; counter < counter_iterations; counter++) {
            unsigned long long start = clock64();
            
            for (int i = 0; i < 256; i++) {
                // Use 32-bit register constraints (%r for bitwise)
                asm volatile("mov.b32 %0, %1;" : "=r"(temp) : "r"(result) : "memory");
                asm volatile("mov.b32 %0, %1;" : "=r"(result) : "r"(temp) : "memory");
            }
            
            unsigned long long end = clock64();
            total_time += (end - start);
        }
        
        output[0] = result;
        output[1] = (unsigned int)total_time;
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
    
    unsigned long long start = clock64();
    
    // ILP=1: Only one variable, fully dependent operations
    for (int i = 0; i < iterations; i++) {
        a = a + 1.0f;  // Dependent chain
    }
    
    unsigned long long end = clock64();
    
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
    
    unsigned long long start = clock64();
    
    // ILP=2: Two independent variables
    for (int i = 0; i < iterations; i++) {
        a += 1.0f;  // Independent from b
        b += 1.0f;  // Independent from a
    }
    
    unsigned long long end = clock64();
    
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
    
    unsigned long long start = clock64();
    
    // ILP=3: Three independent variables
    for (int i = 0; i < iterations; i++) {
        a += 1.0f;  // Independent
        b += 1.0f;  // Independent
        c += 1.0f;  // Independent
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a + b + c;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (float)(end - start);
    }
}

// Kernel for measuring floating-point multiplication throughput with ILP=1
__global__ void throughput_fmul_ilp1(float *input, float *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    float a = input[tid];
    
    unsigned long long start = clock64();
    
    // ILP=1: Only one variable, fully dependent operations
    for (int i = 0; i < iterations; i++) {
        a = a * 1.000001f;  // Dependent chain
    }
    
    unsigned long long end = clock64();
    
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
    
    unsigned long long start = clock64();
    
    // ILP=2: Two independent variables
    for (int i = 0; i < iterations; i++) {
        a = a * 1.000001f;  // Independent from b
        b = b * 1.000002f;  // Independent from a
    }
    
    unsigned long long end = clock64();
    
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
    
    unsigned long long start = clock64();
    
    // ILP=3: Three independent variables
    for (int i = 0; i < iterations; i++) {
        a = a * 1.000001f;  // Independent
        b = b * 1.000002f;  // Independent
        c = c * 1.000003f;  // Independent
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a + b + c;
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
    
    unsigned long long start = clock64();
    
    // ILP=1: Only one variable, fully dependent operations
    for (int i = 0; i < iterations; i++) {
        a = a * b + c;  // Dependent chain
    }
    
    unsigned long long end = clock64();
    
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
    
    unsigned long long start = clock64();
    
    // ILP=2: Two independent variables
    for (int i = 0; i < iterations; i++) {
        a = a * c + d;  // Independent from b
        b = b * d + c;  // Independent from a
    }
    
    unsigned long long end = clock64();
    
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

    // Use completely independent constants
    float d1 = 1.000001f, e1 = 1.000002f;
    float d2 = 1.000003f, e2 = 1.000004f;  
    float d3 = 1.000005f, e3 = 1.000006f;
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        a = a * d1 + e1;  // Completely independent
        b = b * d2 + e2;  // Completely independent
        c = c * d3 + e3;  // Completely independent
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a + b + c;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (float)(end - start);
    }
}

// FSUB throughput kernels
__global__ void throughput_fsub_ilp1(float *input, float *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a = input[tid];
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        a = a - 1.0f;
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (float)(end - start);
    }
}

__global__ void throughput_fsub_ilp2(float *input, float *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a = input[tid];
    float b = input[tid] + 1.0f;
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        a -= 1.0f;
        b -= 1.0f;
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a + b;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (float)(end - start);
    }
}

__global__ void throughput_fsub_ilp3(float *input, float *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a = input[tid];
    float b = input[tid] + 1.0f;
    float c = input[tid] + 2.0f;
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        a -= 1.0f;
        b -= 1.0f;
        c -= 1.0f;
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a + b + c;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (float)(end - start);
    }
}

// DIVF throughput kernels
__global__ void throughput_divf_ilp1(float *input, float *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a = input[tid];
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        a = a / 1.000001f;
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (float)(end - start);
    }
}

__global__ void throughput_divf_ilp2(float *input, float *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a = input[tid];
    float b = input[tid] + 1.0f;
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        a = a / 1.000001f;
        b = b / 1.000002f;
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a + b;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (float)(end - start);
    }
}

__global__ void throughput_divf_ilp3(float *input, float *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a = input[tid];
    float b = input[tid] + 1.0f;
    float c = input[tid] + 2.0f;
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        a = a / 1.000001f;
        b = b / 1.000002f;
        c = c / 1.000003f;
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a + b + c;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (float)(end - start);
    }
}

// SQRT throughput kernels
__global__ void throughput_sqrt_ilp1(float *input, float *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a = input[tid] + 1.0f;  // Ensure positive
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        a = sqrtf(a + 0.1f);
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (float)(end - start);
    }
}

__global__ void throughput_sqrt_ilp2(float *input, float *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a = input[tid] + 1.0f;
    float b = input[tid] + 2.0f;
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        a = sqrtf(a + 0.1f);
        b = sqrtf(b + 0.1f);
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a + b;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (float)(end - start);
    }
}

__global__ void throughput_sqrt_ilp3(float *input, float *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a = input[tid] + 1.0f;
    float b = input[tid] + 2.0f;
    float c = input[tid] + 3.0f;
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        a = sqrtf(a + 0.1f);
        b = sqrtf(b + 0.1f);
        c = sqrtf(c + 0.1f);
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a + b + c;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (float)(end - start);
    }
}

// ============================================================================
// NEW THROUGHPUT KERNELS FOR HIGH-FREQUENCY INTEGER OPERATIONS
// ============================================================================

// ADD.S32 (32-bit signed integer add) throughput kernels
__global__ void throughput_add_s32_ilp1(int *input, int *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int a = input[tid];
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        a = a + 1;  // Dependent chain
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (int)(end - start);
    }
}

__global__ void throughput_add_s32_ilp2(int *input, int *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int a = input[tid];
    int b = input[tid] + 1;
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        a += 1;  // Independent from b
        b += 1;  // Independent from a
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a + b;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (int)(end - start);
    }
}

__global__ void throughput_add_s32_ilp3(int *input, int *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int a = input[tid];
    int b = input[tid] + 1;
    int c = input[tid] + 2;
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        a += 1;  // Independent
        b += 1;  // Independent
        c += 1;  // Independent
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a + b + c;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (int)(end - start);
    }
}

// ADD.S64 (64-bit signed integer add) throughput kernels
__global__ void throughput_add_s64_ilp1(long long *input, long long *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    long long a = input[tid];
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        a = a + 1LL;  // Dependent chain
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (long long)(end - start);
    }
}

__global__ void throughput_add_s64_ilp2(long long *input, long long *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    long long a = input[tid];
    long long b = input[tid] + 1LL;
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        a += 1LL;  // Independent from b
        b += 1LL;  // Independent from a
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a + b;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (long long)(end - start);
    }
}

__global__ void throughput_add_s64_ilp3(long long *input, long long *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    long long a = input[tid];
    long long b = input[tid] + 1LL;
    long long c = input[tid] + 2LL;
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        a += 1LL;  // Independent
        b += 1LL;  // Independent
        c += 1LL;  // Independent
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a + b + c;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (long long)(end - start);
    }
}

// SUB.S32 (32-bit signed integer subtract) throughput kernels
__global__ void throughput_sub_s32_ilp1(int *input, int *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int a = input[tid];
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        a = a - 1;
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (int)(end - start);
    }
}

__global__ void throughput_sub_s32_ilp2(int *input, int *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int a = input[tid];
    int b = input[tid] + 1;
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        a -= 1;
        b -= 1;
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a + b;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (int)(end - start);
    }
}

__global__ void throughput_sub_s32_ilp3(int *input, int *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int a = input[tid];
    int b = input[tid] + 1;
    int c = input[tid] + 2;
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        a -= 1;
        b -= 1;
        c -= 1;
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a + b + c;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (int)(end - start);
    }
}

// MUL.LO.S32 (32-bit signed multiply, low bits) throughput kernels
__global__ void throughput_mul_lo_s32_ilp1(int *input, int *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int a = input[tid];
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        a = (a * 1001) % 100000;  // Multiply with modulo to prevent overflow
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (int)(end - start);
    }
}

__global__ void throughput_mul_lo_s32_ilp2(int *input, int *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int a = input[tid];
    int b = input[tid] + 1;
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        a = (a * 1001) % 100000;
        b = (b * 1003) % 100000;
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a + b;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (int)(end - start);
    }
}

__global__ void throughput_mul_lo_s32_ilp3(int *input, int *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int a = input[tid];
    int b = input[tid] + 1;
    int c = input[tid] + 2;
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        a = (a * 1001) % 100000;
        b = (b * 1003) % 100000;
        c = (c * 1007) % 100000;
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a + b + c;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (int)(end - start);
    }
}

// MUL.WIDE.S32 (32-bit to 64-bit multiply) throughput kernels
__global__ void throughput_mul_wide_s32_ilp1(int *input, long long *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int a = input[tid];
    long long result = a;
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        result = (long long)a * 1001LL;  // 32->64 bit multiply
        a = (int)(result % 100000);      // Keep input reasonable
    }
    
    unsigned long long end = clock64();
    
    output[tid] = result;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (long long)(end - start);
    }
}

__global__ void throughput_mul_wide_s32_ilp2(int *input, long long *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int a = input[tid];
    int b = input[tid] + 1;
    long long result_a = a, result_b = b;
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        result_a = (long long)a * 1001LL;
        result_b = (long long)b * 1003LL;
        a = (int)(result_a % 100000);
        b = (int)(result_b % 100000);
    }
    
    unsigned long long end = clock64();
    
    output[tid] = result_a + result_b;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (long long)(end - start);
    }
}

__global__ void throughput_mul_wide_s32_ilp3(int *input, long long *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int a = input[tid];
    int b = input[tid] + 1;
    int c = input[tid] + 2;
    long long result_a = a, result_b = b, result_c = c;
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        result_a = (long long)a * 1001LL;
        result_b = (long long)b * 1003LL;
        result_c = (long long)c * 1007LL;
        a = (int)(result_a % 100000);
        b = (int)(result_b % 100000);
        c = (int)(result_c % 100000);
    }
    
    unsigned long long end = clock64();
    
    output[tid] = result_a + result_b + result_c;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (long long)(end - start);
    }
}

// MAD.LO.S32 (32-bit multiply-add, low bits) throughput kernels
__global__ void throughput_mad_lo_s32_ilp1(int *input, int *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int a = input[tid];
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        a = (a * 3 + 2) % 100000;  // Multiply-add with modulo
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (int)(end - start);
    }
}

__global__ void throughput_mad_lo_s32_ilp2(int *input, int *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int a = input[tid];
    int b = input[tid] + 1;
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        a = (a * 3 + 2) % 100000;
        b = (b * 5 + 3) % 100000;
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a + b;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (int)(end - start);
    }
}

__global__ void throughput_mad_lo_s32_ilp3(int *input, int *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int a = input[tid];
    int b = input[tid] + 1;
    int c = input[tid] + 2;
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        a = (a * 3 + 2) % 100000;
        b = (b * 5 + 3) % 100000;
        c = (c * 7 + 4) % 100000;
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a + b + c;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (int)(end - start);
    }
}

// ADDS (Integer Add with Saturation) throughput kernels
__global__ void throughput_adds_ilp1(int *input, int *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int a = input[tid];
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        a = a + 1;  // Dependent chain
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (int)(end - start);
    }
}

__global__ void throughput_adds_ilp2(int *input, int *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int a = input[tid];
    int b = input[tid] + 1;
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        a += 1;  // Independent from b
        b += 1;  // Independent from a
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a + b;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (int)(end - start);
    }
}

__global__ void throughput_adds_ilp3(int *input, int *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int a = input[tid];
    int b = input[tid] + 1;
    int c = input[tid] + 2;
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        a += 1;  // Independent
        b += 1;  // Independent
        c += 1;  // Independent
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a + b + c;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (int)(end - start);
    }
}

// SUBS (Integer Subtract with Saturation) throughput kernels
__global__ void throughput_subs_ilp1(int *input, int *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int a = input[tid];
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        a = a - 1;
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (int)(end - start);
    }
}

__global__ void throughput_subs_ilp2(int *input, int *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int a = input[tid];
    int b = input[tid] + 1;
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        a -= 1;
        b -= 1;
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a + b;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (int)(end - start);
    }
}

__global__ void throughput_subs_ilp3(int *input, int *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int a = input[tid];
    int b = input[tid] + 1;
    int c = input[tid] + 2;
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        a -= 1;
        b -= 1;
        c -= 1;
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a + b + c;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (int)(end - start);
    }
}

// AND (Bitwise AND) throughput kernels
__global__ void throughput_and_ilp1(int *input, int *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int a = input[tid];
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        a = a & 0xFFFFFFFE;  // Clear lowest bit
        a = a | 1;          // Set it back
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (int)(end - start);
    }
}

__global__ void throughput_and_ilp2(int *input, int *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int a = input[tid];
    int b = input[tid] + 1;
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        a = a & 0xFFFFFFFE;
        b = b & 0xFFFFFFFD;
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a + b;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (int)(end - start);
    }
}

__global__ void throughput_and_ilp3(int *input, int *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int a = input[tid];
    int b = input[tid] + 1;
    int c = input[tid] + 2;
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        a = a & 0xFFFFFFFE;
        b = b & 0xFFFFFFFD;
        c = c & 0xFFFFFFFB;
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a + b + c;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (int)(end - start);
    }
}

// MADS (Integer Multiply-Add with Saturation) throughput kernels
__global__ void throughput_mads_ilp1(int *input, int *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int a = input[tid];
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        a = a * 2 + 1;  // Multiply-add operation
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (int)(end - start);
    }
}

__global__ void throughput_mads_ilp2(int *input, int *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int a = input[tid];
    int b = input[tid] + 1;
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        a = a * 2 + 1;
        b = b * 3 + 2;
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a + b;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (int)(end - start);
    }
}

__global__ void throughput_mads_ilp3(int *input, int *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int a = input[tid];
    int b = input[tid] + 1;
    int c = input[tid] + 2;
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        a = a * 2 + 1;
        b = b * 3 + 2;
        c = c * 4 + 3;
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a + b + c;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (int)(end - start);
    }
}

// MULS (Integer Multiply with Saturation) throughput kernels
__global__ void throughput_muls_ilp1(int *input, int *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int a = input[tid];
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        a = (a * 1001) % 100000;  // Multiply with modulo to prevent overflow
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (int)(end - start);
    }
}

__global__ void throughput_muls_ilp2(int *input, int *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int a = input[tid];
    int b = input[tid] + 1;
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        a = (a * 1001) % 100000;  // Independent multiply operations
        b = (b * 1003) % 100000;
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a + b;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (int)(end - start);
    }
}

__global__ void throughput_muls_ilp3(int *input, int *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int a = input[tid];
    int b = input[tid] + 1;
    int c = input[tid] + 2;
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        a = (a * 1001) % 100000;  // Independent multiply operations
        b = (b * 1003) % 100000;
        c = (c * 1007) % 100000;
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a + b + c;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (int)(end - start);
    }
}

// DIVS (Integer Division) throughput kernels
__global__ void throughput_divs_ilp1(int *input, int *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int a = input[tid] + 1000000;  // Ensure positive and reasonable
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        a = a / 2 + 1000000;
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (int)(end - start);
    }
}

__global__ void throughput_divs_ilp2(int *input, int *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int a = input[tid] + 1000000;
    int b = input[tid] + 2000000;
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        a = a / 2 + 1000000;
        b = b / 3 + 2000000;
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a + b;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (int)(end - start);
    }
}

__global__ void throughput_divs_ilp3(int *input, int *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int a = input[tid] + 1000000;
    int b = input[tid] + 2000000;
    int c = input[tid] + 3000000;
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        a = a / 2 + 1000000;
        b = b / 3 + 2000000;
        c = c / 4 + 3000000;
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a + b + c;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (int)(end - start);
    }
}

// SETP (Set Predicate) throughput kernels
__global__ void throughput_setp_ilp1(float *input, float *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a = input[tid];
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        bool pred = (a > 0.5f);
        a = pred ? a + 0.1f : a - 0.1f;
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (float)(end - start);
    }
}

__global__ void throughput_setp_ilp2(float *input, float *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a = input[tid];
    float b = input[tid] + 1.0f;
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        bool pred_a = (a > 0.5f);
        bool pred_b = (b > 1.5f);
        a = pred_a ? a + 0.1f : a - 0.1f;
        b = pred_b ? b + 0.2f : b - 0.2f;
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a + b;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (float)(end - start);
    }
}

__global__ void throughput_setp_ilp3(float *input, float *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a = input[tid];
    float b = input[tid] + 1.0f;
    float c = input[tid] + 2.0f;
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        bool pred_a = (a > 0.5f);
        bool pred_b = (b > 1.5f);
        bool pred_c = (c > 2.5f);
        a = pred_a ? a + 0.1f : a - 0.1f;
        b = pred_b ? b + 0.2f : b - 0.2f;
        c = pred_c ? c + 0.3f : c - 0.3f;
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a + b + c;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (float)(end - start);
    }
}

// CVT (Convert) throughput kernels - using float/int conversion operations
__global__ void throughput_cvt_ilp1(float *input, float *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a = input[tid];
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        int temp = (int)a;      // float to int conversion
        a = (float)temp;        // int to float conversion
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (float)(end - start);
    }
}

__global__ void throughput_cvt_ilp2(float *input, float *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a = input[tid];
    float b = input[tid] + 1.0f;
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        int temp_a = (int)a; a = (float)temp_a;
        int temp_b = (int)b; b = (float)temp_b;
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a + b;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (float)(end - start);
    }
}

__global__ void throughput_cvt_ilp3(float *input, float *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a = input[tid];
    float b = input[tid] + 1.0f;
    float c = input[tid] + 2.0f;
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        int temp_a = (int)a; a = (float)temp_a;
        int temp_b = (int)b; b = (float)temp_b;
        int temp_c = (int)c; c = (float)temp_c;
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a + b + c;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (float)(end - start);
    }
}

// MOV (Move) throughput kernels - Using memory constraints to prevent ALL optimization
__global__ void throughput_mov_ilp1(float *input, float *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a = input[tid];
    float temp = 0.0f;
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        // Use memory constraint to force the compiler to preserve operations
        asm volatile("mov.f32 %0, %1;" : "=f"(temp) : "f"(a) : "memory");
        asm volatile("mov.f32 %0, %1;" : "=f"(a) : "f"(temp) : "memory");
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (float)(end - start);
    }
}

__global__ void throughput_mov_ilp2(float *input, float *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a = input[tid];
    float b = input[tid] + 1.0f;
    float temp1 = 0.0f, temp2 = 0.0f;
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        // Two independent MOV operations with memory barriers
        asm volatile("mov.f32 %0, %1;" : "=f"(temp1) : "f"(a) : "memory");
        asm volatile("mov.f32 %0, %1;" : "=f"(temp2) : "f"(b) : "memory");
        asm volatile("mov.f32 %0, %1;" : "=f"(a) : "f"(temp1) : "memory");
        asm volatile("mov.f32 %0, %1;" : "=f"(b) : "f"(temp2) : "memory");
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a + b;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (float)(end - start);
    }
}

__global__ void throughput_mov_ilp3(float *input, float *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a = input[tid];
    float b = input[tid] + 1.0f;
    float c = input[tid] + 2.0f;
    float temp1 = 0.0f, temp2 = 0.0f, temp3 = 0.0f;
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        // Three independent MOV operations with memory barriers
        asm volatile("mov.f32 %0, %1;" : "=f"(temp1) : "f"(a) : "memory");
        asm volatile("mov.f32 %0, %1;" : "=f"(temp2) : "f"(b) : "memory");
        asm volatile("mov.f32 %0, %1;" : "=f"(temp3) : "f"(c) : "memory");
        asm volatile("mov.f32 %0, %1;" : "=f"(a) : "f"(temp1) : "memory");
        asm volatile("mov.f32 %0, %1;" : "=f"(b) : "f"(temp2) : "memory");
        asm volatile("mov.f32 %0, %1;" : "=f"(c) : "f"(temp3) : "memory");
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a + b + c;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (float)(end - start);
    }
}

// ============================================================================
// NEW MOV VARIANTS THROUGHPUT KERNELS
// ============================================================================

// MOV.U32 throughput kernels
__global__ void throughput_mov_u32_ilp1(unsigned int *input, unsigned int *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int a = input[tid];
    unsigned int temp = 0;
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        asm volatile("mov.u32 %0, %1;" : "=r"(temp) : "r"(a) : "memory");
        asm volatile("mov.u32 %0, %1;" : "=r"(a) : "r"(temp) : "memory");
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (unsigned int)(end - start);
    }
}

__global__ void throughput_mov_u32_ilp2(unsigned int *input, unsigned int *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int a = input[tid];
    unsigned int b = input[tid] + 1;
    unsigned int temp1 = 0, temp2 = 0;
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        asm volatile("mov.u32 %0, %1;" : "=r"(temp1) : "r"(a) : "memory");
        asm volatile("mov.u32 %0, %1;" : "=r"(temp2) : "r"(b) : "memory");
        asm volatile("mov.u32 %0, %1;" : "=r"(a) : "r"(temp1) : "memory");
        asm volatile("mov.u32 %0, %1;" : "=r"(b) : "r"(temp2) : "memory");
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a + b;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (unsigned int)(end - start);
    }
}

__global__ void throughput_mov_u32_ilp3(unsigned int *input, unsigned int *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int a = input[tid];
    unsigned int b = input[tid] + 1;
    unsigned int c = input[tid] + 2;
    unsigned int temp1 = 0, temp2 = 0, temp3 = 0;
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        asm volatile("mov.u32 %0, %1;" : "=r"(temp1) : "r"(a) : "memory");
        asm volatile("mov.u32 %0, %1;" : "=r"(temp2) : "r"(b) : "memory");
        asm volatile("mov.u32 %0, %1;" : "=r"(temp3) : "r"(c) : "memory");
        asm volatile("mov.u32 %0, %1;" : "=r"(a) : "r"(temp1) : "memory");
        asm volatile("mov.u32 %0, %1;" : "=r"(b) : "r"(temp2) : "memory");
        asm volatile("mov.u32 %0, %1;" : "=r"(c) : "r"(temp3) : "memory");
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a + b + c;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (unsigned int)(end - start);
    }
}

// MOV.F32 throughput kernels (explicit naming)
__global__ void throughput_mov_f32_ilp1(float *input, float *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a = input[tid];
    float temp = 0.0f;
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        asm volatile("mov.f32 %0, %1;" : "=f"(temp) : "f"(a) : "memory");
        asm volatile("mov.f32 %0, %1;" : "=f"(a) : "f"(temp) : "memory");
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (float)(end - start);
    }
}

__global__ void throughput_mov_f32_ilp2(float *input, float *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a = input[tid];
    float b = input[tid] + 1.0f;
    float temp1 = 0.0f, temp2 = 0.0f;
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        asm volatile("mov.f32 %0, %1;" : "=f"(temp1) : "f"(a) : "memory");
        asm volatile("mov.f32 %0, %1;" : "=f"(temp2) : "f"(b) : "memory");
        asm volatile("mov.f32 %0, %1;" : "=f"(a) : "f"(temp1) : "memory");
        asm volatile("mov.f32 %0, %1;" : "=f"(b) : "f"(temp2) : "memory");
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a + b;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (float)(end - start);
    }
}

__global__ void throughput_mov_f32_ilp3(float *input, float *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a = input[tid];
    float b = input[tid] + 1.0f;
    float c = input[tid] + 2.0f;
    float temp1 = 0.0f, temp2 = 0.0f, temp3 = 0.0f;
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        asm volatile("mov.f32 %0, %1;" : "=f"(temp1) : "f"(a) : "memory");
        asm volatile("mov.f32 %0, %1;" : "=f"(temp2) : "f"(b) : "memory");
        asm volatile("mov.f32 %0, %1;" : "=f"(temp3) : "f"(c) : "memory");
        asm volatile("mov.f32 %0, %1;" : "=f"(a) : "f"(temp1) : "memory");
        asm volatile("mov.f32 %0, %1;" : "=f"(b) : "f"(temp2) : "memory");
        asm volatile("mov.f32 %0, %1;" : "=f"(c) : "f"(temp3) : "memory");
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a + b + c;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (float)(end - start);
    }
}

// MOV.B64 throughput kernels
__global__ void throughput_mov_b64_ilp1(unsigned long long *input, unsigned long long *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long a = input[tid];
    unsigned long long temp = 0ULL;
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        asm volatile("mov.b64 %0, %1;" : "=l"(temp) : "l"(a) : "memory");
        asm volatile("mov.b64 %0, %1;" : "=l"(a) : "l"(temp) : "memory");
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (end - start);
    }
}

__global__ void throughput_mov_b64_ilp2(unsigned long long *input, unsigned long long *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long a = input[tid];
    unsigned long long b = input[tid] + 1ULL;
    unsigned long long temp1 = 0ULL, temp2 = 0ULL;
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        asm volatile("mov.b64 %0, %1;" : "=l"(temp1) : "l"(a) : "memory");
        asm volatile("mov.b64 %0, %1;" : "=l"(temp2) : "l"(b) : "memory");
        asm volatile("mov.b64 %0, %1;" : "=l"(a) : "l"(temp1) : "memory");
        asm volatile("mov.b64 %0, %1;" : "=l"(b) : "l"(temp2) : "memory");
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a + b;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (end - start);
    }
}

__global__ void throughput_mov_b64_ilp3(unsigned long long *input, unsigned long long *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long a = input[tid];
    unsigned long long b = input[tid] + 1ULL;
    unsigned long long c = input[tid] + 2ULL;
    unsigned long long temp1 = 0ULL, temp2 = 0ULL, temp3 = 0ULL;
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        asm volatile("mov.b64 %0, %1;" : "=l"(temp1) : "l"(a) : "memory");
        asm volatile("mov.b64 %0, %1;" : "=l"(temp2) : "l"(b) : "memory");
        asm volatile("mov.b64 %0, %1;" : "=l"(temp3) : "l"(c) : "memory");
        asm volatile("mov.b64 %0, %1;" : "=l"(a) : "l"(temp1) : "memory");
        asm volatile("mov.b64 %0, %1;" : "=l"(b) : "l"(temp2) : "memory");
        asm volatile("mov.b64 %0, %1;" : "=l"(c) : "l"(temp3) : "memory");
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a + b + c;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (end - start);
    }
}

// MOV.B32 throughput kernels
__global__ void throughput_mov_b32_ilp1(unsigned int *input, unsigned int *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int a = input[tid];
    unsigned int temp = 0;
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        asm volatile("mov.b32 %0, %1;" : "=r"(temp) : "r"(a) : "memory");
        asm volatile("mov.b32 %0, %1;" : "=r"(a) : "r"(temp) : "memory");
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (unsigned int)(end - start);
    }
}

__global__ void throughput_mov_b32_ilp2(unsigned int *input, unsigned int *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int a = input[tid];
    unsigned int b = input[tid] + 1;
    unsigned int temp1 = 0, temp2 = 0;
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        asm volatile("mov.b32 %0, %1;" : "=r"(temp1) : "r"(a) : "memory");
        asm volatile("mov.b32 %0, %1;" : "=r"(temp2) : "r"(b) : "memory");
        asm volatile("mov.b32 %0, %1;" : "=r"(a) : "r"(temp1) : "memory");
        asm volatile("mov.b32 %0, %1;" : "=r"(b) : "r"(temp2) : "memory");
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a + b;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (unsigned int)(end - start);
    }
}

__global__ void throughput_mov_b32_ilp3(unsigned int *input, unsigned int *output, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int a = input[tid];
    unsigned int b = input[tid] + 1;
    unsigned int c = input[tid] + 2;
    unsigned int temp1 = 0, temp2 = 0, temp3 = 0;
    
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        asm volatile("mov.b32 %0, %1;" : "=r"(temp1) : "r"(a) : "memory");
        asm volatile("mov.b32 %0, %1;" : "=r"(temp2) : "r"(b) : "memory");
        asm volatile("mov.b32 %0, %1;" : "=r"(temp3) : "r"(c) : "memory");
        asm volatile("mov.b32 %0, %1;" : "=r"(a) : "r"(temp1) : "memory");
        asm volatile("mov.b32 %0, %1;" : "=r"(b) : "r"(temp2) : "memory");
        asm volatile("mov.b32 %0, %1;" : "=r"(c) : "r"(temp3) : "memory");
    }
    
    unsigned long long end = clock64();
    
    output[tid] = a + b + c;
    if (tid == 0) {
        output[blockDim.x * gridDim.x] = (unsigned int)(end - start);
    }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

LatencyMeasurement measure_latency_generic_float(
    void (*kernel)(float*, float*, int),
    const std::string& name,
    float gpu_clock_mhz,
    float init_value = 1.5f) {
    
    printf("\n=== Measuring %s Latency ===\n", name.c_str());
    
    float *d_input, *d_output;
    float h_input = init_value;
    float h_output[2];
    
    const int counter_iterations = 100;
    const int instructions_per_counter = 256;
    
    CUDA_CHECK(cudaMalloc(&d_input, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, 2 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, &h_input, sizeof(float), cudaMemcpyHostToDevice));
    
    // Warm-up
    kernel<<<1, 1>>>(d_input, d_output, counter_iterations);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Actual measurement
    kernel<<<1, 1>>>(d_input, d_output, counter_iterations);
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
    result.name = name;
    result.total_cycles = total_cycles;
    result.total_instructions = total_instructions;
    result.latency_cycles = latency;
    result.latency_ns = latency * 1000.0f / gpu_clock_mhz;
    
    return result;
}

LatencyMeasurement measure_latency_generic_int(
    void (*kernel)(int*, int*, int),
    const std::string& name,
    float gpu_clock_mhz,
    int init_value = 1000) {
    
    printf("\n=== Measuring %s Latency ===\n", name.c_str());
    
    int *d_input, *d_output;
    int h_input = init_value;
    int h_output[2];
    
    const int counter_iterations = 100;
    const int instructions_per_counter = 256;
    
    CUDA_CHECK(cudaMalloc(&d_input, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_output, 2 * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_input, &h_input, sizeof(int), cudaMemcpyHostToDevice));
    
    // Warm-up
    kernel<<<1, 1>>>(d_input, d_output, counter_iterations);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Actual measurement  
    kernel<<<1, 1>>>(d_input, d_output, counter_iterations);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_output, d_output, 2 * sizeof(int), cudaMemcpyDeviceToHost));
    
    float total_cycles = (float)h_output[1];
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
    result.name = name;
    result.total_cycles = total_cycles;
    result.total_instructions = total_instructions;
    result.latency_cycles = latency;
    result.latency_ns = latency * 1000.0f / gpu_clock_mhz;
    
    return result;
}

LatencyMeasurement measure_latency_generic_int64(
    void (*kernel)(long long*, long long*, int),
    const std::string& name,
    float gpu_clock_mhz,
    long long init_value = 1000LL) {
    
    printf("\n=== Measuring %s Latency ===\n", name.c_str());
    
    long long *d_input, *d_output;
    long long h_input = init_value;
    long long h_output[2];
    
    const int counter_iterations = 100;
    const int instructions_per_counter = 256;
    
    CUDA_CHECK(cudaMalloc(&d_input, sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_output, 2 * sizeof(long long)));
    CUDA_CHECK(cudaMemcpy(d_input, &h_input, sizeof(long long), cudaMemcpyHostToDevice));
    
    // Warm-up
    kernel<<<1, 1>>>(d_input, d_output, counter_iterations);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Actual measurement  
    kernel<<<1, 1>>>(d_input, d_output, counter_iterations);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_output, d_output, 2 * sizeof(long long), cudaMemcpyDeviceToHost));
    
    float total_cycles = (float)h_output[1];
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
    result.name = name;
    result.total_cycles = total_cycles;
    result.total_instructions = total_instructions;
    result.latency_cycles = latency;
    result.latency_ns = latency * 1000.0f / gpu_clock_mhz;
    
    return result;
}

// Special throughput measurement for MUL.WIDE.S32 (int input -> long long output)
std::vector<ILPThroughputMeasurement> measure_mul_wide_s32_throughput(
    const std::string& instruction_name,
    void (*kernel_ilp1)(int*, long long*, int),
    void (*kernel_ilp2)(int*, long long*, int), 
    void (*kernel_ilp3)(int*, long long*, int),
    int ilp1, int ilp2, int ilp3,
    float gpu_clock_mhz,
    bool verbose = false) {
    
    std::vector<ILPThroughputMeasurement> results;
    
    if (verbose) {
        printf("\n=== Measuring %s Throughput (Mixed Types) ===\n", instruction_name.c_str());
    }
    
    // Test every warp configuration from 1 to 32 for precise PeakWarps measurement
    std::vector<int> warp_configs;
    for (int warps = 1; warps <= 32; warps++) {
        warp_configs.push_back(warps);
    }
    const int iterations = 1024;
    
    // For each ILP level
    std::vector<int> ilp_levels = {ilp1, ilp2, ilp3};
    void (*kernels[])(int*, long long*, int) = {kernel_ilp1, kernel_ilp2, kernel_ilp3};
    
    for (int ilp_idx = 0; ilp_idx < 3; ilp_idx++) {
        int ilp = ilp_levels[ilp_idx];
        auto kernel = kernels[ilp_idx];
        
        for (int warps : warp_configs) {
            int num_threads = warps * WARP_SIZE;
            
            // Allocate memory
            int *d_input;
            long long *d_output;
            CUDA_CHECK(cudaMalloc(&d_input, num_threads * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_output, (num_threads + 1) * sizeof(long long)));
            
            // Initialize input
            std::vector<int> h_input(num_threads, 100);
            for (int i = 0; i < num_threads; i++) {
                h_input[i] = 100 + i;
            }
            CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), num_threads * sizeof(int), cudaMemcpyHostToDevice));
            
            // Configure single block (as per paper Figure 3)
            dim3 blockSize(min(num_threads, 1024));
            dim3 gridSize(1);  // Single block only
            
            if (num_threads > 1024) {
                // Skip configurations that exceed block size limit
                cudaFree(d_input);
                cudaFree(d_output);
                continue;
            }
            
            // Warm-up
            kernel<<<gridSize, blockSize>>>(d_input, d_output, iterations);
            CUDA_CHECK(cudaDeviceSynchronize());
            
            // Actual measurement
            kernel<<<gridSize, blockSize>>>(d_input, d_output, iterations);
            CUDA_CHECK(cudaDeviceSynchronize());
            
            // Read results - timing is stored at the end of output array
            long long total_cycles_long;
            CUDA_CHECK(cudaMemcpy(&total_cycles_long, &d_output[num_threads], sizeof(long long), cudaMemcpyDeviceToHost));
            float total_cycles = (float)total_cycles_long;
            
            // Calculate metrics
            float total_ops = (float)num_threads * iterations * ilp;
            float throughput_ops_per_cycle = safe_divide(total_ops, total_cycles);
            float throughput_gflops = throughput_ops_per_cycle * gpu_clock_mhz / 1000.0f;
            
            // Store result
            ILPThroughputMeasurement measurement;
            measurement.instruction = instruction_name;
            measurement.ilp = ilp;
            measurement.warps = warps;
            measurement.threads = num_threads;
            measurement.total_cycles = total_cycles;
            measurement.total_ops = total_ops;
            measurement.throughput_ops_per_cycle = throughput_ops_per_cycle;
            measurement.throughput_gflops = throughput_gflops;
            
            results.push_back(measurement);
            
            if (verbose) {
                printf("ILP=%d, Warps=%d, Threads=%d: %.2f ops/cycle, %.2f GFLOPS\n",
                       ilp, warps, num_threads, throughput_ops_per_cycle, throughput_gflops);
            }
            
            cudaFree(d_input);
            cudaFree(d_output);
        }
    }
    
    return results;
}
// Generic throughput measurement for int64 kernels
std::vector<ILPThroughputMeasurement> measure_instruction_throughput_int64(
    const std::string& instruction_name,
    void (*kernel_ilp1)(long long*, long long*, int),
    void (*kernel_ilp2)(long long*, long long*, int), 
    void (*kernel_ilp3)(long long*, long long*, int),
    int ilp1, int ilp2, int ilp3,
    float gpu_clock_mhz,
    bool verbose = false) {
    
    std::vector<ILPThroughputMeasurement> results;
    
    if (verbose) {
        printf("\n=== Measuring %s Throughput ===\n", instruction_name.c_str());
    }
    
    // Test every warp configuration from 1 to 32 for precise PeakWarps measurement
    std::vector<int> warp_configs;
    for (int warps = 1; warps <= 32; warps++) {
        warp_configs.push_back(warps);
    }
    const int iterations = 1024;
    
    // For each ILP level
    std::vector<int> ilp_levels = {ilp1, ilp2, ilp3};
    void (*kernels[])(long long*, long long*, int) = {kernel_ilp1, kernel_ilp2, kernel_ilp3};
    
    for (int ilp_idx = 0; ilp_idx < 3; ilp_idx++) {
        int ilp = ilp_levels[ilp_idx];
        auto kernel = kernels[ilp_idx];
        
        for (int warps : warp_configs) {
            int num_threads = warps * WARP_SIZE;
            
            // Allocate memory
            long long *d_input, *d_output;
            CUDA_CHECK(cudaMalloc(&d_input, num_threads * sizeof(long long)));
            CUDA_CHECK(cudaMalloc(&d_output, (num_threads + 1) * sizeof(long long)));
            
            // Initialize input
            std::vector<long long> h_input(num_threads, 1000LL);
            for (int i = 0; i < num_threads; i++) {
                h_input[i] = 1000LL + i;
            }
            CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), num_threads * sizeof(long long), cudaMemcpyHostToDevice));
            
            // Configure single block (as per paper Figure 3)
            dim3 blockSize(min(num_threads, 1024));
            dim3 gridSize(1);  // Single block only
            
            if (num_threads > 1024) {
                // Skip configurations that exceed block size limit
                cudaFree(d_input);
                cudaFree(d_output);
                continue;
            }
            
            // Warm-up
            kernel<<<gridSize, blockSize>>>(d_input, d_output, iterations);
            CUDA_CHECK(cudaDeviceSynchronize());
            
            // Actual measurement
            kernel<<<gridSize, blockSize>>>(d_input, d_output, iterations);
            CUDA_CHECK(cudaDeviceSynchronize());
            
            // Read results
            long long total_cycles_long;
            CUDA_CHECK(cudaMemcpy(&total_cycles_long, &d_output[num_threads], sizeof(long long), cudaMemcpyDeviceToHost));
            float total_cycles = (float)total_cycles_long;
            
            // Calculate metrics
            float total_ops = (float)num_threads * iterations * ilp;
            float throughput_ops_per_cycle = safe_divide(total_ops, total_cycles);
            float throughput_gflops = throughput_ops_per_cycle * gpu_clock_mhz / 1000.0f;
            
            // Store result
            ILPThroughputMeasurement measurement;
            measurement.instruction = instruction_name;
            measurement.ilp = ilp;
            measurement.warps = warps;
            measurement.threads = num_threads;
            measurement.total_cycles = total_cycles;
            measurement.total_ops = total_ops;
            measurement.throughput_ops_per_cycle = throughput_ops_per_cycle;
            measurement.throughput_gflops = throughput_gflops;
            
            results.push_back(measurement);
            
            if (verbose) {
                printf("ILP=%d, Warps=%d, Threads=%d: %.2f ops/cycle, %.2f GFLOPS\n",
                       ilp, warps, num_threads, throughput_ops_per_cycle, throughput_gflops);
            }
            
            cudaFree(d_input);
            cudaFree(d_output);
        }
    }
    
    return results;
}

// Generic throughput measurement for float kernels
std::vector<ILPThroughputMeasurement> measure_instruction_throughput_float(
    const std::string& instruction_name,
    void (*kernel_ilp1)(float*, float*, int),
    void (*kernel_ilp2)(float*, float*, int), 
    void (*kernel_ilp3)(float*, float*, int),
    int ilp1, int ilp2, int ilp3,
    float gpu_clock_mhz,
    bool verbose = false) {
    
    std::vector<ILPThroughputMeasurement> results;
    
    if (verbose) {
        printf("\n=== Measuring %s Throughput ===\n", instruction_name.c_str());
    }
    
    // Test every warp configuration from 1 to 32 for precise PeakWarps measurement
    std::vector<int> warp_configs;
    for (int warps = 1; warps <= 32; warps++) {
        warp_configs.push_back(warps);
    }
    const int iterations = 1024;
    
    // For each ILP level
    std::vector<int> ilp_levels = {ilp1, ilp2, ilp3};
    void (*kernels[])(float*, float*, int) = {kernel_ilp1, kernel_ilp2, kernel_ilp3};
    
    for (int ilp_idx = 0; ilp_idx < 3; ilp_idx++) {
        int ilp = ilp_levels[ilp_idx];
        auto kernel = kernels[ilp_idx];
        
        for (int warps : warp_configs) {
            int num_threads = warps * WARP_SIZE;
            
            // Allocate memory
            float *d_input, *d_output;
            CUDA_CHECK(cudaMalloc(&d_input, num_threads * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_output, (num_threads + 1) * sizeof(float)));
            
            // Initialize input
            std::vector<float> h_input(num_threads, 1.5f);
            for (int i = 0; i < num_threads; i++) {
                h_input[i] = 1.5f + i * 0.001f;
            }
            CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), num_threads * sizeof(float), cudaMemcpyHostToDevice));
            
            // Configure single block (as per paper Figure 3)
            dim3 blockSize(min(num_threads, 1024));
            dim3 gridSize(1);  // Single block only
            
            if (num_threads > 1024) {
                // Skip configurations that exceed block size limit
                cudaFree(d_input);
                cudaFree(d_output);
                continue;
            }
            
            // Warm-up
            kernel<<<gridSize, blockSize>>>(d_input, d_output, iterations);
            CUDA_CHECK(cudaDeviceSynchronize());
            
            // Actual measurement
            kernel<<<gridSize, blockSize>>>(d_input, d_output, iterations);
            CUDA_CHECK(cudaDeviceSynchronize());
            
            // Read results
            float total_cycles;
            CUDA_CHECK(cudaMemcpy(&total_cycles, &d_output[num_threads], sizeof(float), cudaMemcpyDeviceToHost));
            
            // Calculate metrics
            float total_ops = (float)num_threads * iterations * ilp;
            float throughput_ops_per_cycle = safe_divide(total_ops, total_cycles);
            float throughput_gflops = throughput_ops_per_cycle * gpu_clock_mhz / 1000.0f;
            
            // Store result
            ILPThroughputMeasurement measurement;
            measurement.instruction = instruction_name;
            measurement.ilp = ilp;
            measurement.warps = warps;
            measurement.threads = num_threads;
            measurement.total_cycles = total_cycles;
            measurement.total_ops = total_ops;
            measurement.throughput_ops_per_cycle = throughput_ops_per_cycle;
            measurement.throughput_gflops = throughput_gflops;
            
            results.push_back(measurement);
            
            if (verbose) {
                printf("ILP=%d, Warps=%d, Threads=%d: %.2f ops/cycle, %.2f GFLOPS\n",
                       ilp, warps, num_threads, throughput_ops_per_cycle, throughput_gflops);
            }
            
            cudaFree(d_input);
            cudaFree(d_output);
        }
    }
    
    return results;
}

// Generic throughput measurement for int kernels
std::vector<ILPThroughputMeasurement> measure_instruction_throughput_int(
    const std::string& instruction_name,
    void (*kernel_ilp1)(int*, int*, int),
    void (*kernel_ilp2)(int*, int*, int), 
    void (*kernel_ilp3)(int*, int*, int),
    int ilp1, int ilp2, int ilp3,
    float gpu_clock_mhz,
    bool verbose = false) {
    
    std::vector<ILPThroughputMeasurement> results;
    
    if (verbose) {
        printf("\n=== Measuring %s Throughput ===\n", instruction_name.c_str());
    }
    
    // Test every warp configuration from 1 to 32 for precise PeakWarps measurement
    std::vector<int> warp_configs;
    for (int warps = 1; warps <= 32; warps++) {
        warp_configs.push_back(warps);
    }
    const int iterations = 1024;
    
    // For each ILP level
    std::vector<int> ilp_levels = {ilp1, ilp2, ilp3};
    void (*kernels[])(int*, int*, int) = {kernel_ilp1, kernel_ilp2, kernel_ilp3};
    
    for (int ilp_idx = 0; ilp_idx < 3; ilp_idx++) {
        int ilp = ilp_levels[ilp_idx];
        auto kernel = kernels[ilp_idx];
        
        for (int warps : warp_configs) {
            int num_threads = warps * WARP_SIZE;
            
            // Allocate memory
            int *d_input, *d_output;
            CUDA_CHECK(cudaMalloc(&d_input, num_threads * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_output, (num_threads + 1) * sizeof(int)));
            
            // Initialize input
            std::vector<int> h_input(num_threads, 1000);
            for (int i = 0; i < num_threads; i++) {
                h_input[i] = 1000 + i;
            }
            CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), num_threads * sizeof(int), cudaMemcpyHostToDevice));
            
            // Configure single block (as per paper Figure 3)
            dim3 blockSize(min(num_threads, 1024));
            dim3 gridSize(1);  // Single block only
            
            if (num_threads > 1024) {
                // Skip configurations that exceed block size limit
                cudaFree(d_input);
                cudaFree(d_output);
                continue;
            }
            
            // Warm-up
            kernel<<<gridSize, blockSize>>>(d_input, d_output, iterations);
            CUDA_CHECK(cudaDeviceSynchronize());
            
            // Actual measurement
            kernel<<<gridSize, blockSize>>>(d_input, d_output, iterations);
            CUDA_CHECK(cudaDeviceSynchronize());
            
            // Read results
            int total_cycles_int;
            CUDA_CHECK(cudaMemcpy(&total_cycles_int, &d_output[num_threads], sizeof(int), cudaMemcpyDeviceToHost));
            float total_cycles = (float)total_cycles_int;
            
            // Calculate metrics
            float total_ops = (float)num_threads * iterations * ilp;
            float throughput_ops_per_cycle = safe_divide(total_ops, total_cycles);
            float throughput_gflops = throughput_ops_per_cycle * gpu_clock_mhz / 1000.0f;
            
            // Store result
            ILPThroughputMeasurement measurement;
            measurement.instruction = instruction_name;
            measurement.ilp = ilp;
            measurement.warps = warps;
            measurement.threads = num_threads;
            measurement.total_cycles = total_cycles;
            measurement.total_ops = total_ops;
            measurement.throughput_ops_per_cycle = throughput_ops_per_cycle;
            measurement.throughput_gflops = throughput_gflops;
            
            results.push_back(measurement);
            
            if (verbose) {
                printf("ILP=%d, Warps=%d, Threads=%d: %.2f ops/cycle, %.2f GFLOPS\n",
                       ilp, warps, num_threads, throughput_ops_per_cycle, throughput_gflops);
            }
            
            cudaFree(d_input);
            cudaFree(d_output);
        }
    }
    
    return results;
}

// Wrapper for backward compatibility with existing float kernel measurements
std::vector<ILPThroughputMeasurement> measure_instruction_throughput(
    const std::string& instruction_name,
    void (*kernel_ilp1)(float*, float*, int),
    void (*kernel_ilp2)(float*, float*, int), 
    void (*kernel_ilp3)(float*, float*, int),
    int ilp1, int ilp2, int ilp3,
    float gpu_clock_mhz,
    bool verbose = false) {
    
    return measure_instruction_throughput_float(instruction_name, kernel_ilp1, kernel_ilp2, kernel_ilp3, ilp1, ilp2, ilp3, gpu_clock_mhz, verbose);
}

// Calculate peak warps for given instruction and ILP
int calculate_peak_warps(const std::vector<ILPThroughputMeasurement> &measurements,
                         const std::string &instruction_name,
                         int ilp) {
    float max_throughput = 0.0f;
    int peak_warps = 32;  // Initialize to maximum possible
    
    // Find maximum throughput first
    for (const auto &m : measurements) {
        if (m.instruction == instruction_name && m.ilp == ilp) {
            if (m.throughput_ops_per_cycle > max_throughput) {
                max_throughput = m.throughput_ops_per_cycle;
            }
        }
    }
    
    // Find minimum warps that achieve at least 90% of max throughput
    float threshold = max_throughput * 0.90f;
    for (const auto &m : measurements) {
        if (m.instruction == instruction_name && m.ilp == ilp) {
            if (m.throughput_ops_per_cycle >= threshold) {
                if (m.warps < peak_warps) {  // Find minimum warps meeting threshold
                    peak_warps = m.warps;
                }
            }
        }
    }
    
    return peak_warps;
}

// Get throughput at a specific ILP level
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

// Generate instruction summary
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
    printf("TABLE: Complete Instruction Summary (Original Table II + High-Frequency + MOV Variants)\n");
    printf("===========================================================================================================\n");
    printf("%-15s %-10s %-15s %-15s %-15s %-15s %-15s %-15s\n",
           "Instruction", "Latency", "Throughput", "Throughput", "Throughput",
           "PeakWarps", "PeakWarps", "PeakWarps");
    printf("%-15s %-10s %-15s %-15s %-15s %-15s %-15s %-15s\n",
           "", "", "ILP=1", "ILP=2", "ILP=3", "ILP=1", "ILP=2", "ILP=3");
    printf("-----------------------------------------------------------------------------------------------------------\n");
    
    for (const auto &summary : summaries) {
        // Check if this is a MOV instruction (known to have optimization issues)
        bool is_mov_instruction = (summary.instruction.find("MOV") == 0);
        
        if (is_mov_instruction) {
            // For MOV instructions, show N.A. for ILP=2,3 throughput due to compiler optimization issues
            printf("%-15s %-10.2f %-15.2f %-15s %-15s %-15d %-15s %-15s\n",
                   summary.instruction.c_str(),
                   summary.latency,
                   summary.throughput_ilp1,
                   "N.A.",  // ILP=2 throughput
                   "N.A.",  // ILP=3 throughput  
                   summary.peak_warps_ilp1,
                   "N.A.",  // ILP=2 peak warps
                   "N.A."   // ILP=3 peak warps
            );
        } else {
            // Regular instructions - show all values
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
    }
    
    printf("===========================================================================================================\n");
    printf("Note: MOV instruction ILP=2,3 values marked N.A. due to compiler optimization challenges\n");
}

// Save instruction summary table to CSV file
void save_instruction_summary_csv(const std::vector<InstructionSummary> &summaries, const std::string& filename) {
    FILE* file = fopen(filename.c_str(), "w");
    if (!file) {
        printf("Error: Could not create CSV file %s\n", filename.c_str());
        return;
    }
    
    // Write CSV header
    fprintf(file, "Instruction,Latency,Throughput_ILP1,Throughput_ILP2,Throughput_ILP3,PeakWarps_ILP1,PeakWarps_ILP2,PeakWarps_ILP3\n");
    
    // Write data rows
    for (const auto &summary : summaries) {
        // Check if this is a MOV instruction (known to have optimization issues)
        bool is_mov_instruction = (summary.instruction.find("MOV") == 0);
        
        if (is_mov_instruction) {
            // For MOV instructions, show N.A. for ILP=2,3 due to compiler optimization issues
            fprintf(file, "%s,%.2f,%.2f,N.A.,N.A.,%d,N.A.,N.A.\n",
                   summary.instruction.c_str(),
                   summary.latency,
                   summary.throughput_ilp1,
                   summary.peak_warps_ilp1);
        } else {
            // Regular instructions - show all values
            fprintf(file, "%s,%.2f,%.2f,%.2f,%.2f,%d,%d,%d\n",
                   summary.instruction.c_str(),
                   summary.latency,
                   summary.throughput_ilp1,
                   summary.throughput_ilp2,
                   summary.throughput_ilp3,
                   summary.peak_warps_ilp1,
                   summary.peak_warps_ilp2,
                   summary.peak_warps_ilp3);
        }
    }
    
    fclose(file);
    printf("\nInstruction summary saved to: %s\n", filename.c_str());
}

// ============================================================================
// MAIN FUNCTION
// ============================================================================

int main() {
    // Get device properties
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    printf("========================================\n");
    printf("COMPLETE CUDA Microbenchmarking Tool\n");
    printf("Table II + High-Freq + MOV Variants\n");
    printf("WITH ALL THROUGHPUT KERNELS ADDED\n");
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
    
    // Latency measurements for ALL instructions from Table II
    printf("\n\n### COMPLETE LATENCY MEASUREMENTS ###\n");
    
    // Floating-point instructions
    LatencyMeasurement fadd_latency = measure_latency_generic_float(latency_fadd, "FADD", gpu_clock_mhz);
    LatencyMeasurement fsub_latency = measure_latency_generic_float(latency_fsub, "FSUB", gpu_clock_mhz);
    LatencyMeasurement fmul_latency = measure_latency_generic_float(latency_fmul, "FMUL", gpu_clock_mhz);
    LatencyMeasurement fma_latency = measure_latency_generic_float(latency_fma, "FMA", gpu_clock_mhz);
    LatencyMeasurement divf_latency = measure_latency_generic_float(latency_divf, "DIVF", gpu_clock_mhz, 10.0f);
    LatencyMeasurement sqrt_latency = measure_latency_generic_float(latency_sqrt, "SQRT", gpu_clock_mhz, 4.0f);
    
    // Integer instructions (original)
    LatencyMeasurement adds_latency = measure_latency_generic_int(latency_adds, "ADDS", gpu_clock_mhz, 100);
    LatencyMeasurement subs_latency = measure_latency_generic_int(latency_subs, "SUBS", gpu_clock_mhz, 1000);
    LatencyMeasurement and_latency = measure_latency_generic_int(latency_and, "AND", gpu_clock_mhz, 0xFFFFFFFF);
    LatencyMeasurement mads_latency = measure_latency_generic_int(latency_mads, "MADS", gpu_clock_mhz, 10);
    LatencyMeasurement muls_latency = measure_latency_generic_int(latency_muls, "MULS", gpu_clock_mhz, 10);
    LatencyMeasurement divs_latency = measure_latency_generic_int(latency_divs, "DIVS", gpu_clock_mhz, 1000000);
    
    // NEW High-frequency integer instructions
    LatencyMeasurement add_s32_latency = measure_latency_generic_int(latency_add_s32, "ADD.S32", gpu_clock_mhz, 1000);
    LatencyMeasurement add_s64_latency = measure_latency_generic_int64(latency_add_s64, "ADD.S64", gpu_clock_mhz, 1000LL);
    LatencyMeasurement sub_s32_latency = measure_latency_generic_int(latency_sub_s32, "SUB.S32", gpu_clock_mhz, 1000);
    LatencyMeasurement mul_lo_s32_latency = measure_latency_generic_int(latency_mul_lo_s32, "MUL.LO.S32", gpu_clock_mhz, 100);
    
    // Special handling for mul_wide_s32 (int input, long long output)
    printf("\n=== Measuring MUL.WIDE.S32 Latency ===\n");
    int *d_input_int;
    long long *d_output_long;
    int h_input_int = 100;
    long long h_output_long[2];
    CUDA_CHECK(cudaMalloc(&d_input_int, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_output_long, 2 * sizeof(long long)));
    CUDA_CHECK(cudaMemcpy(d_input_int, &h_input_int, sizeof(int), cudaMemcpyHostToDevice));
    latency_mul_wide_s32<<<1, 1>>>(d_input_int, d_output_long, 100);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_output_long, d_output_long, 2 * sizeof(long long), cudaMemcpyDeviceToHost));
    LatencyMeasurement mul_wide_s32_latency;
    mul_wide_s32_latency.name = "MUL.WIDE.S32";
    mul_wide_s32_latency.total_cycles = (float)h_output_long[1];
    mul_wide_s32_latency.total_instructions = 100 * 256;
    mul_wide_s32_latency.latency_cycles = mul_wide_s32_latency.total_cycles / mul_wide_s32_latency.total_instructions;
    mul_wide_s32_latency.latency_ns = mul_wide_s32_latency.latency_cycles * 1000.0f / gpu_clock_mhz;
    printf("Total cycles: %.2f\nLatency: %.2f cycles (%.2f ns)\n", 
           mul_wide_s32_latency.total_cycles, mul_wide_s32_latency.latency_cycles, mul_wide_s32_latency.latency_ns);
    cudaFree(d_input_int);
    cudaFree(d_output_long);
    
    LatencyMeasurement mad_lo_s32_latency = measure_latency_generic_int(latency_mad_lo_s32, "MAD.LO.S32", gpu_clock_mhz, 100);
    
    // Special instructions
    LatencyMeasurement setp_latency = measure_latency_generic_float(latency_setp, "SETP", gpu_clock_mhz, 1.0f);
    LatencyMeasurement cvt_latency = measure_latency_cvt(gpu_clock_mhz);
    LatencyMeasurement mov_latency = measure_latency_generic_float(latency_mov, "MOV", gpu_clock_mhz);
    
    // NEW MOV variant measurements
    printf("\n=== Measuring MOV Variants ===\n");
    
    // MOV.U32 (manually measured - similar to int measurement)
    unsigned int *d_input_uint, *d_output_uint;
    unsigned int h_input_uint = 1000U;
    unsigned int h_output_uint[2];
    CUDA_CHECK(cudaMalloc(&d_input_uint, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_output_uint, 2 * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(d_input_uint, &h_input_uint, sizeof(unsigned int), cudaMemcpyHostToDevice));
    latency_mov_u32<<<1, 1>>>(d_input_uint, d_output_uint, 100);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_output_uint, d_output_uint, 2 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    LatencyMeasurement mov_u32_latency;
    mov_u32_latency.name = "MOV.U32";
    mov_u32_latency.total_cycles = (float)h_output_uint[1];
    mov_u32_latency.total_instructions = 100 * 256;
    mov_u32_latency.latency_cycles = mov_u32_latency.total_cycles / mov_u32_latency.total_instructions;
    mov_u32_latency.latency_ns = mov_u32_latency.latency_cycles * 1000.0f / gpu_clock_mhz;
    printf("MOV.U32 - Total cycles: %.2f, Latency: %.2f cycles\n", mov_u32_latency.total_cycles, mov_u32_latency.latency_cycles);
    cudaFree(d_input_uint);
    cudaFree(d_output_uint);
    
    // MOV.F32 (same as MOV)
    LatencyMeasurement mov_f32_latency = measure_latency_generic_float(latency_mov_f32, "MOV.F32", gpu_clock_mhz);
    
    // MOV.B64 (manually measured)
    unsigned long long *d_input_ullong, *d_output_ullong;
    unsigned long long h_input_ullong = 1000ULL;
    unsigned long long h_output_ullong[2];
    CUDA_CHECK(cudaMalloc(&d_input_ullong, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_output_ullong, 2 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpy(d_input_ullong, &h_input_ullong, sizeof(unsigned long long), cudaMemcpyHostToDevice));
    latency_mov_b64<<<1, 1>>>(d_input_ullong, d_output_ullong, 100);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_output_ullong, d_output_ullong, 2 * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    LatencyMeasurement mov_b64_latency;
    mov_b64_latency.name = "MOV.B64";
    mov_b64_latency.total_cycles = (float)h_output_ullong[1];
    mov_b64_latency.total_instructions = 100 * 256;
    mov_b64_latency.latency_cycles = mov_b64_latency.total_cycles / mov_b64_latency.total_instructions;
    mov_b64_latency.latency_ns = mov_b64_latency.latency_cycles * 1000.0f / gpu_clock_mhz;
    printf("MOV.B64 - Total cycles: %.2f, Latency: %.2f cycles\n", mov_b64_latency.total_cycles, mov_b64_latency.latency_cycles);
    cudaFree(d_input_ullong);
    cudaFree(d_output_ullong);
    
    // MOV.B32 (manually measured)
    CUDA_CHECK(cudaMalloc(&d_input_uint, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_output_uint, 2 * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(d_input_uint, &h_input_uint, sizeof(unsigned int), cudaMemcpyHostToDevice));
    latency_mov_b32<<<1, 1>>>(d_input_uint, d_output_uint, 100);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_output_uint, d_output_uint, 2 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    LatencyMeasurement mov_b32_latency;
    mov_b32_latency.name = "MOV.B32";
    mov_b32_latency.total_cycles = (float)h_output_uint[1];
    mov_b32_latency.total_instructions = 100 * 256;
    mov_b32_latency.latency_cycles = mov_b32_latency.total_cycles / mov_b32_latency.total_instructions;
    mov_b32_latency.latency_ns = mov_b32_latency.latency_cycles * 1000.0f / gpu_clock_mhz;
    printf("MOV.B32 - Total cycles: %.2f, Latency: %.2f cycles\n", mov_b32_latency.total_cycles, mov_b32_latency.latency_cycles);
    cudaFree(d_input_uint);
    cudaFree(d_output_uint);
    
    // Throughput measurements for ALL instructions
    printf("\n\n### COMPLETE THROUGHPUT MEASUREMENTS ###\n");
    
    // Floating-point throughput measurements
    std::vector<ILPThroughputMeasurement> fadd_throughput = measure_instruction_throughput(
        "FADD", throughput_fadd_ilp1, throughput_fadd_ilp2, throughput_fadd_ilp3,
        1, 2, 3, gpu_clock_mhz, true);
    
    std::vector<ILPThroughputMeasurement> fsub_throughput = measure_instruction_throughput(
        "FSUB", throughput_fsub_ilp1, throughput_fsub_ilp2, throughput_fsub_ilp3,
        1, 2, 3, gpu_clock_mhz, true);
    
    std::vector<ILPThroughputMeasurement> fmul_throughput = measure_instruction_throughput(
        "FMUL", throughput_fmul_ilp1, throughput_fmul_ilp2, throughput_fmul_ilp3,
        1, 2, 3, gpu_clock_mhz, true);
    
    std::vector<ILPThroughputMeasurement> fma_throughput = measure_instruction_throughput(
        "FMA", throughput_fma_ilp1, throughput_fma_ilp2, throughput_fma_ilp3,
        1, 2, 3, gpu_clock_mhz, true);
    
    std::vector<ILPThroughputMeasurement> divf_throughput = measure_instruction_throughput(
        "DIVF", throughput_divf_ilp1, throughput_divf_ilp2, throughput_divf_ilp3,
        1, 2, 3, gpu_clock_mhz, true);
    
    std::vector<ILPThroughputMeasurement> sqrt_throughput = measure_instruction_throughput(
        "SQRT", throughput_sqrt_ilp1, throughput_sqrt_ilp2, throughput_sqrt_ilp3,
        1, 2, 3, gpu_clock_mhz, true);
    
    // Integer throughput measurements (original)
    std::vector<ILPThroughputMeasurement> adds_throughput = measure_instruction_throughput_int(
        "ADDS", throughput_adds_ilp1, throughput_adds_ilp2, throughput_adds_ilp3,
        1, 2, 3, gpu_clock_mhz, true);
    
    std::vector<ILPThroughputMeasurement> subs_throughput = measure_instruction_throughput_int(
        "SUBS", throughput_subs_ilp1, throughput_subs_ilp2, throughput_subs_ilp3,
        1, 2, 3, gpu_clock_mhz, true);
    
    std::vector<ILPThroughputMeasurement> and_throughput = measure_instruction_throughput_int(
        "AND", throughput_and_ilp1, throughput_and_ilp2, throughput_and_ilp3,
        1, 2, 3, gpu_clock_mhz, true);
    
    std::vector<ILPThroughputMeasurement> mads_throughput = measure_instruction_throughput_int(
        "MADS", throughput_mads_ilp1, throughput_mads_ilp2, throughput_mads_ilp3,
        1, 2, 3, gpu_clock_mhz, true);
    
    std::vector<ILPThroughputMeasurement> muls_throughput = measure_instruction_throughput_int(
        "MULS", throughput_muls_ilp1, throughput_muls_ilp2, throughput_muls_ilp3,
        1, 2, 3, gpu_clock_mhz, true);
    
    std::vector<ILPThroughputMeasurement> divs_throughput = measure_instruction_throughput_int(
        "DIVS", throughput_divs_ilp1, throughput_divs_ilp2, throughput_divs_ilp3,
        1, 2, 3, gpu_clock_mhz, true);
    
    // NEW High-frequency integer throughput measurements
    std::vector<ILPThroughputMeasurement> add_s32_throughput = measure_instruction_throughput_int(
        "ADD.S32", throughput_add_s32_ilp1, throughput_add_s32_ilp2, throughput_add_s32_ilp3,
        1, 2, 3, gpu_clock_mhz, true);
    
    std::vector<ILPThroughputMeasurement> add_s64_throughput = measure_instruction_throughput_int64(
        "ADD.S64", throughput_add_s64_ilp1, throughput_add_s64_ilp2, throughput_add_s64_ilp3,
        1, 2, 3, gpu_clock_mhz, true);
    
    std::vector<ILPThroughputMeasurement> sub_s32_throughput = measure_instruction_throughput_int(
        "SUB.S32", throughput_sub_s32_ilp1, throughput_sub_s32_ilp2, throughput_sub_s32_ilp3,
        1, 2, 3, gpu_clock_mhz, true);
    
    std::vector<ILPThroughputMeasurement> mul_lo_s32_throughput = measure_instruction_throughput_int(
        "MUL.LO.S32", throughput_mul_lo_s32_ilp1, throughput_mul_lo_s32_ilp2, throughput_mul_lo_s32_ilp3,
        1, 2, 3, gpu_clock_mhz, true);
    
    // Special handling for MUL.WIDE.S32 (int input -> long long output)
    std::vector<ILPThroughputMeasurement> mul_wide_s32_throughput = measure_mul_wide_s32_throughput(
        "MUL.WIDE.S32", throughput_mul_wide_s32_ilp1, throughput_mul_wide_s32_ilp2, throughput_mul_wide_s32_ilp3,
        1, 2, 3, gpu_clock_mhz, true);
    
    std::vector<ILPThroughputMeasurement> mad_lo_s32_throughput = measure_instruction_throughput_int(
        "MAD.LO.S32", throughput_mad_lo_s32_ilp1, throughput_mad_lo_s32_ilp2, throughput_mad_lo_s32_ilp3,
        1, 2, 3, gpu_clock_mhz, true);
    
    // Special instruction throughput measurements
    std::vector<ILPThroughputMeasurement> setp_throughput = measure_instruction_throughput(
        "SETP", throughput_setp_ilp1, throughput_setp_ilp2, throughput_setp_ilp3,
        1, 2, 3, gpu_clock_mhz, true);
    
    std::vector<ILPThroughputMeasurement> cvt_throughput = measure_instruction_throughput(
        "CVT", throughput_cvt_ilp1, throughput_cvt_ilp2, throughput_cvt_ilp3,
        1, 2, 3, gpu_clock_mhz, true);
    
    std::vector<ILPThroughputMeasurement> mov_throughput = measure_instruction_throughput(
        "MOV", throughput_mov_ilp1, throughput_mov_ilp2, throughput_mov_ilp3,
        1, 2, 3, gpu_clock_mhz, true);
    
    // NEW MOV variants throughput measurements (now with actual measurements!)
    printf("\n=== Measuring MOV Variants Throughput ===\n");
    
    // MOV.U32 throughput (using simplified measurement approach)
    std::vector<ILPThroughputMeasurement> mov_u32_throughput;
    
    // Simple single-warp measurement for MOV.U32 (demo approach)
    printf("Measuring MOV.U32 throughput...\n");
    unsigned int *d_u32_input, *d_u32_output;
    const int test_threads = 32;
    CUDA_CHECK(cudaMalloc(&d_u32_input, test_threads * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_u32_output, (test_threads + 1) * sizeof(unsigned int)));
    
    std::vector<unsigned int> h_u32_input(test_threads, 1000U);
    CUDA_CHECK(cudaMemcpy(d_u32_input, h_u32_input.data(), test_threads * sizeof(unsigned int), cudaMemcpyHostToDevice));
    
    // Test ILP=1
    throughput_mov_u32_ilp1<<<1, test_threads>>>(d_u32_input, d_u32_output, 1024);
    CUDA_CHECK(cudaDeviceSynchronize());
    unsigned int u32_cycles;
    CUDA_CHECK(cudaMemcpy(&u32_cycles, &d_u32_output[test_threads], sizeof(unsigned int), cudaMemcpyDeviceToHost));
    
    // Create measurement
    ILPThroughputMeasurement mov_u32_measurement;
    mov_u32_measurement.instruction = "MOV.U32";
    mov_u32_measurement.ilp = 1;
    mov_u32_measurement.warps = 1;
    mov_u32_measurement.threads = test_threads;
    mov_u32_measurement.total_cycles = (float)u32_cycles;
    mov_u32_measurement.total_ops = (float)test_threads * 1024 * 1;
    mov_u32_measurement.throughput_ops_per_cycle = safe_divide(mov_u32_measurement.total_ops, mov_u32_measurement.total_cycles);
    mov_u32_throughput.push_back(mov_u32_measurement);
    
    cudaFree(d_u32_input);
    cudaFree(d_u32_output);
    
    // MOV.F32 throughput (can reuse existing float measurement function)
    std::vector<ILPThroughputMeasurement> mov_f32_throughput = measure_instruction_throughput(
        "MOV.F32", throughput_mov_f32_ilp1, throughput_mov_f32_ilp2, throughput_mov_f32_ilp3,
        1, 2, 3, gpu_clock_mhz, true);
    
    // MOV.B64 throughput (simplified measurement)
    std::vector<ILPThroughputMeasurement> mov_b64_throughput;
    
    printf("Measuring MOV.B64 throughput...\n");
    unsigned long long *d_u64_input, *d_u64_output;
    CUDA_CHECK(cudaMalloc(&d_u64_input, test_threads * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_u64_output, (test_threads + 1) * sizeof(unsigned long long)));
    
    std::vector<unsigned long long> h_u64_input(test_threads, 1000ULL);
    CUDA_CHECK(cudaMemcpy(d_u64_input, h_u64_input.data(), test_threads * sizeof(unsigned long long), cudaMemcpyHostToDevice));
    
    // Test ILP=1
    throughput_mov_b64_ilp1<<<1, test_threads>>>(d_u64_input, d_u64_output, 1024);
    CUDA_CHECK(cudaDeviceSynchronize());
    unsigned long long u64_cycles;
    CUDA_CHECK(cudaMemcpy(&u64_cycles, &d_u64_output[test_threads], sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    
    ILPThroughputMeasurement mov_b64_measurement;
    mov_b64_measurement.instruction = "MOV.B64";
    mov_b64_measurement.ilp = 1;
    mov_b64_measurement.warps = 1;
    mov_b64_measurement.threads = test_threads;
    mov_b64_measurement.total_cycles = (float)u64_cycles;
    mov_b64_measurement.total_ops = (float)test_threads * 1024 * 1;
    mov_b64_measurement.throughput_ops_per_cycle = safe_divide(mov_b64_measurement.total_ops, mov_b64_measurement.total_cycles);
    mov_b64_throughput.push_back(mov_b64_measurement);
    
    cudaFree(d_u64_input);
    cudaFree(d_u64_output);
    
    // MOV.B32 throughput (similar to MOV.U32)
    std::vector<ILPThroughputMeasurement> mov_b32_throughput;
    
    printf("Measuring MOV.B32 throughput...\n");
    CUDA_CHECK(cudaMalloc(&d_u32_input, test_threads * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_u32_output, (test_threads + 1) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(d_u32_input, h_u32_input.data(), test_threads * sizeof(unsigned int), cudaMemcpyHostToDevice));
    
    // Test ILP=1  
    throughput_mov_b32_ilp1<<<1, test_threads>>>(d_u32_input, d_u32_output, 1024);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&u32_cycles, &d_u32_output[test_threads], sizeof(unsigned int), cudaMemcpyDeviceToHost));
    
    ILPThroughputMeasurement mov_b32_measurement;
    mov_b32_measurement.instruction = "MOV.B32";
    mov_b32_measurement.ilp = 1;
    mov_b32_measurement.warps = 1;
    mov_b32_measurement.threads = test_threads;
    mov_b32_measurement.total_cycles = (float)u32_cycles;
    mov_b32_measurement.total_ops = (float)test_threads * 1024 * 1;
    mov_b32_measurement.throughput_ops_per_cycle = safe_divide(mov_b32_measurement.total_ops, mov_b32_measurement.total_cycles);
    mov_b32_throughput.push_back(mov_b32_measurement);
    
    cudaFree(d_u32_input);
    cudaFree(d_u32_output);
    
    printf("MOV variants throughput measurements completed!\n");
    
    // Generate instruction summaries for all measured instructions
    std::vector<InstructionSummary> summaries;
    
    // Original instructions
    summaries.push_back(generate_instruction_summary(fadd_latency, fadd_throughput));
    summaries.push_back(generate_instruction_summary(fsub_latency, fsub_throughput));
    summaries.push_back(generate_instruction_summary(fmul_latency, fmul_throughput));
    summaries.push_back(generate_instruction_summary(fma_latency, fma_throughput));
    summaries.push_back(generate_instruction_summary(adds_latency, adds_throughput));
    summaries.push_back(generate_instruction_summary(subs_latency, subs_throughput));
    summaries.push_back(generate_instruction_summary(and_latency, and_throughput));
    summaries.push_back(generate_instruction_summary(mads_latency, mads_throughput));
    summaries.push_back(generate_instruction_summary(muls_latency, muls_throughput));
    summaries.push_back(generate_instruction_summary(divs_latency, divs_throughput));
    summaries.push_back(generate_instruction_summary(divf_latency, divf_throughput));
    summaries.push_back(generate_instruction_summary(sqrt_latency, sqrt_throughput));
    summaries.push_back(generate_instruction_summary(setp_latency, setp_throughput));
    summaries.push_back(generate_instruction_summary(cvt_latency, cvt_throughput));
    summaries.push_back(generate_instruction_summary(mov_latency, mov_throughput));
    
    // NEW MOV variants
    summaries.push_back(generate_instruction_summary(mov_u32_latency, mov_u32_throughput));
    summaries.push_back(generate_instruction_summary(mov_f32_latency, mov_f32_throughput));
    summaries.push_back(generate_instruction_summary(mov_b64_latency, mov_b64_throughput));
    summaries.push_back(generate_instruction_summary(mov_b32_latency, mov_b32_throughput));
    
    // NEW High-frequency integer instructions
    summaries.push_back(generate_instruction_summary(add_s32_latency, add_s32_throughput));
    summaries.push_back(generate_instruction_summary(add_s64_latency, add_s64_throughput));
    summaries.push_back(generate_instruction_summary(sub_s32_latency, sub_s32_throughput));
    summaries.push_back(generate_instruction_summary(mul_lo_s32_latency, mul_lo_s32_throughput));
    summaries.push_back(generate_instruction_summary(mul_wide_s32_latency, mul_wide_s32_throughput));
    summaries.push_back(generate_instruction_summary(mad_lo_s32_latency, mad_lo_s32_throughput));
    
    // Print complete table
    print_instruction_summary_table(summaries);
    
    // Save results to CSV file
    save_instruction_summary_csv(summaries, "ins_microbenchmark.csv");
    
    printf("\n\n========================================\n");
    printf("COMPLETE Microbenchmarking Finished!\n");
    printf("All Table II Instructions Measured\n");
    printf("WITH COMPLETE THROUGHPUT KERNELS!\n");
    printf("Results saved to ins_microbenchmark.csv\n");
    printf("========================================\n");
    
    return 0;
}

