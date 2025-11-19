/*
 * Kernel Launch Overhead Measurement Experiment
 * =============================================
 * 
 * This experiment measures kernel launch overhead on RTX 2080 Ti
 * by using minimal computation kernels and precise CUDA event timing.
 * 
 * Strategy:
 * 1. Create kernels with known, minimal computation
 * 2. Measure total execution time with CUDA events
 * 3. Subtract theoretical computation time to isolate launch overhead
 * 4. Test across different thread counts to derive overhead formula
 * 
 * Compile: nvcc -o launch_overhead_test launch_overhead_test.cu
 * Run: ./launch_overhead_test
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include <cmath>

// Minimal computation kernel - just a few arithmetic operations
__global__ void minimal_kernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = input[idx];
        val = val + 1.0f;           // 1 FADD
        val = val * 2.0f;           // 1 FMUL  
        val = val - 0.5f;           // 1 FSUB
        output[idx] = val;          // 1 store
        // Total: ~4 simple operations per thread
    }
}

// Empty kernel - absolute minimum overhead
__global__ void empty_kernel() {
    // Literally does nothing - just launch overhead
}

// Single operation kernel
__global__ void single_op_kernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] + 1.0f;  // Just 1 FADD + 1 store
    }
}

class LaunchOverheadMeasurement {
private:
    cudaEvent_t start_event, stop_event;
    float* d_input;
    float* d_output;
    int max_threads;

public:
    LaunchOverheadMeasurement(int max_threads = 1024*1024) : max_threads(max_threads) {
        // Create CUDA events for precise timing
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
        
        // Allocate GPU memory
        cudaMalloc(&d_input, max_threads * sizeof(float));
        cudaMalloc(&d_output, max_threads * sizeof(float));
        
        // Initialize input data
        std::vector<float> h_input(max_threads, 1.0f);
        cudaMemcpy(d_input, h_input.data(), max_threads * sizeof(float), cudaMemcpyHostToDevice);
        
        std::cout << "LaunchOverheadMeasurement initialized for RTX 2080 Ti" << std::endl;
        std::cout << "Max threads: " << max_threads << std::endl;
    }
    
    ~LaunchOverheadMeasurement() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
        cudaFree(d_input);
        cudaFree(d_output);
    }
    
    float measure_kernel_time(int num_blocks, int threads_per_block, int kernel_type = 0) {
        int total_threads = num_blocks * threads_per_block;
        
        // Warm up
        for (int i = 0; i < 5; i++) {
            switch (kernel_type) {
                case 0:
                    empty_kernel<<<num_blocks, threads_per_block>>>();
                    break;
                case 1:
                    single_op_kernel<<<num_blocks, threads_per_block>>>(d_input, d_output, total_threads);
                    break;
                case 2:
                    minimal_kernel<<<num_blocks, threads_per_block>>>(d_input, d_output, total_threads);
                    break;
            }
        }
        cudaDeviceSynchronize();
        
        // Actual measurement - run multiple times for precision
        const int num_iterations = 1000;
        
        cudaEventRecord(start_event);
        
        for (int i = 0; i < num_iterations; i++) {
            switch (kernel_type) {
                case 0:
                    empty_kernel<<<num_blocks, threads_per_block>>>();
                    break;
                case 1:
                    single_op_kernel<<<num_blocks, threads_per_block>>>(d_input, d_output, total_threads);
                    break;
                case 2:
                    minimal_kernel<<<num_blocks, threads_per_block>>>(d_input, d_output, total_threads);
                    break;
            }
        }
        
        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);
        
        float total_time_ms;
        cudaEventElapsedTime(&total_time_ms, start_event, stop_event);
        
        // Return average time per kernel launch in microseconds
        return (total_time_ms * 1000.0f) / num_iterations;
    }
    
    void run_comprehensive_experiment() {
        std::cout << "\n=== Comprehensive Launch Overhead Experiment ===" << std::endl;
        std::cout << "Testing different kernel types and thread configurations" << std::endl;
        
        // Test configurations
        std::vector<std::pair<int, int>> configs = {
            // (blocks, threads_per_block)
            {1, 32},      {1, 64},      {1, 128},     {1, 256},     {1, 512},     {1, 1024},
            {2, 32},      {4, 32},      {8, 32},      {16, 32},     {32, 32},
            {2, 256},     {4, 256},     {8, 256},     {16, 256},    {32, 256},
            {5, 225},     // Your specific config: 5×5 blocks, 15×15 threads per block  
            {25, 225},    // 25 blocks of 225 threads (your total config)
            {64, 256},    {128, 256},   {256, 256},   {512, 256},
            {100, 100},   {200, 200},   {500, 200}
        };
        
        // Open output file
        std::ofstream outfile("launch_overhead_results.csv");
        outfile << "num_blocks,threads_per_block,total_threads,empty_kernel_us,single_op_us,minimal_us,launch_overhead_us\n";
        
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "\nConfiguration,Total_Threads,Empty_Kernel(μs),Single_Op(μs),Minimal(μs),Launch_Overhead(μs)\n";
        std::cout << std::string(90, '-') << std::endl;
        
        for (auto& config : configs) {
            int blocks = config.first;
            int threads_per_block = config.second;
            int total_threads = blocks * threads_per_block;
            
            if (total_threads > max_threads) continue;
            
            // Measure all three kernel types
            float empty_time_us = measure_kernel_time(blocks, threads_per_block, 0);
            float single_op_time_us = measure_kernel_time(blocks, threads_per_block, 1);
            float minimal_time_us = measure_kernel_time(blocks, threads_per_block, 2);
            
            // Launch overhead is approximately the empty kernel time
            float launch_overhead_us = empty_time_us;
            
            // Print results
            std::cout << "(" << blocks << "," << threads_per_block << ")," 
                      << std::setw(8) << total_threads << ","
                      << std::setw(10) << empty_time_us << ","
                      << std::setw(10) << single_op_time_us << ","
                      << std::setw(10) << minimal_time_us << ","
                      << std::setw(12) << launch_overhead_us << std::endl;
            
            // Write to CSV
            outfile << blocks << "," << threads_per_block << "," << total_threads << ","
                    << empty_time_us << "," << single_op_time_us << "," << minimal_time_us << ","
                    << launch_overhead_us << "\n";
        }
        
        outfile.close();
        std::cout << "\nResults saved to: launch_overhead_results.csv" << std::endl;
    }
    
    void analyze_overhead_formula() {
        std::cout << "\n=== Analyzing Launch Overhead Formula ===" << std::endl;
        
        // Test systematic thread counts to derive formula  
        std::vector<int> thread_counts;
        for (int threads = 32; threads <= 65536; threads *= 2) {
            thread_counts.push_back(threads);
        }
        
        std::cout << "Total_Threads,Launch_Overhead(μs),Old_Formula(μs),Ratio\n";
        std::cout << std::string(60, '-') << std::endl;
        
        std::ofstream formula_file("overhead_formula_analysis.csv");
        formula_file << "total_threads,measured_overhead_us,old_formula_us,ratio\n";
        
        for (int total_threads : thread_counts) {
            if (total_threads > max_threads) break;
            
            // Use 256 threads per block (good balance)
            int threads_per_block = std::min(256, total_threads);
            int num_blocks = (total_threads + threads_per_block - 1) / threads_per_block;
            
            float measured_us = measure_kernel_time(num_blocks, threads_per_block, 0);
            
            // Calculate old formula prediction (convert to microseconds)
            float old_formula_us = (1.260e-08 * total_threads + 4.260e-02) * 1e6;
            float ratio = measured_us / old_formula_us;
            
            std::cout << std::setw(12) << total_threads << ","
                      << std::setw(12) << measured_us << ","
                      << std::setw(12) << old_formula_us << ","
                      << std::setw(8) << ratio << std::endl;
            
            formula_file << total_threads << "," << measured_us << "," 
                        << old_formula_us << "," << ratio << "\n";
        }
        
        formula_file.close();
        std::cout << "\nFormula analysis saved to: overhead_formula_analysis.csv" << std::endl;
    }
};

int main() {
    std::cout << "Kernel Launch Overhead Measurement for RTX 2080 Ti" << std::endl;
    std::cout << "===================================================" << std::endl;
    
    // Check GPU
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "Error: No CUDA devices found!" << std::endl;
        return 1;
    }
    
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    std::cout << "GPU: " << props.name << std::endl;
    std::cout << "Compute Capability: " << props.major << "." << props.minor << std::endl;
    std::cout << "Memory Clock Rate: " << props.memoryClockRate/1000 << " MHz" << std::endl;
    std::cout << "GPU Clock Rate: " << props.clockRate/1000 << " MHz" << std::endl;
    std::cout << std::endl;
    
    // Run experiments
    LaunchOverheadMeasurement experiment;
    
    // Test 1: Comprehensive experiment across configurations
    experiment.run_comprehensive_experiment();
    
    // Test 2: Analyze overhead formula
    experiment.analyze_overhead_formula();
    
    std::cout << "\n=== Experiment Complete ===" << std::endl;
    std::cout << "Files generated:" << std::endl;
    std::cout << "  1. launch_overhead_results.csv - Comprehensive results" << std::endl;
    std::cout << "  2. overhead_formula_analysis.csv - Formula comparison" << std::endl;
    std::cout << "\nUse these results to derive a new launch overhead formula for RTX 2080 Ti!" << std::endl;
    
    return 0;
}
