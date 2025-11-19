#!/usr/bin/env python3
"""
Multi-GPU Batch PTX Simulator
Extends the simulator to support all GPU architectures with their specific launch overheads.
"""

import os
import sys
import subprocess
import csv
import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class GPUConfig:
    """Configuration for each GPU architecture"""
    name: str
    cuda_id: str
    microbench_file: str
    launch_overhead_us: float  # Launch overhead in microseconds
    clock_mhz: float          # GPU base clock in MHz

# GPU configurations based on your launch overhead analysis
GPU_CONFIGS = {
    'cuda2': GPUConfig(
        name='RTX_2080_Ti',
        cuda_id='cuda2', 
        microbench_file='ins_microbenchmark_RTX_2080_Ti.csv',
        launch_overhead_us=2.287,  # 2.287 μs (POWER formula best fit)
        clock_mhz=1755.0
    ),
    'cuda3': GPUConfig(
        name='TITAN_V',
        cuda_id='cuda3',
        microbench_file='ins_microbenchmark_TITAN_V.csv', 
        launch_overhead_us=2.051,  # 2.051 μs constant
        clock_mhz=1455.0  # TITAN V base clock
    ),
    'cuda4': GPUConfig(
        name='GTX_TITAN_X',
        cuda_id='cuda4',
        microbench_file='ins_microbenchmark_GTX_TITAN_X.csv',
        launch_overhead_us=2.510,  # 2.510 μs constant  
        clock_mhz=1000.0  # GTX TITAN X base clock
    ),
    'cuda5': GPUConfig(
        name='RTX_4070',
        cuda_id='cuda5',
        microbench_file='ins_microbenchmark_RTX_4070.csv',
        launch_overhead_us=4.532,  # 4.532 μs constant
        clock_mhz=2475.0  # RTX 4070 boost clock
    )
}

@dataclass 
class KernelInfo:
    """Information about a kernel to simulate"""
    i: int
    j: int  
    kernel_id: str
    gpu_name: str
    ptx_path: str

@dataclass
class SimulationConfig:
    """Fixed simulation configuration"""
    grid_x: int = 5
    grid_y: int = 5
    block_x: int = 15
    block_y: int = 15
    
    @property
    def total_blocks(self) -> int:
        return self.grid_x * self.grid_y
    
    @property
    def threads_per_block(self) -> int:
        return self.block_x * self.block_y
    
    @property
    def total_threads(self) -> int:
        return self.total_blocks * self.threads_per_block

class MultiGPUKernelDiscovery:
    """Discovers kernels for all GPU architectures"""
    
    def __init__(self, repo_root: str):
        self.repo_root = Path(repo_root)
        self.kernels_dir = self.repo_root / "kernels_src" / "test"
        self.simulator_path = self.repo_root / "static_analysis_model" / "enhanced_ptx_simulator.py"
        self.microbench_dir = self.repo_root / "microbenchmarking"
        
    def discover_kernels(self) -> Dict[str, List[KernelInfo]]:
        """Discover all kernels for all GPU architectures"""
        kernels_by_gpu = {gpu_id: [] for gpu_id in GPU_CONFIGS.keys()}
        
        if not self.kernels_dir.exists():
            print(f"Error: Kernels directory not found: {self.kernels_dir}")
            return kernels_by_gpu
            
        # Search for PTX files in all subdirectories
        pattern = re.compile(r'kernel_(\d+)_(\d+)_(\w+)\.ptx')
        
        for ptx_file in self.kernels_dir.rglob("*.ptx"):
            match = pattern.match(ptx_file.name)
            if match:
                i, j, gpu_name = match.groups()
                
                # Find matching GPU config
                gpu_id = None
                for gid, config in GPU_CONFIGS.items():
                    if config.name == gpu_name:
                        gpu_id = gid
                        break
                
                if gpu_id:
                    kernel_info = KernelInfo(
                        i=int(i),
                        j=int(j),
                        kernel_id=f"{i}_{j}",
                        gpu_name=gpu_name,
                        ptx_path=str(ptx_file)
                    )
                    kernels_by_gpu[gpu_id].append(kernel_info)
                else:
                    print(f"Warning: Unknown GPU architecture in file: {ptx_file.name}")
        
        return kernels_by_gpu
    
    def validate_environment(self) -> bool:
        """Validate that all required files exist"""
        paths_to_check = [
            (self.kernels_dir, "Kernels directory"),
            (self.simulator_path, "Enhanced PTX simulator")
        ]
        
        all_valid = True
        for path, description in paths_to_check:
            if not path.exists():
                print(f"Error: {description} not found at: {path}")
                all_valid = False
            else:
                print(f"✓ Found {description}: {path}")
        
        # Check microbenchmark files
        for gpu_id, config in GPU_CONFIGS.items():
            microbench_path = self.microbench_dir / config.microbench_file
            if not microbench_path.exists():
                print(f"Warning: Microbenchmark file not found for {config.name}: {microbench_path}")
                # Don't fail completely, just warn
            else:
                print(f"✓ Found microbenchmark for {config.name}: {microbench_path}")
        
        return all_valid

class MultiGPUSimulator:
    """Runs simulations across multiple GPU architectures"""
    
    def __init__(self, repo_root: str):
        self.discovery = MultiGPUKernelDiscovery(repo_root)
        self.config = SimulationConfig()
        self.results = []
        
    def run_single_simulation(self, kernel_info: KernelInfo, gpu_config: GPUConfig) -> Optional[Dict]:
        """Run simulation for a single kernel on specific GPU"""
        
        try:
            # Build microbenchmark path
            microbench_path = self.discovery.microbench_dir / gpu_config.microbench_file
            
            if not microbench_path.exists():
                print(f"Warning: Microbenchmark file not found: {microbench_path}")
                # Use a fallback or skip
                return None
            
            # Build command for enhanced_ptx_simulator.py with automatic GPU architecture detection
            cmd = [
                sys.executable,
                str(self.discovery.simulator_path),
                kernel_info.ptx_path,
                "--csv", str(microbench_path),
                "--num-blocks", str(self.config.total_blocks),
                "--threads-per-block", str(self.config.threads_per_block),
                "--gpu-clock", str(gpu_config.clock_mhz),
                "--ilp", "1",
                "--tlp", "32",
                "--gpu-architecture", gpu_config.name,  # Pass GPU architecture automatically
                "--auto-detect-gpu"  # Enable auto-detection from microbenchmark filename
            ]
            
            print(f"[{gpu_config.name}] Simulating {kernel_info.kernel_id} with {gpu_config.name} hardware config...")
            
            # Run simulation 
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode != 0:
                print(f"Error running simulation for {kernel_info.kernel_id}: {result.stderr}")
                return None
            
            # Parse output to extract execution time
            execution_time_us = self._parse_execution_time(result.stdout)
            if execution_time_us is None:
                print(f"Failed to parse execution time for {kernel_info.kernel_id}")
                return None
            
            # Add launch overhead (convert from μs to ms)
            total_time_us = execution_time_us + gpu_config.launch_overhead_us
            total_time_ms = total_time_us / 1000.0
            
            return {
                'gpu_architecture': gpu_config.name,
                'gpu_id': gpu_config.cuda_id,
                'kernel_i': kernel_info.i,
                'kernel_j': kernel_info.j, 
                'kernel_id': kernel_info.kernel_id,
                'grid_x': self.config.grid_x,
                'grid_y': self.config.grid_y,
                'block_x': self.config.block_x,
                'block_y': self.config.block_y,
                'total_blocks': self.config.total_blocks,
                'threads_per_block': self.config.threads_per_block,
                'total_threads': self.config.total_threads,
                'execution_time_us': execution_time_us,
                'launch_overhead_us': gpu_config.launch_overhead_us,
                'predicted_time_us': total_time_us,
                'predicted_time_ms': total_time_ms
            }
            
        except subprocess.TimeoutExpired:
            print(f"Timeout running simulation for {kernel_info.kernel_id}")
            return None
        except Exception as e:
            print(f"Unexpected error simulating {kernel_info.kernel_id}: {e}")
            return None
    
    def _parse_execution_time(self, output: str) -> Optional[float]:
        """Parse execution time from simulator output"""
        
        # Look for the final prediction time specifically
        patterns = [
            # Look for the final prediction in the results section - most specific first
            r'Final Kernel Execution Time Prediction:\s*[\s\S]*?Predicted Time:\s*([\d.]+)\s*μs',
            r'Predicted Time:\s*([\d.]+)\s*μs',  # Direct match for "Predicted Time: X.X μs"
            r'Final.*Time.*:\s*([\d.]+)\s*(?:microseconds|μs|us)',
            r'Execution Time.*:\s*([\d.]+)\s*(?:microseconds|μs|us)',
            # Fallback patterns - less specific
            r'Total time:\s*([\d.]+)\s*(?:microseconds|μs|us)',
            r'([\d.]+)\s*(?:microseconds|μs|us)'  # Last resort
        ]
        
        for i, pattern in enumerate(patterns):
            matches = re.findall(pattern, output, re.IGNORECASE | re.DOTALL)
            if matches:
                try:
                    # For the final prediction, there might be multiple "Predicted Time" lines
                    # Take the last one (which should be the final result)
                    time_val = float(matches[-1])
                    print(f"    ✅ Found execution time using pattern {i}: {time_val} μs")
                    return time_val
                except (ValueError, IndexError):
                    print(f"    ❌ Pattern {i} matched but couldn't parse: {matches}")
                    continue
        
        # If all patterns fail, print debug info
        print(f"    ❌ Failed to parse execution time from output")
        print(f"    📋 Raw output (first 1000 chars):\n{output[:1000]}")
        
        # Try to find the specific section with final results
        final_section = re.search(r'Final Kernel Execution Time Prediction:(.*?)====', output, re.DOTALL)
        if final_section:
            section_text = final_section.group(1)
            print(f"    🔍 Final section found:\n{section_text}")
            
            # Look for any time in microseconds in the final section
            time_matches = re.findall(r'([\d.]+)\s*μs', section_text)
            if time_matches:
                try:
                    time_val = float(time_matches[0])  # First time in final section should be the prediction
                    print(f"    ⚡ Using time from final section: {time_val} μs")
                    return time_val
                except ValueError:
                    pass
        
        return None
    
    def run_all_simulations(self, output_file: str = "multi_gpu_simulation_results.csv"):
        """Run simulations for all kernels on all GPUs"""
        
        if not self.discovery.validate_environment():
            print("Environment validation failed!")
            return False
        
        # Discover all kernels 
        kernels_by_gpu = self.discovery.discover_kernels()
        
        total_kernels = sum(len(kernels) for kernels in kernels_by_gpu.values())
        if total_kernels == 0:
            print("No kernels found!")
            return False
            
        print(f"\nFound {total_kernels} kernels across {len(kernels_by_gpu)} GPU architectures")
        for gpu_id, kernels in kernels_by_gpu.items():
            if kernels:
                gpu_config = GPU_CONFIGS[gpu_id]
                print(f"  {gpu_config.name}: {len(kernels)} kernels")
        
        print(f"\nFixed Configuration:")
        print(f"  Grid: {self.config.grid_x}×{self.config.grid_y} = {self.config.total_blocks} blocks")
        print(f"  Block: {self.config.block_x}×{self.config.block_y} = {self.config.threads_per_block} threads per block")
        print(f"  Total threads: {self.config.total_threads}")
        print()
        
        # Run simulations
        simulation_count = 0
        successful_count = 0
        
        for gpu_id, kernels in kernels_by_gpu.items():
            if not kernels:
                continue
                
            gpu_config = GPU_CONFIGS[gpu_id]
            print(f"\n--- Simulating {gpu_config.name} ({len(kernels)} kernels) ---")
            
            for kernel_info in kernels:
                simulation_count += 1
                result = self.run_single_simulation(kernel_info, gpu_config)
                
                if result:
                    self.results.append(result)
                    successful_count += 1
                    print(f"  ✓ {kernel_info.kernel_id}: {result['predicted_time_ms']:.3f} ms")
                else:
                    print(f"  ✗ {kernel_info.kernel_id}: Failed")
        
        # Save results
        if self.results:
            self._save_results(output_file)
            print(f"\n📊 Results saved to: {output_file}")
            print(f"Successful simulations: {successful_count}/{simulation_count}")
            
            # Print summary statistics
            self._print_summary_stats()
            return True
        else:
            print("\nNo successful simulations!")
            return False
    
    def _save_results(self, output_file: str):
        """Save results to CSV with proper error handling"""
        
        try:
            # Ensure output directory exists
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', newline='') as csvfile:
                if not self.results:
                    print("Warning: No results to save!")
                    return
                
                fieldnames = self.results[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                writer.writerows(self.results)
                
            print(f"✓ Successfully saved {len(self.results)} results to {output_file}")
            
        except PermissionError:
            print(f"Error: Permission denied writing to {output_file}")
            # Try alternative filename
            alt_file = f"results_{os.getpid()}.csv"
            try:
                with open(alt_file, 'w', newline='') as csvfile:
                    fieldnames = self.results[0].keys()
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(self.results)
                print(f"✓ Saved to alternative file: {alt_file}")
            except Exception as e:
                print(f"Error: Could not save results anywhere: {e}")
                
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def _print_summary_stats(self):
        """Print summary statistics by GPU architecture"""
        
        if not self.results:
            return
        
        print(f"\n📈 Summary Statistics:")
        print("="*60)
        
        # Group by GPU architecture
        gpu_stats = {}
        for result in self.results:
            gpu_name = result['gpu_architecture']
            if gpu_name not in gpu_stats:
                gpu_stats[gpu_name] = []
            gpu_stats[gpu_name].append(result['predicted_time_ms'])
        
        for gpu_name, times in gpu_stats.items():
            print(f"\n{gpu_name}:")
            print(f"  Count: {len(times)}")
            print(f"  Mean: {sum(times)/len(times):.3f} ms")
            print(f"  Min: {min(times):.3f} ms") 
            print(f"  Max: {max(times):.3f} ms")
            print(f"  Range: {max(times) - min(times):.3f} ms")

def main():
    parser = argparse.ArgumentParser(description='Multi-GPU Batch PTX Simulator')
    parser.add_argument('--repo-root', required=True, 
                       help='Path to GPU-Project/src directory')
    parser.add_argument('--output', default='multi_gpu_simulation_results.csv',
                       help='Output CSV file (default: multi_gpu_simulation_results.csv)')
    
    args = parser.parse_args()
    
    print("Multi-GPU Batch PTX Simulator")
    print("="*50)
    print(f"Repository root: {args.repo_root}")
    print(f"Output file: {args.output}")
    print()
    
    # Print GPU configurations
    print("GPU Configurations:")
    for gpu_id, config in GPU_CONFIGS.items():
        print(f"  {config.name}: {config.launch_overhead_us:.3f} μs launch overhead, {config.clock_mhz} MHz")
    print()
    
    simulator = MultiGPUSimulator(args.repo_root)
    
    success = simulator.run_all_simulations(args.output)
    
    if success:
        print(f"\n🎉 Multi-GPU simulation completed successfully!")
        print(f"Results saved to: {args.output}")
    else:
        print(f"\n❌ Simulation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
