#!/usr/bin/env python3
"""
Enhanced PTX Delay Simulator with Algorithm 2 Implementation
============================================================

Includes:
1. Algorithm 1: Delay Computation (from your existing code)
2. Algorithm 2: GPU Simulation with corrected wave-based scheduling
3. Kernel execution time prediction

Based on your corrected pseudo code that fixes the original paper's Algorithm 2.
"""

import pandas as pd
import numpy as np
import re
import sys
import argparse
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import csv
import math

@dataclass
class MicrobenchmarkData:
    """Stores microbenchmark measurement data for an instruction"""
    instruction: str
    latency: float
    throughput_ilp1: float
    throughput_ilp2: float  
    throughput_ilp3: float
    peak_warps_ilp1: int
    peak_warps_ilp2: int
    peak_warps_ilp3: int

@dataclass
class GPUHardwareConfig:
    """GPU hardware configuration parameters"""
    num_sms: int = 68                    # RTX 2080 Ti has 68 SMs
    cores_per_sm: int = 64              # CUDA cores per SM
    max_threads_per_sm: int = 1024      # Hardware limit
    max_warps_per_sm: int = 32          # Hardware limit (1024/32)
    max_blocks_per_sm: int = 16         # Hardware limit
    warp_size: int = 32                 # Standard warp size
    gpu_clock_mhz: float = 1755.0       # RTX 2080 Ti boost clock
    memory_bandwidth_gbps: float = 616.0 # RTX 2080 Ti memory bandwidth
    
    def warps_issuing_per_cycle(self) -> int:
        """Number of warps that can issue instructions per cycle"""
        return self.cores_per_sm // self.warp_size

@dataclass
class KernelLaunchParams:
    """CUDA kernel launch parameters"""
    num_blocks: int
    threads_per_block: int
    
    @property
    def total_threads(self) -> int:
        return self.num_blocks * self.threads_per_block
    
    @property
    def warps_per_block(self) -> int:
        return math.ceil(self.threads_per_block / 32)
    
    @property
    def total_warps(self) -> int:
        return self.num_blocks * self.warps_per_block

class ImprovedMicrobenchmarkLoader:
    """Enhanced loader that handles N.A. values and missing data"""
    
    def load_csv(self, csv_file: str) -> Dict[str, MicrobenchmarkData]:
        """Load microbenchmark data from CSV file with N.A. handling"""
        benchmark_data = {}
        
        try:
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Handle N.A. values by using fallbacks
                    def safe_float(val, fallback=0.0):
                        if val == 'N.A.' or val == '' or val is None:
                            return fallback
                        try:
                            return float(val)
                        except:
                            return fallback
                    
                    def safe_int(val, fallback=1):
                        if val == 'N.A.' or val == '' or val is None:
                            return fallback
                        try:
                            return int(float(val))
                        except:
                            return fallback
                    
                    # Extract data with fallbacks
                    instruction = row['Instruction']
                    latency = safe_float(row['Latency'], 1.0)
                    
                    # Throughput values - use ILP1 as fallback for missing ILP2/ILP3
                    throughput_ilp1 = safe_float(row['Throughput_ILP1'], 100.0)
                    throughput_ilp2 = safe_float(row['Throughput_ILP2'], throughput_ilp1)
                    throughput_ilp3 = safe_float(row['Throughput_ILP3'], throughput_ilp1)
                    
                    # Peak warps - use ILP1 as fallback
                    peak_warps_ilp1 = safe_int(row['PeakWarps_ILP1'], 32)
                    peak_warps_ilp2 = safe_int(row['PeakWarps_ILP2'], peak_warps_ilp1)
                    peak_warps_ilp3 = safe_int(row['PeakWarps_ILP3'], peak_warps_ilp1)
                    
                    data = MicrobenchmarkData(
                        instruction=instruction,
                        latency=latency,
                        throughput_ilp1=throughput_ilp1,
                        throughput_ilp2=throughput_ilp2,
                        throughput_ilp3=throughput_ilp3,
                        peak_warps_ilp1=peak_warps_ilp1,
                        peak_warps_ilp2=peak_warps_ilp2,
                        peak_warps_ilp3=peak_warps_ilp3
                    )
                    benchmark_data[instruction] = data
                        
        except Exception as e:
            print(f"Error loading CSV file {csv_file}: {e}")
            return self._get_fallback_data()
        
        return benchmark_data
    
    def _get_fallback_data(self) -> Dict[str, MicrobenchmarkData]:
        """Fallback data if CSV loading fails"""
        return {
            'FADD': MicrobenchmarkData('FADD', 4.0, 60.0, 60.0, 60.0, 12, 12, 12),
            'FMUL': MicrobenchmarkData('FMUL', 4.0, 60.0, 60.0, 60.0, 12, 12, 12),
            'MOV': MicrobenchmarkData('MOV', 0.03, 1000.0, 1000.0, 1000.0, 24, 24, 24),
        }

class DelayCalculator:
    """Calculates d_i values using Formula (3)"""
    
    def __init__(self, warp_size: int = 32):
        self.warp_size = warp_size
    
    def calculate_di_formula3(self, latency_cycles: float, throughput_ops_per_cycle: float, 
                             peak_warps: int, ilp: int, tlp: int) -> float:
        """Calculate d_i using Formula (3) from research paper"""
        ilp_tlp_product = ilp * tlp
        
        if ilp_tlp_product <= peak_warps:
            # Case 1: ILP × TLP ≤ m_i^c
            return latency_cycles / ilp_tlp_product if ilp_tlp_product > 0 else latency_cycles
        else:
            # Case 2: ILP × TLP > m_i^c
            first_term = latency_cycles / (ilp_tlp_product * peak_warps) if (ilp_tlp_product * peak_warps) > 0 else 0
            second_term = self.warp_size / throughput_ops_per_cycle if throughput_ops_per_cycle > 0 else 0
            return first_term + second_term

@dataclass
class Instruction:
    """Represents a PTX instruction"""
    opcode: str
    operands: str
    delay: float = 0.0
    line_number: int = 0

@dataclass  
class BasicBlock:
    """Represents a basic block in PTX code"""
    label: str
    instructions: List[Instruction]
    has_loop: bool = False
    loop_iterations: int = 1

class ImprovedPTXParser:
    """Enhanced PTX parser for real compiler output"""
    
    # Enhanced instruction mapping for real PTX
    INSTRUCTION_MAPPING = {
        # Floating point arithmetic
        'add.f32': 'FADD', 'add.rn.f32': 'FADD', 'add.f64': 'FADD',
        'sub.f32': 'FSUB', 'sub.rn.f32': 'FSUB', 'sub.f64': 'FSUB',
        'mul.f32': 'FMUL', 'mul.rn.f32': 'FMUL', 'mul.f64': 'FMUL',
        'fma.rn.f32': 'FMA', 'fma.rn.f64': 'FMA',
        
        # Integer arithmetic  
        'add.s32': 'ADD.S32', 'add.u32': 'ADD.S32', 'add.s64': 'ADD.S64',
        'sub.s32': 'SUB.S32', 'sub.u32': 'SUB.S32',
        'mul.lo.s32': 'MUL.LO.S32', 'mul.lo.u32': 'MUL.LO.S32', 'mul.lo.s64': 'MUL.LO.S32',
        'mul.wide.s32': 'MUL.WIDE.S32', 'mul.wide.u32': 'MUL.WIDE.S32',
        'mad.lo.s32': 'MAD.LO.S32', 'mad.lo.u32': 'MAD.LO.S32',
        
        # Comparison and conversion
        'setp.eq.s32': 'SETP', 'setp.ne.s32': 'SETP', 'setp.lt.s32': 'SETP',
        'setp.le.s32': 'SETP', 'setp.gt.s32': 'SETP', 'setp.ge.s32': 'SETP',
        'setp.eq.f32': 'SETP', 'setp.ne.f32': 'SETP', 'setp.lt.f32': 'SETP',
        'cvt.s32.f32': 'CVT', 'cvt.f32.s32': 'CVT', 'cvt.s64.s32': 'CVT',
        'cvt.rn.f32.f64': 'CVT', 'cvta.to.global.u64': 'CVT',
        
        # Movement and memory
        'mov.u32': 'MOV', 'mov.s32': 'MOV', 'mov.f32': 'MOV', 'mov.b32': 'MOV',
        'mov.u64': 'MOV', 'mov.s64': 'MOV', 'mov.f64': 'MOV', 'mov.b64': 'MOV',
        'ld.global.f32': 'MOV', 'ld.global.s32': 'MOV', 'ld.global.u32': 'MOV',
        'st.global.f32': 'MOV', 'st.global.s32': 'MOV', 'st.global.u32': 'MOV',
        'ld.param.u64': 'MOV', 'ld.param.u32': 'MOV', 'ld.param.f32': 'MOV',
        
        # Bit operations
        'shl.b64': 'ADD.S32', 'shr.u32': 'ADD.S32', 'and.b32': 'AND',
        
        # Special functions
        'div.approx.f32': 'DIVF', 'div.rn.f32': 'DIVF',
        'sqrt.approx.f32': 'SQRT', 'sqrt.rn.f32': 'SQRT',
        'rsqrt.approx.f32': 'SQRT',
        
        # Integer division (map to DIVS if available, else DIVF)
        'div.s32': 'DIVS', 'div.u32': 'DIVS',
    }
    
    def parse_ptx_file(self, ptx_content: str) -> List[BasicBlock]:
        """Parse PTX content with improved handling for real compiler output"""
        lines = ptx_content.strip().split('\n')
        
        # Find function start - look for .visible .entry or .entry
        function_start = -1
        brace_count = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            if '.entry' in line or ('.visible' in line and '(' in line):
                function_start = i
                break
        
        if function_start == -1:
            print("Warning: No .entry function found in PTX file")
            return []
        
        # Find the opening brace of the function
        function_body_start = -1
        for i in range(function_start, len(lines)):
            if '{' in lines[i]:
                function_body_start = i + 1
                break
        
        if function_body_start == -1:
            print("Warning: No function body found")
            return []
        
        # Extract function body until matching closing brace
        body_lines = []
        brace_count = 1
        
        for i in range(function_body_start, len(lines)):
            line = lines[i].strip()
            if not line:
                continue
            
            if '{' in line:
                brace_count += 1
            if '}' in line:
                brace_count -= 1
                if brace_count == 0:
                    break
                    
            body_lines.append((line, i))
        
        return self._parse_basic_blocks(body_lines)
    
    def _parse_basic_blocks(self, body_lines: List[Tuple[str, int]]) -> List[BasicBlock]:
        """Parse function body into basic blocks"""
        basic_blocks = []
        current_block = None
        block_counter = 0
        
        for line, line_num in body_lines:
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('//'):
                continue
                
            # Check if this is a label (ends with :)
            if line.endswith(':'):
                # Save previous block if exists
                if current_block and current_block.instructions:
                    basic_blocks.append(current_block)
                
                # Start new block
                label = line[:-1].strip()
                current_block = BasicBlock(label, [])
                continue
            
            # If no current block, create one (unlabeled block)
            if current_block is None:
                block_counter += 1
                current_block = BasicBlock(f"block_{block_counter}", [])
            
            # Parse instruction
            inst = self._parse_instruction(line, line_num)
            if inst:
                current_block.instructions.append(inst)
            
            # Check for block terminators
            if any(term in line for term in ['ret;', 'exit;', 'bra ', '@']):
                if current_block.instructions:
                    basic_blocks.append(current_block)
                current_block = None
        
        # Add final block
        if current_block and current_block.instructions:
            basic_blocks.append(current_block)
        
        return basic_blocks
    
    def _parse_instruction(self, line: str, line_num: int) -> Optional[Instruction]:
        """Parse a single PTX instruction"""
        # Remove comments and extra whitespace
        if '//' in line:
            line = line[:line.find('//')]
        line = line.strip()
        
        if not line or line.endswith(':'):
            return None
        
        # Split into opcode and operands
        parts = line.split(None, 1)
        if not parts:
            return None
            
        opcode = parts[0]
        operands = parts[1] if len(parts) > 1 else ""
        
        # Remove semicolon
        operands = operands.rstrip(';')
        
        return Instruction(opcode=opcode, operands=operands, line_number=line_num)
    
    def map_instruction_to_benchmark(self, ptx_opcode: str) -> str:
        """Map PTX opcode to microbenchmark instruction"""
        return self.INSTRUCTION_MAPPING.get(ptx_opcode, 'MOV')  # Default to MOV

class GPUSimulator:
    """Implements corrected Algorithm 2 for GPU simulation"""
    
    def __init__(self, hardware_config: GPUHardwareConfig):
        self.hardware = hardware_config
    
    def can_schedule_block(self, sm_workload: int, launch_params: KernelLaunchParams) -> bool:
        """Check if a block can be scheduled on an SM based on resource constraints"""
        # Check maximum blocks per SM
        if sm_workload >= self.hardware.max_blocks_per_sm:
            return False
        
        # Check if there are enough warps available
        warps_needed = launch_params.warps_per_block
        current_warps = sm_workload * warps_needed
        if current_warps + warps_needed > self.hardware.max_warps_per_sm:
            return False
        
        # Check thread count
        threads_needed = launch_params.threads_per_block
        current_threads = sm_workload * threads_needed
        if current_threads + threads_needed > self.hardware.max_threads_per_sm:
            return False
        
        return True
    
    def simulate_kernel_execution(self, dk: float, launch_params: KernelLaunchParams) -> Dict:
        """
        Corrected Algorithm 2: GPU Simulation with wave-based scheduling
        
        Args:
            dk: Total kernel delay from Algorithm 1 (cycles)
            launch_params: Kernel launch parameters
            
        Returns:
            Dict with simulation results
        """
        print(f"\n=== Algorithm 2: GPU Simulation ===")
        print(f"Total blocks: {launch_params.num_blocks}")
        print(f"Threads per block: {launch_params.threads_per_block}")
        print(f"Warps per block: {launch_params.warps_per_block}")
        print(f"Hardware: {self.hardware.num_sms} SMs, {self.hardware.max_blocks_per_sm} max blocks/SM")
        
        # Calculate single thread execution time
        t_thread = dk / (self.hardware.gpu_clock_mhz * 1e6)  # Convert to seconds
        
        # Algorithm 2 implementation (corrected version)
        ts_kernel = 0.0
        next_available_block = 1
        num_blocks = launch_params.num_blocks
        wave_number = 1
        
        while next_available_block <= num_blocks:
            print(f"\n--- Wave {wave_number} ---")
            
            # Reset counters for the new scheduling wave
            blocks_scheduled_this_wave = 0
            sm_workload = [0] * self.hardware.num_sms  # Track work assigned to each SM
            
            # Assignment Phase: Load blocks onto SMs concurrently
            for sm_id in range(self.hardware.num_sms):
                current_block = next_available_block
                
                # Greedily load blocks onto the current SM until it's full
                while (current_block <= num_blocks and 
                       self.can_schedule_block(sm_workload[sm_id], launch_params)):
                    
                    sm_workload[sm_id] += 1                     # Workload assigned to this specific SM
                    blocks_scheduled_this_wave += 1             # Total blocks assigned in this wave
                    next_available_block += 1                   # Move to the next waiting block
                    current_block = next_available_block
                    
                    # If all blocks are assigned, break the outer SM loop
                    if next_available_block > num_blocks:
                        break
                
                # If all blocks are assigned, no need to check other SMs
                if next_available_block > num_blocks:
                    break
            
            # Time Calculation Phase: The wave finishes when the slowest SM is done
            max_blocks_on_sm = max(sm_workload)  # Finds the bottleneck SM
            t_wave = max_blocks_on_sm * t_thread
            
            print(f"  Blocks scheduled: {blocks_scheduled_this_wave}")
            print(f"  SM workloads: {sm_workload[:8]}...")  # Show first 8 SMs
            print(f"  Bottleneck SM has: {max_blocks_on_sm} blocks")
            print(f"  Wave time: {t_wave*1e6:.3f} μs")
            
            ts_kernel += t_wave
            wave_number += 1
        
        # Add kernel launch overhead and memory bottleneck penalties
        # Based on equations from paper
        launch_overhead = self._calculate_launch_overhead(launch_params)
        memory_penalty = self._calculate_memory_penalty(launch_params)
        
        t_kernel = ts_kernel + launch_overhead + memory_penalty
        
        results = {
            'scheduling_time_seconds': ts_kernel,
            'launch_overhead_seconds': launch_overhead,
            'memory_penalty_seconds': memory_penalty,
            'total_kernel_time_seconds': t_kernel,
            'total_kernel_time_microseconds': t_kernel * 1e6,
            'total_kernel_time_milliseconds': t_kernel * 1e3,
            'single_thread_time_seconds': t_thread,
            'waves_executed': wave_number - 1,
            'speedup_vs_sequential': (launch_params.total_threads * t_thread) / t_kernel,
            'hardware_utilization': launch_params.num_blocks / (self.hardware.num_sms * self.hardware.max_blocks_per_sm),
            'launch_params': launch_params
        }
        
        return results
    
    def _calculate_launch_overhead(self, launch_params: KernelLaunchParams) -> float:
        """Calculate kernel launch overhead based on paper's linear model"""
        # From paper: l_overhead = 1.260e-08 * nt + 4.260e-2 (in seconds)
        nt = launch_params.total_threads
        return 1.260e-08 * nt + 4.260e-02
    
    def _calculate_memory_penalty(self, launch_params: KernelLaunchParams) -> float:
        """Calculate memory bottleneck penalty (simplified model)"""
        # This is a placeholder - in practice this would be based on
        # memory access patterns, stride, coalescing, etc.
        # For now, assume a small penalty proportional to thread count
        return launch_params.total_threads * 1e-9  # Very small penalty

class EnhancedDelaySimulator:
    """Enhanced simulator with both Algorithm 1 and Algorithm 2"""
    
    def __init__(self, csv_file: Optional[str] = None, 
                 gpu_clock_mhz: float = 1755.0,
                 ilp: int = 1, tlp: int = 32):
        
        self.gpu_clock_mhz = gpu_clock_mhz
        self.ilp = ilp
        self.tlp = tlp
        
        # Load microbenchmark data
        loader = ImprovedMicrobenchmarkLoader()
        self.benchmark_data = loader.load_csv(csv_file) if csv_file else loader._get_fallback_data()
        
        # Calculate d_i values
        self.delay_calculator = DelayCalculator()
        self.instruction_delays = self._calculate_all_delays()
        
        # Initialize components
        self.parser = ImprovedPTXParser()
        self.hardware = GPUHardwareConfig(gpu_clock_mhz=gpu_clock_mhz)
        self.gpu_simulator = GPUSimulator(self.hardware)
        
        print(f"Loaded {len(self.benchmark_data)} microbenchmark entries")
    
    def _calculate_all_delays(self) -> Dict[str, float]:
        """Calculate d_i values for all benchmark instructions"""
        delays = {}
        
        print(f"\nCalculating d_i values with ILP={self.ilp}, TLP={self.tlp}")
        print("=" * 70)
        
        for inst_name, data in self.benchmark_data.items():
            # Choose throughput and peak_warps based on ILP level
            if self.ilp == 1:
                throughput = data.throughput_ilp1
                peak_warps = data.peak_warps_ilp1
            elif self.ilp == 2:
                throughput = data.throughput_ilp2
                peak_warps = data.peak_warps_ilp2
            else:
                throughput = data.throughput_ilp3
                peak_warps = data.peak_warps_ilp3
            
            di = self.delay_calculator.calculate_di_formula3(
                data.latency, throughput, peak_warps, self.ilp, self.tlp
            )
            delays[inst_name] = di
            
            print(f"{inst_name:<15}: latency={data.latency:6.2f}, "
                  f"throughput={throughput:8.1f}, peak_warps={peak_warps:2d}, "
                  f"d_i={di:8.4f} cycles")
        
        return delays
    
    def simulate_full_kernel_execution(self, ptx_content: str, 
                                      launch_params: KernelLaunchParams,
                                      loop_iterations: Optional[Dict[str, int]] = None) -> Dict:
        """
        Full kernel execution simulation using both algorithms
        
        Args:
            ptx_content: PTX code content
            launch_params: CUDA kernel launch parameters
            loop_iterations: Custom loop iteration counts
            
        Returns:
            Complete simulation results
        """
        
        # Algorithm 1: Delay Computation
        print(f"\n{'='*80}")
        print("FULL KERNEL EXECUTION SIMULATION")
        print(f"{'='*80}")
        
        delay_results = self.simulate_ptx_execution(ptx_content, loop_iterations)
        dk = delay_results['total_kernel_delay_cycles']
        
        # Algorithm 2: GPU Simulation
        gpu_results = self.gpu_simulator.simulate_kernel_execution(dk, launch_params)
        
        # Combine results
        combined_results = {
            **delay_results,
            **gpu_results,
            'algorithm_1_delay_cycles': dk,
            'algorithm_2_scheduling_time': gpu_results['scheduling_time_seconds'],
            'predicted_kernel_time_seconds': gpu_results['total_kernel_time_seconds'],
            'predicted_kernel_time_microseconds': gpu_results['total_kernel_time_microseconds'],
            'predicted_kernel_time_milliseconds': gpu_results['total_kernel_time_milliseconds']
        }
        
        return combined_results
    
    def simulate_ptx_execution(self, ptx_content: str, 
                              loop_iterations: Optional[Dict[str, int]] = None) -> Dict:
        """Algorithm 1: Delay computation (from your existing code)"""
        
        # Parse PTX code
        basic_blocks = self.parser.parse_ptx_file(ptx_content)
        
        if not basic_blocks:
            print("Warning: No basic blocks found in PTX file!")
            return self._empty_results()
        
        # Apply custom loop iterations
        if loop_iterations:
            for block in basic_blocks:
                if block.label in loop_iterations:
                    block.loop_iterations = loop_iterations[block.label]
                    block.has_loop = True
        
        # Algorithm 1: Delay Calculation
        dk = 0.0  # Total kernel delay
        block_delays = {}
        instruction_count = 0
        
        print(f"\n=== Algorithm 1: PTX Delay Simulation ===")
        print(f"ILP={self.ilp}, TLP={self.tlp}, GPU Clock={self.gpu_clock_mhz}MHz")
        print(f"Processing {len(basic_blocks)} basic blocks...")
        
        for block in basic_blocks:
            dB = 0.0  # Basic block delay
            
            print(f"\nBasic Block: {block.label}")
            print(f"  Instructions: {len(block.instructions)}")
            
            for inst in block.instructions:
                # Map PTX instruction to benchmark instruction
                bench_inst = self.parser.map_instruction_to_benchmark(inst.opcode)
                di = self.instruction_delays.get(bench_inst, 0.1)  # Small default if not found
                dB += di
                instruction_count += 1
                
                print(f"    {inst.opcode:<20} -> {bench_inst:<15} d_i={di:8.4f}")
            
            # Apply loop multiplier
            if block.has_loop:
                print(f"  Loop detected, iterations: {block.loop_iterations}")
                dB *= block.loop_iterations
            
            print(f"  Block delay: {dB:8.3f} cycles")
            dk += dB
            block_delays[block.label] = dB
        
        # Calculate execution time
        tthread_seconds = dk / (self.gpu_clock_mhz * 1e6)
        
        results = {
            'total_kernel_delay_cycles': dk,
            'single_thread_time_seconds': tthread_seconds,
            'single_thread_time_microseconds': tthread_seconds * 1e6,
            'single_thread_time_nanoseconds': tthread_seconds * 1e9,
            'basic_blocks': len(basic_blocks),
            'instruction_count': instruction_count,
            'block_delays': block_delays,
            'gpu_clock_mhz': self.gpu_clock_mhz,
            'ilp': self.ilp,
            'tlp': self.tlp
        }
        
        return results
    
    def _empty_results(self) -> Dict:
        """Return empty results when parsing fails"""
        return {
            'total_kernel_delay_cycles': 0.0,
            'single_thread_time_seconds': 0.0,
            'single_thread_time_microseconds': 0.0,
            'single_thread_time_nanoseconds': 0.0,
            'basic_blocks': 0,
            'instruction_count': 0,
            'block_delays': {},
            'gpu_clock_mhz': self.gpu_clock_mhz,
            'ilp': self.ilp,
            'tlp': self.tlp
        }
    
    def print_full_results(self, results: Dict):
        """Print comprehensive simulation results for both algorithms"""
        print(f"\n{'='*80}")
        print(f"COMPLETE KERNEL EXECUTION PREDICTION RESULTS")
        print(f"{'='*80}")
        
        # Hardware configuration
        print(f"Hardware Configuration:")
        print(f"  GPU Model:                RTX 2080 Ti")
        print(f"  GPU Clock Frequency:      {results['gpu_clock_mhz']:8.1f} MHz") 
        print(f"  Number of SMs:            {self.hardware.num_sms:8d}")
        print(f"  Max Blocks per SM:        {self.hardware.max_blocks_per_sm:8d}")
        print(f"  Max Warps per SM:         {self.hardware.max_warps_per_sm:8d}")
        print(f"  Instruction Level Par.:   {results['ilp']:8d}")
        print(f"  Thread Level Par.:        {results['tlp']:8d}")
        
        # Launch parameters
        launch_params = results.get('launch_params')
        if launch_params:
            print(f"\nKernel Launch Parameters:")
            print(f"  Number of Blocks:         {launch_params.num_blocks:8d}")
            print(f"  Threads per Block:        {launch_params.threads_per_block:8d}")
            print(f"  Total Threads:            {launch_params.total_threads:8d}")
            print(f"  Total Warps:              {launch_params.total_warps:8d}")
        
        # Algorithm 1 results
        print(f"\nAlgorithm 1 (Delay Computation):")
        print(f"  Basic Blocks:             {results['basic_blocks']:8d}")
        print(f"  Instructions:             {results['instruction_count']:8d}")
        print(f"  Total Delay (dk):         {results['total_kernel_delay_cycles']:8.3f} cycles")
        print(f"  Single Thread Time:       {results['single_thread_time_nanoseconds']:8.1f} ns")
        
        # Algorithm 2 results
        if 'waves_executed' in results:
            print(f"\nAlgorithm 2 (GPU Simulation):")
            print(f"  Waves Executed:           {results['waves_executed']:8d}")
            print(f"  Scheduling Time:          {results['scheduling_time_seconds']*1e6:8.3f} μs")
            print(f"  Launch Overhead:          {results['launch_overhead_seconds']*1e6:8.3f} μs")
            print(f"  Memory Penalty:           {results['memory_penalty_seconds']*1e6:8.3f} μs")
            print(f"  Hardware Utilization:     {results['hardware_utilization']*100:8.1f}%")
            print(f"  Speedup vs Sequential:    {results['speedup_vs_sequential']:8.1f}x")
        
        # Final prediction
        print(f"\nFinal Kernel Execution Time Prediction:")
        print(f"  Predicted Time:           {results['predicted_kernel_time_microseconds']:8.3f} μs")
        print(f"  Predicted Time:           {results['predicted_kernel_time_milliseconds']:8.6f} ms")
        print(f"  Predicted Time:           {results['predicted_kernel_time_seconds']:12.9f} s")
        
        if results['block_delays']:
            print(f"\nPer-Block Analysis:")
            for block, delay in results['block_delays'].items():
                print(f"  {block:<20}: {delay:8.3f} cycles")

def main():
    """Main function with enhanced command line interface"""
    parser = argparse.ArgumentParser(description='Enhanced PTX Kernel Execution Time Predictor')
    parser.add_argument('ptx_file', nargs='?', help='PTX file to analyze')
    parser.add_argument('--csv', help='CSV file with microbenchmark results')
    parser.add_argument('--gpu-clock', type=float, default=1755.0, help='GPU clock frequency in MHz (default: RTX 2080 Ti boost clock)')
    parser.add_argument('--ilp', type=int, default=1, help='Instruction Level Parallelism')
    parser.add_argument('--tlp', type=int, default=32, help='Thread Level Parallelism')
    
    # Kernel launch parameters
    parser.add_argument('--num-blocks', type=int, default=256, help='Number of thread blocks')
    parser.add_argument('--threads-per-block', type=int, default=256, help='Threads per block')
    
    args = parser.parse_args()
    
    print("Enhanced PTX Kernel Execution Time Predictor")
    print("Algorithms 1 & 2 with corrected GPU simulation")
    print("="*80)
    
    # Create simulator
    simulator = EnhancedDelaySimulator(
        csv_file=args.csv,
        gpu_clock_mhz=args.gpu_clock,
        ilp=args.ilp, 
        tlp=args.tlp
    )
    
    # Load PTX code
    if args.ptx_file:
        try:
            with open(args.ptx_file, 'r') as f:
                ptx_content = f.read()
            print(f"\nLoaded PTX file: {args.ptx_file}")
        except FileNotFoundError:
            print(f"Error: PTX file {args.ptx_file} not found")
            return
    else:
        print("Error: Please provide a PTX file")
        return
    
    # Create launch parameters
    launch_params = KernelLaunchParams(
        num_blocks=args.num_blocks,
        threads_per_block=args.threads_per_block
    )
    
    # Run full simulation
    results = simulator.simulate_full_kernel_execution(ptx_content, launch_params)
    
    # Print results
    simulator.print_full_results(results)
    
    print(f"\n{'='*80}")
    print("Kernel Execution Time Prediction Complete!")
    print(f"Launch: {args.num_blocks} blocks × {args.threads_per_block} threads")
    print(f"Hardware: RTX 2080 Ti @ {args.gpu_clock}MHz")
    print(f"Parameters: ILP={args.ilp}, TLP={args.tlp}")
    print("="*80)

if __name__ == "__main__":
    main()
