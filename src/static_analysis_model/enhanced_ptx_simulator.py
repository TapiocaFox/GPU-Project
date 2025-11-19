#!/usr/bin/env python3
"""
Enhanced PTX Delay Simulator with Algorithm 2 Implementation
============================================================

Includes:
1. Algorithm 1: Delay Computation (from your existing code)
2. Algorithm 2: GPU Simulation with corrected wave-based scheduling
3. Kernel execution time prediction
4. Multi-GPU architecture support with automatic configuration

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
import os

# GPU Hardware Configuration System
@dataclass
class GPUHardwareConfigSpec:
    """Hardware specifications for different GPU architectures"""
    
    # Basic specifications
    name: str
    compute_capability: tuple  # (major, minor)
    clock_mhz: float
    boost_clock_mhz: float
    memory_clock_mhz: float
    
    # SM specifications  
    num_sms: int
    max_blocks_per_sm: int
    max_warps_per_sm: int
    max_threads_per_sm: int
    
    # Memory specifications
    shared_memory_per_sm_kb: int
    shared_memory_per_block_kb: int
    registers_per_sm: int
    registers_per_thread_max: int
    
    # L1/L2 cache
    l1_cache_kb: int
    l2_cache_mb: float
    
    # Memory bandwidth
    memory_bandwidth_gb_s: float
    memory_bus_width: int
    
    # Instruction throughput (instructions per clock cycle)
    fp32_cores_per_sm: int
    fp64_cores_per_sm: int
    
    # Architecture-specific features
    architecture: str
    launch_overhead_us: float  # Measured launch overhead
    
    # Default values
    warp_size: int = 32
    tensor_cores_per_sm: int = 0

# GPU configurations based on NVIDIA specifications and your measurements
GPU_HARDWARE_CONFIGS: Dict[str, GPUHardwareConfigSpec] = {
    
    'RTX_2080_Ti': GPUHardwareConfigSpec(
        name='RTX_2080_Ti',
        compute_capability=(7, 5),
        clock_mhz=1350.0,
        boost_clock_mhz=1755.0,
        memory_clock_mhz=7000.0,
        
        num_sms=68,
        max_blocks_per_sm=16,
        max_warps_per_sm=32,
        max_threads_per_sm=1024,
        
        shared_memory_per_sm_kb=96,
        shared_memory_per_block_kb=48,
        registers_per_sm=65536,
        registers_per_thread_max=255,
        
        l1_cache_kb=32,  # Per SM
        l2_cache_mb=5.5,
        
        memory_bandwidth_gb_s=616.0,
        memory_bus_width=352,
        
        fp32_cores_per_sm=64,
        fp64_cores_per_sm=2,
        tensor_cores_per_sm=8,
        
        architecture='Turing',
        launch_overhead_us=2.287  # From your measurement
    ),
    
    'TITAN_V': GPUHardwareConfigSpec(
        name='TITAN_V',
        compute_capability=(7, 0),
        clock_mhz=1200.0,
        boost_clock_mhz=1455.0,
        memory_clock_mhz=1700.0,
        
        num_sms=80,
        max_blocks_per_sm=16,
        max_warps_per_sm=32,
        max_threads_per_sm=1024,
        
        shared_memory_per_sm_kb=96,
        shared_memory_per_block_kb=48,
        registers_per_sm=65536,
        registers_per_thread_max=255,
        
        l1_cache_kb=32,
        l2_cache_mb=4.5,
        
        memory_bandwidth_gb_s=653.0,
        memory_bus_width=3072,  # HBM2
        
        fp32_cores_per_sm=64,
        fp64_cores_per_sm=32,  # Volta has more FP64 cores
        tensor_cores_per_sm=8,
        
        architecture='Volta',
        launch_overhead_us=2.051  # From your measurement
    ),
    
    'GTX_TITAN_X': GPUHardwareConfigSpec(
        name='GTX_TITAN_X',
        compute_capability=(5, 2),
        clock_mhz=1000.0,
        boost_clock_mhz=1075.0,
        memory_clock_mhz=3505.0,
        
        num_sms=24,
        max_blocks_per_sm=16,
        max_warps_per_sm=32,
        max_threads_per_sm=1024,
        
        shared_memory_per_sm_kb=96,
        shared_memory_per_block_kb=48,
        registers_per_sm=65536,
        registers_per_thread_max=255,
        
        l1_cache_kb=48,  # Maxwell has larger L1
        l2_cache_mb=3.0,
        
        memory_bandwidth_gb_s=336.5,
        memory_bus_width=384,
        
        fp32_cores_per_sm=128,  # Maxwell has more CUDA cores per SM
        fp64_cores_per_sm=4,   # Maxwell has fewer FP64 cores
        tensor_cores_per_sm=0,  # No tensor cores
        
        architecture='Maxwell',
        launch_overhead_us=2.510  # From your measurement
    ),
    
    'RTX_4070': GPUHardwareConfigSpec(
        name='RTX_4070',
        compute_capability=(8, 9),
        clock_mhz=1920.0,
        boost_clock_mhz=2475.0,
        memory_clock_mhz=10500.0,
        
        num_sms=46,
        max_blocks_per_sm=16,
        max_warps_per_sm=32,
        max_threads_per_sm=1024,
        
        shared_memory_per_sm_kb=128,  # Ada Lovelace has more shared memory
        shared_memory_per_block_kb=48,
        registers_per_sm=65536,
        registers_per_thread_max=255,
        
        l1_cache_kb=32,
        l2_cache_mb=36.0,  # Much larger L2 cache
        
        memory_bandwidth_gb_s=504.2,
        memory_bus_width=192,
        
        fp32_cores_per_sm=128,
        fp64_cores_per_sm=2,
        tensor_cores_per_sm=16,  # Ada Lovelace has more tensor cores
        
        architecture='Ada_Lovelace',
        launch_overhead_us=4.532  # From your measurement
    )
}

def get_gpu_config(gpu_name: str) -> Optional[GPUHardwareConfigSpec]:
    """Get GPU hardware configuration by name"""
    return GPU_HARDWARE_CONFIGS.get(gpu_name)

def list_supported_gpus() -> list[str]:
    """List all supported GPU architectures"""
    return list(GPU_HARDWARE_CONFIGS.keys())

def setup_gpu_hardware_config(args):
    """Setup GPU hardware configuration based on arguments"""
    
    gpu_name = args.gpu_architecture
    
    # Auto-detect GPU from microbenchmark filename if requested
    if args.auto_detect_gpu and hasattr(args, 'csv') and args.csv:
        csv_filename = os.path.basename(args.csv)
        if 'RTX_2080_Ti' in csv_filename:
            gpu_name = 'RTX_2080_Ti'
        elif 'TITAN_V' in csv_filename:
            gpu_name = 'TITAN_V'
        elif 'GTX_TITAN_X' in csv_filename:
            gpu_name = 'GTX_TITAN_X'
        elif 'RTX_4070' in csv_filename:
            gpu_name = 'RTX_4070'
        
        print(f"Auto-detected GPU from microbenchmark filename: {gpu_name}")
    
    # Get GPU configuration
    gpu_config = get_gpu_config(gpu_name)
    if not gpu_config:
        print(f"Warning: Unknown GPU {gpu_name}, using RTX 2080 Ti defaults")
        gpu_config = get_gpu_config('RTX_2080_Ti')
    
    print(f"Using {gpu_config.name} configuration:")
    print(f"  SMs: {gpu_config.num_sms}")
    print(f"  Clock: {gpu_config.boost_clock_mhz} MHz")
    print(f"  Architecture: {gpu_config.architecture}")
    print(f"  Launch overhead: {gpu_config.launch_overhead_us} μs")
    
    return gpu_config

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
    """GPU hardware configuration parameters (dynamic based on architecture)"""
    def __init__(self, gpu_config_spec: GPUHardwareConfigSpec):
        self.name = gpu_config_spec.name
        self.num_sms = gpu_config_spec.num_sms
        self.cores_per_sm = gpu_config_spec.fp32_cores_per_sm
        self.max_threads_per_sm = gpu_config_spec.max_threads_per_sm
        self.max_warps_per_sm = gpu_config_spec.max_warps_per_sm
        self.max_blocks_per_sm = gpu_config_spec.max_blocks_per_sm
        self.warp_size = gpu_config_spec.warp_size
        self.gpu_clock_mhz = gpu_config_spec.boost_clock_mhz
        self.memory_bandwidth_gbps = gpu_config_spec.memory_bandwidth_gb_s
        self.architecture = gpu_config_spec.architecture
        self.launch_overhead_us = gpu_config_spec.launch_overhead_us
        self.compute_capability = gpu_config_spec.compute_capability
        
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
        'setp.eq.u32': 'SETP', 'setp.ne.u32': 'SETP', 'setp.lt.u32': 'SETP',
        'setp.le.u32': 'SETP', 'setp.gt.u32': 'SETP', 'setp.ge.u32': 'SETP',
        'cvt.rn.f32.s32': 'CVT', 'cvt.rn.f32.u32': 'CVT', 'cvt.rn.f64.s32': 'CVT',
        
        # Memory operations (all map to MOV for simplicity)
        'ld.global.f32': 'MOV', 'ld.global.s32': 'MOV', 'ld.global.u32': 'MOV',
        'ld.shared.f32': 'MOV', 'ld.shared.s32': 'MOV', 'ld.shared.u32': 'MOV',
        'ld.param.u64': 'MOV', 'ld.param.u32': 'MOV', 'ld.param.s32': 'MOV',
        'ld.param.b32': 'MOV', 'ld.param.b64': 'MOV',
        'st.global.f32': 'MOV', 'st.global.s32': 'MOV', 'st.global.u32': 'MOV',
        'st.shared.f32': 'MOV', 'st.shared.s32': 'MOV', 'st.shared.u32': 'MOV',
        'st.param.b64': 'MOV', 'st.param.b32': 'MOV',
        'st.local.u32': 'MOV', 'st.local.s32': 'MOV',
        
        # Data movement and conversion
        'mov.u32': 'MOV.U32', 'mov.s32': 'MOV.U32', 'mov.u64': 'MOV.B64',
        'mov.b32': 'MOV.B32', 'mov.b64': 'MOV.B64', 'mov.f32': 'MOV.F32',
        'cvta.to.global.u64': 'CVT', 'cvta.global.u64': 'MOV',
        'cvta.local.u64': 'MOV', 'cvta.shared.u64': 'MOV',
        
        # Bit operations
        'shl.b32': 'MOV', 'shr.u32': 'ADD.S32', 'shr.s32': 'ADD.S32',
        'and.b32': 'AND', 'or.b32': 'AND', 'xor.b32': 'AND',
        
        # Control flow and synchronization
        'bar.sync': 'MOV', 'bra': 'MOV', 'ret': 'MOV',
        'call.uni': 'MOV', 'call': 'MOV',
        
        # Special operations
        '.reg': 'MOV', '.local': 'MOV', '.param': 'MOV', '.shared': 'MOV',
        'vprintf': 'MOV', 'tex.2d.v4.f32.f32': 'MOV'
    }
    
    def map_instruction_to_benchmark(self, ptx_instruction: str) -> str:
        """Map PTX instruction to benchmark instruction"""
        # Clean the instruction (remove operands and modifiers)
        base_inst = ptx_instruction.split()[0] if ' ' in ptx_instruction else ptx_instruction
        
        # Handle special cases and syntax
        if base_inst.startswith('@'):  # Predicated execution
            return 'MOV'
        if base_inst in ['{', '}', '(', ')', ';', ',']:  # Syntax elements
            return 'MOV'
        if base_inst.endswith(',') or base_inst.endswith(';'):
            base_inst = base_inst[:-1]
        
        # Direct mapping
        if base_inst in self.INSTRUCTION_MAPPING:
            return self.INSTRUCTION_MAPPING[base_inst]
        
        # Pattern-based fallbacks
        if 'add' in base_inst.lower():
            if 'f32' in base_inst or 'f64' in base_inst:
                return 'FADD'
            return 'ADD.S32'
        elif 'sub' in base_inst.lower():
            if 'f32' in base_inst or 'f64' in base_inst:
                return 'FSUB'
            return 'SUB.S32'
        elif 'mul' in base_inst.lower():
            if 'f32' in base_inst or 'f64' in base_inst:
                return 'FMUL'
            elif 'wide' in base_inst:
                return 'MUL.WIDE.S32'
            return 'MUL.LO.S32'
        elif 'mad' in base_inst.lower():
            return 'MAD.LO.S32'
        elif any(x in base_inst.lower() for x in ['ld.', 'st.', 'mov', 'cvta']):
            return 'MOV'
        elif 'setp' in base_inst.lower():
            return 'SETP'
        elif 'cvt' in base_inst.lower():
            return 'CVT'
        
        # Unknown instruction fallback
        return 'MOV'
    
    def parse_ptx(self, ptx_content: str) -> List[BasicBlock]:
        """Parse PTX content into basic blocks with instructions"""
        lines = ptx_content.split('\n')
        basic_blocks = []
        current_block = None
        line_number = 0
        
        for line in lines:
            line_number += 1
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('//') or line.startswith('#'):
                continue
            
            # Skip directives that don't represent computation
            if any(line.startswith(directive) for directive in ['.version', '.target', '.address_size', '.visible', '.entry', '.maxntid', '.reqntid']):
                continue
            
            # Detect basic block labels
            if (line.endswith(':') and not line.startswith('.')) or '$L_' in line:
                # Save previous block
                if current_block and current_block.instructions:
                    basic_blocks.append(current_block)
                
                # Start new block
                block_label = line.replace(':', '').strip() if ':' in line else f"block_{len(basic_blocks)+1}"
                current_block = BasicBlock(label=block_label, instructions=[])
                continue
            
            # If no block started yet, create a default one
            if current_block is None:
                current_block = BasicBlock(label="block_1", instructions=[])
            
            # Parse instruction line
            # Clean up the line
            clean_line = line.replace('\t', ' ').replace('{', ' { ').replace('}', ' } ')
            tokens = clean_line.split()
            
            # Process each meaningful token as potential instruction
            for token in tokens:
                if token and not token.isspace():
                    instruction = Instruction(
                        opcode=token,
                        operands="",
                        line_number=line_number
                    )
                    current_block.instructions.append(instruction)
        
        # Add final block
        if current_block and current_block.instructions:
            basic_blocks.append(current_block)
        
        return basic_blocks

class CorrectedAlgorithm2:
    """Your corrected Algorithm 2 implementation"""
    
    def __init__(self, hardware: GPUHardwareConfig):
        self.hardware = hardware
    
    def can_schedule_more_blocks(self, sm_workload: int) -> bool:
        """Check if SM can schedule more blocks"""
        return sm_workload < self.hardware.max_blocks_per_sm
    
    def simulate_gpu_execution(self, launch_params: KernelLaunchParams, t_thread_seconds: float) -> Dict:
        """Corrected Algorithm 2: GPU simulation with wave-based scheduling"""
        
        print(f"\n=== Algorithm 2: GPU Simulation ===")
        print(f"Total blocks: {launch_params.num_blocks}")
        print(f"Threads per block: {launch_params.threads_per_block}")
        print(f"Warps per block: {launch_params.warps_per_block}")
        print(f"Hardware: {self.hardware.num_sms} SMs, {self.hardware.max_blocks_per_sm} max blocks/SM")
        
        # Algorithm 2 variables
        ts_kernel = 0.0
        nextAvailableBlock = 1
        numblocks = launch_params.num_blocks
        wave_count = 0
        
        while nextAvailableBlock <= numblocks:
            wave_count += 1
            print(f"\n--- Wave {wave_count} ---")
            
            # Reset counters for the new scheduling wave
            blocksScheduledThisWave = 0
            SM_Workload = [0] * self.hardware.num_sms  # Track work assigned to each SM
            
            # Assignment Phase: Load blocks onto SMs concurrently
            for sm_id in range(self.hardware.num_sms):
                CurrentBlock = nextAvailableBlock
                
                # Greedily load blocks onto the current SM until it's full
                while CurrentBlock <= numblocks and self.can_schedule_more_blocks(SM_Workload[sm_id]):
                    SM_Workload[sm_id] += 1  # Workload assigned to this specific SM
                    blocksScheduledThisWave += 1  # Total blocks assigned in this wave
                    nextAvailableBlock += 1  # Move to the next waiting block
                    CurrentBlock = nextAvailableBlock
                    
                    # If all blocks are assigned, break the outer SM loop
                    if nextAvailableBlock > numblocks:
                        break
                
                # Break outer loop if all blocks assigned
                if nextAvailableBlock > numblocks:
                    break
            
            # Time Calculation Phase: The wave finishes when the slowest SM is done
            maxBlocksOnSM = max(SM_Workload)  # Finds the bottleneck SM
            t_wave = maxBlocksOnSM * t_thread_seconds
            
            print(f"  Blocks scheduled: {blocksScheduledThisWave}")
            print(f"  SM workloads: {SM_Workload[:8]}...")  # Show first 8 SMs
            print(f"  Bottleneck SM has: {maxBlocksOnSM} blocks")
            print(f"  Wave time: {t_wave*1e6:.3f} μs")
            
            ts_kernel += t_wave
        
        # Calculate additional metrics
        total_waves = wave_count
        scheduling_time = ts_kernel
        hardware_utilization = min(launch_params.num_blocks, self.hardware.num_sms * self.hardware.max_blocks_per_sm) / (self.hardware.num_sms * self.hardware.max_blocks_per_sm)
        
        # Estimate sequential execution time for speedup calculation
        sequential_time = launch_params.num_blocks * t_thread_seconds
        speedup_vs_sequential = sequential_time / ts_kernel if ts_kernel > 0 else 0
        
        return {
            'waves_executed': total_waves,
            'scheduling_time_seconds': scheduling_time,
            'hardware_utilization': hardware_utilization,
            'speedup_vs_sequential': speedup_vs_sequential,
            'total_blocks_scheduled': launch_params.num_blocks
        }

class EnhancedDelaySimulator:
    """Enhanced simulator with both Algorithm 1 and Algorithm 2"""
    
    def __init__(self, gpu_config: GPUHardwareConfig, csv_file: str = None, 
                 gpu_clock_mhz: float = None, ilp: int = 1, tlp: int = 32):
        
        # Use provided GPU configuration
        self.hardware = gpu_config
        self.gpu_clock_mhz = gpu_clock_mhz or gpu_config.gpu_clock_mhz  # Allow override
        self.ilp = ilp
        self.tlp = tlp
        
        # Load microbenchmark data
        if csv_file:
            loader = ImprovedMicrobenchmarkLoader()
            self.benchmark_data = loader.load_csv(csv_file)
            print(f"Loaded {len(self.benchmark_data)} microbenchmark entries")
        else:
            loader = ImprovedMicrobenchmarkLoader()
            self.benchmark_data = loader._get_fallback_data()
            print("Using fallback microbenchmark data")
        
        # Initialize components
        self.delay_calc = DelayCalculator()
        self.parser = ImprovedPTXParser()
        self.algorithm2 = CorrectedAlgorithm2(self.hardware)
        
        # Calculate d_i values
        self._calculate_instruction_delays()
    
    def _calculate_launch_overhead(self, launch_params: KernelLaunchParams) -> float:
        """Calculate kernel launch overhead using measured values"""
        return self.hardware.launch_overhead_us / 1e6  # Convert μs to seconds
    
    def _calculate_memory_penalty(self, launch_params: KernelLaunchParams) -> float:
        """Estimate memory access penalty"""
        # Simple model: penalty based on total threads accessing memory
        # This is a simplified model - you can make this more sophisticated
        memory_accesses_per_thread = 2  # Assume 2 memory accesses per thread on average
        total_memory_accesses = launch_params.total_threads * memory_accesses_per_thread
        
        # Memory penalty in seconds (rough estimate)
        penalty_per_access = 1e-9  # 1 nanosecond per memory access
        return total_memory_accesses * penalty_per_access * launch_params.threads_per_block / 1000.0
    
    def _calculate_instruction_delays(self):
        """Calculate d_i values for all instructions using Formula (3)"""
        print(f"\nCalculating d_i values with ILP={self.ilp}, TLP={self.tlp}")
        print("="*70)
        
        self.instruction_delays = {}
        
        for instruction, data in self.benchmark_data.items():
            # Choose appropriate values based on ILP
            if self.ilp == 1:
                throughput = data.throughput_ilp1
                peak_warps = data.peak_warps_ilp1
            elif self.ilp == 2:
                throughput = data.throughput_ilp2
                peak_warps = data.peak_warps_ilp2
            else:  # ILP == 3
                throughput = data.throughput_ilp3
                peak_warps = data.peak_warps_ilp3
            
            # Calculate d_i using Formula (3)
            di = self.delay_calc.calculate_di_formula3(
                data.latency, throughput, peak_warps, self.ilp, self.tlp
            )
            
            self.instruction_delays[instruction] = di
            print(f"{instruction:<15}: latency={data.latency:6.2f}, throughput={throughput:8.1f}, peak_warps={peak_warps:2d}, d_i={di:8.4f} cycles")
    
    def simulate_full_kernel_execution(self, ptx_content: str, launch_params: KernelLaunchParams, 
                                     loop_iterations: Dict[str, int] = None) -> Dict:
        """Complete kernel execution simulation combining Algorithm 1 and 2"""
        
        print(f"\n{'='*80}")
        print(f"FULL KERNEL EXECUTION SIMULATION")
        print(f"{'='*80}")
        
        # Parse PTX code
        basic_blocks = self.parser.parse_ptx(ptx_content)
        
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
        
        # Calculate single thread execution time
        tthread_seconds = dk / (self.gpu_clock_mhz * 1e6)
        
        # Algorithm 2: GPU Simulation
        algorithm2_results = self.algorithm2.simulate_gpu_execution(launch_params, tthread_seconds)
        
        # Calculate overheads
        launch_overhead_seconds = self._calculate_launch_overhead(launch_params)
        memory_penalty_seconds = self._calculate_memory_penalty(launch_params)
        
        # Final kernel execution time prediction
        predicted_kernel_time = (algorithm2_results['scheduling_time_seconds'] + 
                                launch_overhead_seconds + 
                                memory_penalty_seconds)
        
        # Combine all results
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
            'tlp': self.tlp,
            'launch_params': launch_params,
            'gpu_hardware_config': self.hardware,
            
            # Algorithm 2 results
            'waves_executed': algorithm2_results['waves_executed'],
            'scheduling_time_seconds': algorithm2_results['scheduling_time_seconds'],
            'hardware_utilization': algorithm2_results['hardware_utilization'],
            'speedup_vs_sequential': algorithm2_results['speedup_vs_sequential'],
            
            # Overhead calculations
            'launch_overhead_seconds': launch_overhead_seconds,
            'memory_penalty_seconds': memory_penalty_seconds,
            
            # Final prediction
            'predicted_kernel_time_seconds': predicted_kernel_time,
            'predicted_kernel_time_microseconds': predicted_kernel_time * 1e6,
            'predicted_kernel_time_milliseconds': predicted_kernel_time * 1e3,
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
        print(f"  GPU Model:                {self.hardware.name}")
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
    parser.add_argument('--gpu-clock', type=float, help='GPU clock frequency in MHz (overrides architecture default)')
    parser.add_argument('--ilp', type=int, default=1, help='Instruction Level Parallelism')
    parser.add_argument('--tlp', type=int, default=32, help='Thread Level Parallelism')
    
    # GPU architecture selection
    parser.add_argument('--gpu-architecture', type=str, default='RTX_2080_Ti',
                       choices=['RTX_2080_Ti', 'TITAN_V', 'GTX_TITAN_X', 'RTX_4070'],
                       help='GPU architecture for hardware configuration')
    parser.add_argument('--auto-detect-gpu', action='store_true',
                       help='Auto-detect GPU from microbenchmark filename')
    
    # Kernel launch parameters
    parser.add_argument('--num-blocks', type=int, default=256, help='Number of thread blocks')
    parser.add_argument('--threads-per-block', type=int, default=256, help='Threads per block')
    
    args = parser.parse_args()
    
    print("Enhanced PTX Kernel Execution Time Predictor")
    print("Algorithms 1 & 2 with corrected GPU simulation")
    print("="*80)
    
    # Setup GPU hardware configuration
    gpu_config_spec = setup_gpu_hardware_config(args)
    gpu_hardware_config = GPUHardwareConfig(gpu_config_spec)
    
    # Create simulator
    simulator = EnhancedDelaySimulator(
        gpu_config=gpu_hardware_config,
        csv_file=args.csv,
        gpu_clock_mhz=args.gpu_clock,  # Can override architecture default
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
    print(f"Hardware: {gpu_config_spec.name} @ {simulator.gpu_clock_mhz}MHz")
    print(f"Parameters: ILP={args.ilp}, TLP={args.tlp}")
    print("="*80)

if __name__ == "__main__":
    main()
