#!/usr/bin/env python3

import os
import subprocess
import csv
import re
from pathlib import Path
from typing import Optional, Tuple


def parse_output(output: str) -> Optional[Tuple[float, float]]:
    """
    Parse kernel output in format [time_us, time_ms]
    Handles both regular decimal and scientific notation
    
    Args:
        output: Output string from kernel execution
        
    Returns:
        Tuple of (time_us, time_ms) or None if parsing fails
    """
    pattern = r'\[([\d.]+(?:[eE][+-]?\d+)?),([\d.]+(?:[eE][+-]?\d+)?)\]'
    match = re.search(pattern, output)
    
    if match:
        try:
            time_us = float(match.group(1))
            time_ms = float(match.group(2))
            return (time_us, time_ms)
        except ValueError:
            return None
    
    return None


def compile_and_run_kernel(kernel_file: Path, output_dir: Path) -> dict:
    """
    Compile and run a single kernel file
    
    Args:
        kernel_file: Path to .cu file
        output_dir: Directory for compiled executables
        
    Returns:
        Dictionary with results
    """
    kernel_name = kernel_file.stem
    executable = output_dir / f"{kernel_name}_exec"
    
    result = {
        'filename': kernel_file.name,
        'kernel_name': kernel_name,
        'compilation_success': False,
        'execution_success': False,
        'time_us': None,
        'time_ms': None,
        'error': None
    }
    
    # Compile
    compile_cmd = [
        "nvcc",
        "-o",
        str(executable),
        str(kernel_file)
    ]
    
    try:
        compile_result = subprocess.run(
            compile_cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if compile_result.returncode != 0:
            result['error'] = f"Compilation failed: {compile_result.stderr[:200]}"
            return result
        
        result['compilation_success'] = True
    except subprocess.TimeoutExpired:
        result['error'] = "Compilation timeout"
        return result
    except Exception as e:
        result['error'] = f"Compilation error: {str(e)}"
        return result
    
    try:
        run_result = subprocess.run(
            [str(executable)],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(output_dir)
        )
        
        if run_result.returncode != 0:
            result['error'] = f"Execution failed: {run_result.stderr[:200]}"
            return result
        
        # Parse output
        parsed = parse_output(run_result.stdout)
        if parsed:
            result['execution_success'] = True
            result['time_us'], result['time_ms'] = parsed
        else:
            result['error'] = f"Could not parse output: {run_result.stdout[:200]}"
    
    except subprocess.TimeoutExpired:
        result['error'] = "Execution timeout"
    except Exception as e:
        result['error'] = f"Execution error: {str(e)}"
    
    # Cleanup executable
    if executable.exists():
        try:
            executable.unlink()
        except:
            pass
    
    return result


def main():
    """Main function to process all kernels"""
    # Setup paths
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data_b5_t15"
    output_dir = script_dir / "temp_executables"
    csv_output = script_dir / "kernel_results_b5_t15.csv"
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Find all .cu files
    cu_files = sorted(data_dir.glob("*.cu"))
    
    if not cu_files:
        print(f"No .cu files found in {data_dir}")
        return
    
    print(f"Found {len(cu_files)} kernel files")
    print(f"Processing kernels in {data_dir}...")
    print(f"Results will be saved to {csv_output}\n")
    
    results = []
    
    # Process each kernel
    for i, kernel_file in enumerate(cu_files, 1):
        print(f"[{i}/{len(cu_files)}] Processing {kernel_file.name}...", end=" ", flush=True)
        
        result = compile_and_run_kernel(kernel_file, output_dir)
        results.append(result)
        
        if result['execution_success']:
            print(f"✓ {result['time_us']:.4f} us ({result['time_ms']:.6f} ms)")
        else:
            print(f"✗ {result['error']}")
    
    # Write CSV
    print(f"\nWriting results to {csv_output}...")
    
    with open(csv_output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'filename',
            'kernel_name',
            'compilation_success',
            'execution_success',
            'time_us',
            'time_ms',
            'error'
        ])
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    # Print summary
    successful = sum(1 for r in results if r['execution_success'])
    failed = len(results) - successful
    
    print(f"\nSummary:")
    print(f"  Total kernels: {len(results)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"\nResults saved to: {csv_output}")
    
    try:
        output_dir.rmdir()
    except:
        pass


if __name__ == "__main__":
    main()

