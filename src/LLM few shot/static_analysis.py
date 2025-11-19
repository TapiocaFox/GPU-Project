#!/usr/bin/env python3
"""
Static Analysis Only - CUDA Kernel Execution Time Prediction
No LLM required - uses only regex parsing and numerical formulas
"""

import csv
import argparse
from pathlib import Path
from typing import Dict, List, Optional

# Import our modules
from predict_cuda_kernel_time import KernelParameters, calculate_manual_prediction
from ptx_parser import PTXParser


class StaticAnalyzer:
    """Static analyzer for CUDA kernels - no LLM, pure numerical analysis"""
    
    def __init__(self, data_dir: str = "data_b5_t15"):
        """
        Initialize static analyzer
        
        Args:
            data_dir: Directory containing CUDA/PTX files
        """
        self.data_dir = Path(data_dir)
        self.parser = PTXParser()
        self.results: List[Dict] = []
    
    def get_kernel_files(self) -> List[Path]:
        """Get all CUDA/PTX files from data directory"""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Find all .cu and .ptx files
        files = list(self.data_dir.glob("*.cu")) + list(self.data_dir.glob("*.ptx"))
        return sorted(files)
    
    def process_file(self, file_path: Path) -> Optional[Dict]:
        """
        Process a single kernel file using static analysis only
        
        Args:
            file_path: Path to kernel file
        
        Returns:
            Dictionary with results or None if failed
        """
        print(f"Processing: {file_path.name}")
        
        try:
            # Extract parameters using regex (no LLM)
            print(f"  Extracting parameters...")
            extracted = self.parser.parse_ptx_file(str(file_path))
            
            print(f"  Parameters: comp={extracted['comp']:.1f}, "
                  f"ld1={extracted['ld1']:.1f}, st1={extracted['st1']:.1f}, "
                  f"ld0={extracted['ld0']:.1f}, st0={extracted['st0']:.1f}")
            
            # Create kernel parameters
            params = KernelParameters(
                comp=extracted['comp'],
                ld1=extracted['ld1'],
                st1=extracted['st1'],
                ld0=extracted['ld0'],
                st0=extracted['st0'],
                L1=extracted['L1'],
                L2=extracted['L2']
            )
            
            # Calculate execution time using formulas
            print(f"  Calculating execution time...")
            manual_result = calculate_manual_prediction(params)
            
            print(f"  Execution time: {manual_result['execution_time']:.6f} {manual_result['units']}")
            
            # Compile results
            result = {
                'filename': file_path.name,
                'filepath': str(file_path),
                # Extracted parameters
                'comp': extracted['comp'],
                'ld1': extracted['ld1'],
                'st1': extracted['st1'],
                'ld0': extracted['ld0'],
                'st0': extracted['st0'],
                'L1': extracted['L1'],
                'L2': extracted['L2'],
                # Calculation results
                'execution_time': manual_result['execution_time'],
                'comm_gm': manual_result['comm_gm'],
                'comm_sm': manual_result['comm_sm'],
                'total_cost': manual_result['total_cost'],
                'units': manual_result['units'],
                # Architecture parameters used
                'R': params.R,
                'P': params.P,
                'lambda': params.lambda_val,
                'g_GM': params.g_GM,
                'g_L1': params.g_L1,
                'g_L2': params.g_L2,
                'g_SM': params.g_SM,
            }
            
            return result
            
        except Exception as e:
            print(f"  ERROR: {e}")
            return {
                'filename': file_path.name,
                'filepath': str(file_path),
                'error': str(e)
            }
    
    def process_all(self, output_csv: str = "static_analysis_results.csv"):
        """
        Process all kernel files and save to CSV
        
        Args:
            output_csv: Output CSV file path
        """
        files = self.get_kernel_files()
        print(f"Found {len(files)} kernel files to process")
        print("=" * 80)
        print("Static Analysis Mode - No LLM, Pure Numerical Calculation")
        print("=" * 80)
        
        for i, file_path in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}] ", end="")
            result = self.process_file(file_path)
            if result:
                self.results.append(result)
        
        # Save to CSV
        self.save_to_csv(output_csv)
        print(f"\n{'=' * 80}")
        print(f"Static analysis complete! Results saved to: {output_csv}")
        print(f"Total files processed: {len(self.results)}")
        print(f"Files with errors: {sum(1 for r in self.results if 'error' in r)}")
    
    def save_to_csv(self, output_csv: str):
        """Save results to CSV file"""
        if not self.results:
            print("No results to save")
            return
        
        # Define column order
        columns = [
            'filename',
            'comp', 'ld1', 'st1', 'ld0', 'st0', 'L1', 'L2',
            'execution_time', 'comm_gm', 'comm_sm', 'total_cost', 'units',
            'R', 'P', 'lambda', 'g_GM', 'g_L1', 'g_L2', 'g_SM',
            'error', 'filepath'
        ]
        
        # Write CSV
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
            writer.writeheader()
            for result in self.results:
                # Convert None to empty string for CSV
                csv_row = {k: (v if v is not None else '') for k, v in result.items()}
                writer.writerow(csv_row)
        
        print(f"\nSaved {len(self.results)} results to {output_csv}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Static analysis of CUDA/PTX kernel files - No LLM required"
    )
    parser.add_argument("--data-dir", type=str, default="data_b5_t15",
                         help="Directory containing kernel files")
    parser.add_argument("--output", type=str, default="static_analysis_results.csv",
                         help="Output CSV file path")
    parser.add_argument("--limit", type=int, default=None,
                         help="Limit number of files to process (for testing)")
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = StaticAnalyzer(data_dir=args.data_dir)
    
    # Process files
    if args.limit:
        files = analyzer.get_kernel_files()[:args.limit]
        print(f"Processing first {len(files)} files (limited)")
        for file_path in files:
            result = analyzer.process_file(file_path)
            if result:
                analyzer.results.append(result)
        analyzer.save_to_csv(args.output)
    else:
        analyzer.process_all(args.output)


if __name__ == "__main__":
    main()

