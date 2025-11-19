#!/usr/bin/env python3
"""
Workflow script to process all CUDA/PTX files, extract parameters, and predict execution times
Outputs results to CSV file
"""

import os
import sys
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional
import argparse
from datetime import datetime

# Import our modules
from predict_cuda_kernel_time import KernelParameters, LLMPredictor, calculate_manual_prediction
from ptx_parser import PTXParser


class KernelWorkflow:
    """Workflow for processing multiple kernel files"""
    
    def __init__(self, 
                 data_dir: str = "data_b5_t15",
                 prompt_file: str = "gpu_analytical_model_prompt.md",
                 api_key: Optional[str] = None,
                 model: str = "gpt-4",
                 api_provider: str = "openai",
                 use_llm_extraction: bool = False):
        """
        Initialize workflow
        
        Args:
            data_dir: Directory containing CUDA/PTX files
            prompt_file: Path to prompt file
            api_key: API key for LLM
            model: LLM model name
            api_provider: "openai" or "anthropic"
            use_llm_extraction: Use LLM to extract parameters (more accurate but slower)
        """
        self.data_dir = Path(data_dir)
        self.parser = PTXParser()
        self.predictor = LLMPredictor(
            prompt_file=prompt_file,
            api_key=api_key,
            model=model
        )
        self.api_provider = api_provider
        self.use_llm_extraction = use_llm_extraction
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
        Process a single kernel file
        
        Args:
            file_path: Path to kernel file
        
        Returns:
            Dictionary with results or None if failed
        """
        print(f"Processing: {file_path.name}")
        
        try:
            # Extract parameters
            if self.use_llm_extraction:
                print(f"  Using LLM to extract parameters...")
                extracted = self.parser.extract_with_llm(str(file_path), self.predictor)
            else:
                print(f"  Using regex parsing to extract parameters...")
                extracted = self.parser.parse_ptx_file(str(file_path))
            
            print(f"  Extracted: comp={extracted['comp']:.1f}, "
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
            
            # Get manual prediction first (always works)
            manual_result = calculate_manual_prediction(params)
            
            # Try LLM prediction
            llm_result = None
            llm_error = None
            try:
                print(f"  Calling LLM for prediction...")
                llm_result = self.predictor.predict(params, api_provider=self.api_provider)
                
                # Check if we got a valid result
                if 'error' in llm_result and 'raw_response' in llm_result:
                    print(f"  Warning: Could not parse JSON, got raw response")
                    print(f"  Raw response preview: {llm_result.get('raw_response', '')[:200]}...")
                elif 'execution_time' in llm_result:
                    print(f"  LLM prediction: {llm_result.get('execution_time')} {llm_result.get('units', '')}")
                else:
                    print(f"  LLM prediction: {llm_result.get('execution_time', 'N/A')} (check raw_response for details)")
            except Exception as e:
                llm_error = str(e)
                print(f"  LLM prediction failed: {e}")
            
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
                # Manual calculation results
                'manual_comm_gm': manual_result['comm_gm'],
                'manual_comm_sm': manual_result['comm_sm'],
                'manual_total_cost': manual_result['total_cost'],
                'manual_execution_time': manual_result['execution_time'],
                # LLM results (if available)
                'llm_execution_time': llm_result.get('execution_time') if llm_result else None,
                'llm_comm_gm': llm_result.get('comm_gm') if llm_result else None,
                'llm_comm_sm': llm_result.get('comm_sm') if llm_result else None,
                'llm_total_cost': llm_result.get('total_cost') if llm_result else None,
                'llm_units': llm_result.get('units') if llm_result else None,
                'llm_error': llm_error,
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
            print(f"  ERROR processing {file_path.name}: {e}")
            return {
                'filename': file_path.name,
                'filepath': str(file_path),
                'error': str(e)
            }
    
    def process_all(self, output_csv: str = "kernel_predictions.csv"):
        """
        Process all kernel files and save to CSV
        
        Args:
            output_csv: Output CSV file path
        """
        files = self.get_kernel_files()
        print(f"Found {len(files)} kernel files to process")
        print("=" * 80)
        
        for i, file_path in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}] Processing: {file_path.name}")
            result = self.process_file(file_path)
            if result:
                self.results.append(result)
        
        # Save to CSV
        self.save_to_csv(output_csv)
        print(f"\n{'=' * 80}")
        print(f"Processing complete! Results saved to: {output_csv}")
        print(f"Total files processed: {len(self.results)}")
    
    def save_to_csv(self, output_csv: str):
        """Save results to CSV file"""
        if not self.results:
            print("No results to save")
            return
        
        # Get all unique keys from all results
        all_keys = set()
        for result in self.results:
            all_keys.update(result.keys())
        
        # Define column order
        columns = [
            'filename',
            'comp', 'ld1', 'st1', 'ld0', 'st0', 'L1', 'L2',
            'manual_execution_time', 'manual_comm_gm', 'manual_comm_sm', 'manual_total_cost',
            'llm_execution_time', 'llm_comm_gm', 'llm_comm_sm', 'llm_total_cost', 'llm_units',
            'R', 'P', 'lambda', 'g_GM', 'g_L1', 'g_L2', 'g_SM',
            'llm_error', 'error'
        ]
        
        # Add any missing columns
        for key in sorted(all_keys):
            if key not in columns:
                columns.append(key)
        
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
        description="Process CUDA/PTX kernel files and predict execution times"
    )
    parser.add_argument("--data-dir", type=str, default="data_b5_t15",
                         help="Directory containing kernel files")
    parser.add_argument("--output", type=str, default="kernel_predictions.csv",
                         help="Output CSV file path")
    parser.add_argument("--prompt-file", type=str, default="gpu_analytical_model_prompt.md",
                         help="Path to prompt file")
    parser.add_argument("--api-key", type=str, default=None,
                         help="API key (or set OPENAI_API_KEY/ANTHROPIC_API_KEY env var)")
    parser.add_argument("--api-provider", type=str, default="openai",
                         choices=["openai", "anthropic"],
                         help="LLM API provider")
    parser.add_argument("--model", type=str, default="gpt-4",
                         help="LLM model name")
    parser.add_argument("--use-llm-extraction", action="store_true",
                         help="Use LLM to extract parameters (more accurate but slower)")
    parser.add_argument("--limit", type=int, default=None,
                         help="Limit number of files to process (for testing)")
    
    args = parser.parse_args()
    
    # Create workflow
    workflow = KernelWorkflow(
        data_dir=args.data_dir,
        prompt_file=args.prompt_file,
        api_key=args.api_key,
        model=args.model,
        api_provider=args.api_provider,
        use_llm_extraction=args.use_llm_extraction
    )
    
    # Process files
    if args.limit:
        files = workflow.get_kernel_files()[:args.limit]
        print(f"Processing first {len(files)} files (limited)")
        for file_path in files:
            result = workflow.process_file(file_path)
            if result:
                workflow.results.append(result)
        workflow.save_to_csv(args.output)
    else:
        workflow.process_all(args.output)


if __name__ == "__main__":
    main()

