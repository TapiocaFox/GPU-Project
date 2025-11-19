#!/usr/bin/env python3
"""
Launch Overhead Analysis Script
===============================

Analyzes the results from launch_overhead_test.cu to derive
a new launch overhead formula for RTX 2080 Ti.

Usage:
    python analyze_launch_overhead.py
    
Expected input files:
    - launch_overhead_results.csv
    - overhead_formula_analysis.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import seaborn as sns

class LaunchOverheadAnalyzer:
    def __init__(self):
        self.comprehensive_data = None
        self.formula_data = None
    
    def load_data(self):
        """Load experimental data from CSV files"""
        try:
            print("Loading experimental data...")
            self.comprehensive_data = pd.read_csv('launch_overhead_results_cuda5.csv')
            self.formula_data = pd.read_csv('overhead_formula_analysis_cuda5.csv')
            
            print(f"Loaded {len(self.comprehensive_data)} comprehensive measurements")
            print(f"Loaded {len(self.formula_data)} formula analysis points")
            return True
            
        except FileNotFoundError as e:
            print(f"Error: Could not find data files. Please run launch_overhead_test.cu first.")
            print(f"Missing file: {e.filename}")
            return False
    
    def analyze_comprehensive_results(self):
        """Analyze comprehensive experimental results"""
        print("\n" + "="*60)
        print("COMPREHENSIVE RESULTS ANALYSIS")
        print("="*60)
        
        data = self.comprehensive_data
        
        # Summary statistics
        print(f"\nLaunch Overhead Statistics (from empty kernels):")
        print(f"  Mean: {data['launch_overhead_us'].mean():.3f} μs")
        print(f"  Median: {data['launch_overhead_us'].median():.3f} μs")
        print(f"  Min: {data['launch_overhead_us'].min():.3f} μs")
        print(f"  Max: {data['launch_overhead_us'].max():.3f} μs")
        print(f"  Std Dev: {data['launch_overhead_us'].std():.3f} μs")
        
        # Compare kernel types
        print(f"\nKernel Type Comparison:")
        print(f"  Empty kernel: {data['empty_kernel_us'].mean():.3f} ± {data['empty_kernel_us'].std():.3f} μs")
        print(f"  Single op: {data['single_op_us'].mean():.3f} ± {data['single_op_us'].std():.3f} μs")
        print(f"  Minimal: {data['minimal_us'].mean():.3f} ± {data['minimal_us'].std():.3f} μs")
        
        # Correlation with thread count
        correlation = data['total_threads'].corr(data['launch_overhead_us'])
        print(f"\nCorrelation with thread count: {correlation:.4f}")
        
        if abs(correlation) < 0.1:
            print("  → Launch overhead appears mostly independent of thread count!")
        elif abs(correlation) < 0.5:
            print("  → Weak correlation with thread count")
        else:
            print("  → Significant correlation with thread count")
    
    def derive_new_formula(self):
        """Derive new launch overhead formula"""
        print("\n" + "="*60)
        print("NEW FORMULA DERIVATION")
        print("="*60)
        
        data = self.formula_data
        threads = data['total_threads'].values
        measured_us = data['measured_overhead_us'].values
        
        # Try different formula types
        formulas = {
            'constant': lambda x, a: np.full_like(x, a),
            'linear': lambda x, a, b: a * x + b,
            'sqrt': lambda x, a, b: a * np.sqrt(x) + b,
            'log': lambda x, a, b: a * np.log(x) + b,
            'power': lambda x, a, b, c: a * np.power(x, b) + c
        }
        
        results = {}
        
        for formula_name, formula_func in formulas.items():
            try:
                if formula_name == 'constant':
                    popt = [np.mean(measured_us)]
                    predicted = formula_func(threads, *popt)
                else:
                    popt, _ = curve_fit(formula_func, threads, measured_us, maxfev=10000)
                    predicted = formula_func(threads, *popt)
                
                r2 = r2_score(measured_us, predicted)
                rmse = np.sqrt(np.mean((measured_us - predicted)**2))
                
                results[formula_name] = {
                    'params': popt,
                    'r2': r2,
                    'rmse': rmse,
                    'predicted': predicted
                }
                
                print(f"\n{formula_name.upper()} Formula:")
                if formula_name == 'constant':
                    print(f"  overhead = {popt[0]:.3f} μs")
                elif formula_name == 'linear':
                    print(f"  overhead = {popt[0]:.6e} * threads + {popt[1]:.3f} μs")
                elif formula_name == 'sqrt':
                    print(f"  overhead = {popt[0]:.6f} * sqrt(threads) + {popt[1]:.3f} μs")
                elif formula_name == 'log':
                    print(f"  overhead = {popt[0]:.6f} * log(threads) + {popt[1]:.3f} μs")
                elif formula_name == 'power':
                    print(f"  overhead = {popt[0]:.6f} * threads^{popt[1]:.3f} + {popt[2]:.3f} μs")
                
                print(f"  R²: {r2:.6f}")
                print(f"  RMSE: {rmse:.3f} μs")
                
            except Exception as e:
                print(f"Failed to fit {formula_name}: {e}")
        
        # Find best formula
        best_formula = max(results.keys(), key=lambda k: results[k]['r2'])
        print(f"\n🏆 BEST FORMULA: {best_formula.upper()}")
        print(f"   R² = {results[best_formula]['r2']:.6f}")
        
        return results, best_formula
    
    def compare_with_old_formula(self):
        """Compare new measurements with old Tesla K20 formula"""
        print("\n" + "="*60)
        print("COMPARISON WITH OLD FORMULA")
        print("="*60)
        
        data = self.formula_data
        
        # Calculate improvement
        old_rmse = np.sqrt(np.mean((data['measured_overhead_us'] - data['old_formula_us'])**2))
        mean_measured = data['measured_overhead_us'].mean()
        mean_old_prediction = data['old_formula_us'].mean()
        
        print(f"RTX 2080 Ti measured overhead: {mean_measured:.3f} μs")
        print(f"Tesla K20 formula prediction: {mean_old_prediction:.3f} μs")
        print(f"Ratio (new/old): {mean_measured/mean_old_prediction:.3f}")
        print(f"Improvement: {mean_old_prediction - mean_measured:.1f} μs faster ({((mean_old_prediction - mean_measured)/mean_old_prediction)*100:.1f}% reduction)")
        
        # For your specific configuration (5625 threads)
        threads_5625 = 5625
        old_overhead_5625 = (1.260e-08 * threads_5625 + 4.260e-02) * 1e6  # Convert to μs
        
        # Find closest measurement
        closest_idx = np.argmin(np.abs(data['total_threads'] - threads_5625))
        measured_5625 = data.iloc[closest_idx]['measured_overhead_us']
        actual_threads = data.iloc[closest_idx]['total_threads']
        
        print(f"\nFor your specific configuration (~{threads_5625} threads):")
        print(f"  Closest measurement: {actual_threads} threads")
        print(f"  Measured overhead: {measured_5625:.1f} μs ({measured_5625/1000:.3f} ms)")
        print(f"  Old formula: {old_overhead_5625:.1f} μs ({old_overhead_5625/1000:.3f} ms)")
        print(f"  Difference: {old_overhead_5625 - measured_5625:.1f} μs ({(old_overhead_5625 - measured_5625)/1000:.3f} ms faster!)")
    
    def generate_plots(self):
        """Generate visualization plots"""
        print("\nGenerating plots...")
        
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('RTX 2080 Ti Kernel Launch Overhead Analysis', fontsize=16)
        
        # Plot 1: Overhead vs Thread Count
        ax1 = axes[0, 0]
        data = self.comprehensive_data
        ax1.scatter(data['total_threads'], data['launch_overhead_us'], alpha=0.6, s=30)
        ax1.set_xlabel('Total Threads')
        ax1.set_ylabel('Launch Overhead (μs)')
        ax1.set_title('Launch Overhead vs Thread Count')
        ax1.set_xscale('log')
        
        # Plot 2: Kernel Type Comparison
        ax2 = axes[0, 1]
        kernel_types = ['Empty', 'Single Op', 'Minimal']
        means = [data['empty_kernel_us'].mean(), data['single_op_us'].mean(), data['minimal_us'].mean()]
        stds = [data['empty_kernel_us'].std(), data['single_op_us'].std(), data['minimal_us'].std()]
        
        bars = ax2.bar(kernel_types, means, yerr=stds, capsize=5, alpha=0.7)
        ax2.set_ylabel('Execution Time (μs)')
        ax2.set_title('Kernel Type Comparison')
        
        # Plot 3: Old vs New Formula
        ax3 = axes[1, 0]
        formula_data = self.formula_data
        ax3.plot(formula_data['total_threads'], formula_data['old_formula_us'], 'r-', label='Tesla K20 Formula', linewidth=2)
        ax3.scatter(formula_data['total_threads'], formula_data['measured_overhead_us'], 
                   color='blue', label='RTX 2080 Ti Measured', alpha=0.7, s=50)
        ax3.set_xlabel('Total Threads')
        ax3.set_ylabel('Launch Overhead (μs)')
        ax3.set_title('Old Formula vs Measured')
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.legend()
        
        # Plot 4: Distribution of overhead measurements
        ax4 = axes[1, 1]
        ax4.hist(data['launch_overhead_us'], bins=20, alpha=0.7, edgecolor='black')
        ax4.axvline(data['launch_overhead_us'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {data["launch_overhead_us"].mean():.1f} μs')
        ax4.set_xlabel('Launch Overhead (μs)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Launch Overhead')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig('launch_overhead_analysis.png', dpi=300, bbox_inches='tight')
        print("Plot saved: launch_overhead_analysis.png")
    
    def generate_updated_simulator_code(self, best_formula, results):
        """Generate updated code for the PTX simulator"""
        print("\n" + "="*60)
        print("UPDATED SIMULATOR CODE")
        print("="*60)
        
        params = results[best_formula]['params']
        
        if best_formula == 'constant':
            overhead_code = f"return {params[0]:.3f}e-6  # {params[0]:.1f} μs constant overhead"
        elif best_formula == 'linear':
            overhead_code = f"return {params[0]:.6e} * nt + {params[1]:.3f}e-6  # Linear formula"
        else:
            # For more complex formulas, provide the constant approximation as fallback
            mean_overhead = np.mean(results[best_formula]['predicted'])
            overhead_code = f"return {mean_overhead:.3f}e-6  # ~{mean_overhead:.1f} μs constant (RTX 2080 Ti)"
        
        print("\nTo update your enhanced PTX simulator, replace the launch overhead calculation:")
        print("\nOLD CODE (Tesla K20):")
        print("    def _calculate_launch_overhead(self, launch_params):")
        print("        nt = launch_params.total_threads") 
        print("        return 1.260e-08 * nt + 4.260e-02")
        print("\nNEW CODE (RTX 2080 Ti):")
        print("    def _calculate_launch_overhead(self, launch_params):")
        print("        nt = launch_params.total_threads")
        print(f"        {overhead_code}")
        
        # Calculate the improvement for 5625 threads
        if best_formula == 'constant':
            new_overhead_s = params[0] * 1e-6
        elif best_formula == 'linear':
            new_overhead_s = params[0] * 5625 + params[1] * 1e-6
        else:
            new_overhead_s = np.mean(results[best_formula]['predicted']) * 1e-6
            
        old_overhead_s = 1.260e-08 * 5625 + 4.260e-02
        
        print(f"\nFor your configuration (5625 threads):")
        print(f"  Old formula: {old_overhead_s*1000:.3f} ms")
        print(f"  New formula: {new_overhead_s*1000:.6f} ms")
        print(f"  Speedup: {old_overhead_s/new_overhead_s:.0f}x faster!")

def main():
    print("RTX 2080 Ti Launch Overhead Analysis")
    print("====================================")
    
    analyzer = LaunchOverheadAnalyzer()
    
    if not analyzer.load_data():
        print("\nPlease run the launch overhead experiment first:")
        print("  nvcc -o launch_overhead_test launch_overhead_test.cu")
        print("  ./launch_overhead_test")
        return
    
    # Run analysis
    analyzer.analyze_comprehensive_results()
    results, best_formula = analyzer.derive_new_formula()
    analyzer.compare_with_old_formula()
    analyzer.generate_plots()
    analyzer.generate_updated_simulator_code(best_formula, results)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("Generated files:")
    print("  - launch_overhead_analysis.png (visualization)")
    print("Use the updated formula in your PTX simulator for much more accurate predictions!")

if __name__ == "__main__":
    main()
