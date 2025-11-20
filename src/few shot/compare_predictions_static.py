#!/usr/bin/env python3
"""
Script to compare kernel predictions with static analysis results.
Compares kernel_predictions_cuda3.csv with static_analysis_results_cuda3.csv
Includes visualization of comparisons
"""

import csv
import json
import sys
from statistics import mean, median, stdev

# Try to import matplotlib for visualization
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Visualizations will be skipped.")
    print("Install with: pip install matplotlib numpy")

def safe_float(value):
    """Safely convert value to float, handling None and empty strings"""
    if not value or value == '' or value == 'nan':
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def extract_llm_time(value):
    """Extract numeric time value from llm_execution_time column"""
    if not value or value == '' or value == 'nan':
        return None
    try:
        # Try to parse as JSON/dict string
        if isinstance(value, str) and value.startswith('{'):
            # Replace single quotes with double quotes for JSON parsing
            value_clean = value.replace("'", '"')
            parsed = json.loads(value_clean)
            if 'value' in parsed:
                return float(parsed['value'])
        # If it's already a number string, convert directly
        return float(value)
    except (ValueError, TypeError, json.JSONDecodeError):
        return None

def create_visualizations(matched_data):
    """Create visualization plots for the comparison"""
    if not HAS_MATPLOTLIB:
        return
    
    print("\nGenerating visualizations...")
    
    # Prepare data for plotting
    manual_times = []
    llm_times = []
    static_times = []
    manual_percent_errors = []
    llm_percent_errors = []
    
    for item in matched_data:
        if item.get('static_time') is not None:
            static_times.append(item['static_time'])
            
            if item.get('manual_time') is not None:
                manual_times.append(item['manual_time'])
            else:
                manual_times.append(None)
            
            if item.get('llm_time') is not None:
                llm_times.append(item['llm_time'])
            else:
                llm_times.append(None)
            
            if item.get('manual_percent_error') is not None:
                manual_percent_errors.append(item['manual_percent_error'])
            
            if item.get('llm_percent_error') is not None:
                llm_percent_errors.append(item['llm_percent_error'])
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Scatter plot: Manual vs Static (linear scale)
    ax1 = plt.subplot(3, 3, 1)
    manual_valid = [(m, s) for m, s in zip(manual_times, static_times) if m is not None and s is not None]
    if manual_valid:
        manual_vals, static_vals = zip(*manual_valid)
        ax1.scatter(static_vals, manual_vals, alpha=0.6, s=20)
        # Perfect prediction line
        min_val = min(min(static_vals), min(manual_vals))
        max_val = max(max(static_vals), max(manual_vals))
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
        ax1.set_xlabel('Static Analysis Time')
        ax1.set_ylabel('Manual Prediction Time')
        ax1.set_title('Manual vs Static Analysis (Linear)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. Scatter plot: LLM vs Static (log scale)
    ax2 = plt.subplot(3, 3, 2)
    llm_valid = [(l, s) for l, s in zip(llm_times, static_times) if l is not None and s is not None and l > 0 and s > 0]
    if llm_valid:
        llm_vals, static_vals = zip(*llm_valid)
        ax2.loglog(static_vals, llm_vals, 'o', alpha=0.6, markersize=4)
        # Perfect prediction line
        min_val = min(min(static_vals), min(llm_vals))
        max_val = max(max(static_vals), max(llm_vals))
        ax2.loglog([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
        ax2.set_xlabel('Static Analysis Time (log)')
        ax2.set_ylabel('LLM Prediction Time (log)')
        ax2.set_title('LLM vs Static Analysis (Log-Log)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. Histogram: Manual percentage errors
    ax3 = plt.subplot(3, 3, 3)
    if manual_percent_errors:
        # Cap extreme values for better visualization
        capped_errors = [min(e, 1000) for e in manual_percent_errors]
        ax3.hist(capped_errors, bins=50, edgecolor='black', alpha=0.7)
        ax3.set_xlabel('Percentage Error (%)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Manual Prediction Error Distribution')
        ax3.axvline(mean(manual_percent_errors), color='r', linestyle='--', label=f'Mean: {mean(manual_percent_errors):.2f}%')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Histogram: LLM percentage errors
    ax4 = plt.subplot(3, 3, 4)
    if llm_percent_errors:
        # Cap extreme values for better visualization
        capped_errors = [min(e, 1000) for e in llm_percent_errors]
        ax4.hist(capped_errors, bins=50, edgecolor='black', alpha=0.7, color='orange')
        ax4.set_xlabel('Percentage Error (%)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('LLM Prediction Error Distribution')
        if llm_percent_errors:
            ax4.axvline(mean(llm_percent_errors), color='r', linestyle='--', label=f'Mean: {mean(llm_percent_errors):.2f}%')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # 5. Box plot: Error comparison
    ax5 = plt.subplot(3, 3, 5)
    if manual_percent_errors and llm_percent_errors:
        # Cap extreme values for box plot
        manual_capped = [min(e, 1000) for e in manual_percent_errors]
        llm_capped = [min(e, 1000) for e in llm_percent_errors]
        bp = ax5.boxplot([manual_capped, llm_capped], labels=['Manual', 'LLM'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        ax5.set_ylabel('Percentage Error (%)')
        ax5.set_title('Error Distribution Comparison')
        ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Scatter plot: Manual vs Static (log scale)
    ax6 = plt.subplot(3, 3, 6)
    if manual_valid:
        manual_vals, static_vals = zip(*manual_valid)
        # Filter out zeros for log scale
        log_valid = [(m, s) for m, s in zip(manual_vals, static_vals) if m > 0 and s > 0]
        if log_valid:
            m_vals, s_vals = zip(*log_valid)
            ax6.loglog(s_vals, m_vals, 'o', alpha=0.6, markersize=4)
            min_val = min(min(s_vals), min(m_vals))
            max_val = max(max(s_vals), max(m_vals))
            ax6.loglog([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
            ax6.set_xlabel('Static Analysis Time (log)')
            ax6.set_ylabel('Manual Prediction Time (log)')
            ax6.set_title('Manual vs Static Analysis (Log-Log)')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
    
    # 7. Bar chart: Top 20 kernels by static time (comparing all three)
    ax7 = plt.subplot(3, 3, 7)
    # Sort by static time and take top 20
    sorted_data = sorted(matched_data, 
                       key=lambda x: x.get('static_time') or 0, 
                       reverse=True)[:20]
    filenames = [item['filename'][:30] + '...' if len(item['filename']) > 30 else item['filename'] 
                 for item in sorted_data]
    static_vals = [item.get('static_time') or 0 for item in sorted_data]
    manual_vals = [item.get('manual_time') or 0 for item in sorted_data]
    llm_vals = [item.get('llm_time') or 0 for item in sorted_data]
    
    x = np.arange(len(filenames))
    width = 0.25
    ax7.bar(x - width, static_vals, width, label='Static', alpha=0.8)
    ax7.bar(x, manual_vals, width, label='Manual', alpha=0.8)
    ax7.bar(x + width, llm_vals, width, label='LLM', alpha=0.8)
    ax7.set_xlabel('Kernel (Top 20 by Static Time)')
    ax7.set_ylabel('Execution Time')
    ax7.set_title('Top 20 Kernels Comparison')
    ax7.set_xticks(x)
    ax7.set_xticklabels(filenames, rotation=45, ha='right', fontsize=6)
    ax7.legend()
    ax7.grid(True, alpha=0.3, axis='y')
    
    # 8. Scatter plot: Percentage error vs Static time
    ax8 = plt.subplot(3, 3, 8)
    if llm_percent_errors and static_times:
        # Match percent errors with static times
        llm_error_static = [(e, s) for e, s in zip(llm_percent_errors, static_times) 
                            if e is not None and s is not None and s > 0]
        if llm_error_static:
            errors, times = zip(*llm_error_static)
            # Cap errors for visualization
            capped_errors = [min(e, 1000) for e in errors]
            ax8.scatter(times, capped_errors, alpha=0.6, s=20, color='orange')
            ax8.set_xlabel('Static Analysis Time')
            ax8.set_ylabel('LLM Percentage Error (%)')
            ax8.set_title('LLM Error vs Static Time')
            ax8.set_yscale('log')
            ax8.grid(True, alpha=0.3)
    
    # 9. Cumulative distribution of errors
    ax9 = plt.subplot(3, 3, 9)
    if manual_percent_errors and llm_percent_errors:
        manual_sorted = sorted([min(e, 1000) for e in manual_percent_errors])
        llm_sorted = sorted([min(e, 1000) for e in llm_percent_errors])
        manual_cdf = np.arange(1, len(manual_sorted) + 1) / len(manual_sorted)
        llm_cdf = np.arange(1, len(llm_sorted) + 1) / len(llm_sorted)
        ax9.plot(manual_sorted, manual_cdf, label='Manual', linewidth=2)
        ax9.plot(llm_sorted, llm_cdf, label='LLM', linewidth=2)
        ax9.set_xlabel('Percentage Error (%)')
        ax9.set_ylabel('Cumulative Probability')
        ax9.set_title('Cumulative Error Distribution')
        ax9.legend()
        ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    output_file = "predictions_vs_static_visualizations.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved visualization to: {output_file}")
    plt.close()
    
    # Create a second figure with log-scale visualizations
    fig2 = plt.figure(figsize=(16, 10))
    
    # 1. LLM vs Static (log-log) - larger view
    ax1 = plt.subplot(2, 2, 1)
    if llm_valid:
        llm_vals, static_vals = zip(*llm_valid)
        ax1.loglog(static_vals, llm_vals, 'o', alpha=0.6, markersize=6)
        min_val = min(min(static_vals), min(llm_vals))
        max_val = max(max(static_vals), max(llm_vals))
        ax1.loglog([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
        ax1.set_xlabel('Static Analysis Time (log scale)', fontsize=12)
        ax1.set_ylabel('LLM Prediction Time (log scale)', fontsize=12)
        ax1.set_title('LLM vs Static Analysis (Log-Log Scale)', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
    
    # 2. Error distribution comparison (log scale)
    ax2 = plt.subplot(2, 2, 2)
    if manual_percent_errors and llm_percent_errors:
        manual_capped = [min(e, 10000) for e in manual_percent_errors]
        llm_capped = [min(e, 10000) for e in llm_percent_errors]
        ax2.hist(manual_capped, bins=50, alpha=0.6, label='Manual', edgecolor='black')
        ax2.hist(llm_capped, bins=50, alpha=0.6, label='LLM', edgecolor='black')
        ax2.set_xlabel('Percentage Error (%)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Error Distribution Comparison', fontsize=14, fontweight='bold')
        ax2.set_yscale('log')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Scatter: Absolute difference vs Static time
    ax3 = plt.subplot(2, 2, 3)
    llm_abs_diffs = []
    static_for_diff = []
    for item in matched_data:
        if item.get('llm_abs_diff') is not None and item.get('static_time') is not None:
            llm_abs_diffs.append(item['llm_abs_diff'])
            static_for_diff.append(item['static_time'])
    if llm_abs_diffs:
        ax3.scatter(static_for_diff, llm_abs_diffs, alpha=0.6, s=20, color='orange')
        ax3.set_xlabel('Static Analysis Time', fontsize=12)
        ax3.set_ylabel('Absolute Difference (LLM - Static)', fontsize=12)
        ax3.set_title('Absolute Difference vs Static Time', fontsize=14, fontweight='bold')
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
    
    # 4. Percentage error by kernel type (.cu vs .ptx)
    ax4 = plt.subplot(2, 2, 4)
    cu_errors = []
    ptx_errors = []
    for item in matched_data:
        if item.get('llm_percent_error') is not None:
            if item['filename'].endswith('.cu'):
                cu_errors.append(min(item['llm_percent_error'], 1000))
            elif item['filename'].endswith('.ptx'):
                ptx_errors.append(min(item['llm_percent_error'], 1000))
    if cu_errors and ptx_errors:
        bp = ax4.boxplot([cu_errors, ptx_errors], labels=['.cu files', '.ptx files'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        ax4.set_ylabel('Percentage Error (%)', fontsize=12)
        ax4.set_title('Error by File Type', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save the second figure
    output_file2 = "predictions_vs_static_visualizations_log.png"
    plt.savefig(output_file2, dpi=150, bbox_inches='tight')
    print(f"  Saved log-scale visualization to: {output_file2}")
    plt.close()

# File paths
predictions_file = "kernel_predictions_cuda3.csv"
static_analysis_file = "static_analysis_results_cuda3.csv"

# Read predictions file
print("Reading kernel predictions file...")
predictions = {}
try:
    with open(predictions_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get('filename', '').strip()
            if filename:
                manual_time = safe_float(row.get('manual_execution_time', ''))
                llm_time = extract_llm_time(row.get('llm_execution_time', ''))
                predictions[filename] = {
                    'manual_time': manual_time,
                    'llm_time': llm_time
                }
    print(f"  Loaded {len(predictions)} predictions")
except FileNotFoundError:
    print(f"Error: {predictions_file} not found!")
    sys.exit(1)

# Read static analysis file
print("Reading static analysis file...")
static_analysis = {}
try:
    with open(static_analysis_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get('filename', '').strip()
            if filename:
                execution_time = safe_float(row.get('execution_time', ''))
                static_analysis[filename] = {
                    'execution_time': execution_time
                }
    print(f"  Loaded {len(static_analysis)} static analysis results")
except FileNotFoundError:
    print(f"Error: {static_analysis_file} not found!")
    sys.exit(1)

# Match and compare
print("\nMatching files and calculating differences...")
matched_data = []
manual_diffs = []
llm_diffs = []
manual_rel_diffs = []
llm_rel_diffs = []

for filename in predictions:
    if filename in static_analysis:
        pred = predictions[filename]
        static = static_analysis[filename]
        
        manual_time = pred['manual_time']
        llm_time = pred['llm_time']
        static_time = static['execution_time']
        
        item = {
            'filename': filename,
            'manual_time': manual_time,
            'llm_time': llm_time,
            'static_time': static_time
        }
        
        # Calculate differences for manual predictions
        if manual_time is not None and static_time is not None:
            diff = manual_time - static_time
            item['manual_diff'] = diff
            item['manual_abs_diff'] = abs(diff)
            manual_diffs.append(diff)
            
            # Calculate relative difference (percentage)
            if static_time != 0:
                rel_diff = (diff / static_time) * 100
                item['manual_rel_diff'] = rel_diff
                item['manual_percent_error'] = abs(rel_diff)
                manual_rel_diffs.append(rel_diff)
            else:
                item['manual_rel_diff'] = None
                item['manual_percent_error'] = None
        else:
            item['manual_diff'] = None
            item['manual_abs_diff'] = None
            item['manual_rel_diff'] = None
            item['manual_percent_error'] = None
        
        # Calculate differences for LLM predictions
        if llm_time is not None and static_time is not None:
            diff = llm_time - static_time
            item['llm_diff'] = diff
            item['llm_abs_diff'] = abs(diff)
            llm_diffs.append(diff)
            
            # Calculate relative difference (percentage)
            if static_time != 0:
                rel_diff = (diff / static_time) * 100
                item['llm_rel_diff'] = rel_diff
                item['llm_percent_error'] = abs(rel_diff)
                llm_rel_diffs.append(rel_diff)
            else:
                item['llm_rel_diff'] = None
                item['llm_percent_error'] = None
        else:
            item['llm_diff'] = None
            item['llm_abs_diff'] = None
            item['llm_rel_diff'] = None
            item['llm_percent_error'] = None
        
        matched_data.append(item)
    else:
        print(f"  Warning: {filename} found in predictions but not in static analysis")

print(f"\nMatched {len(matched_data)} files for comparison")

# Calculate statistics
print("\n" + "=" * 80)
print("COMPARISON RESULTS")
print("=" * 80)

# Manual predictions vs Static analysis
print("\n" + "-" * 80)
print("MANUAL PREDICTIONS vs STATIC ANALYSIS")
print("-" * 80)
if manual_diffs:
    print(f"  Number of comparisons: {len(manual_diffs)}")
    print(f"  Average difference (manual - static): {mean(manual_diffs):.6e}")
    print(f"  Median difference: {median(manual_diffs):.6e}")
    if len(manual_diffs) > 1:
        print(f"  Std deviation: {stdev(manual_diffs):.6e}")
    print(f"  Average absolute difference: {mean([abs(d) for d in manual_diffs]):.6e}")
    
    if manual_rel_diffs:
        print(f"\n  PERCENTAGE DIFFERENCES:")
        print(f"    Average percentage difference: {mean(manual_rel_diffs):.2f}%")
        print(f"    Median percentage difference: {median(manual_rel_diffs):.2f}%")
        if len(manual_rel_diffs) > 1:
            print(f"    Std deviation: {stdev(manual_rel_diffs):.2f}%")
        
        # Average percentage error (absolute value)
        percent_errors = [abs(d) for d in manual_rel_diffs]
        print(f"    Average percentage error: {mean(percent_errors):.2f}%")
        print(f"    Median percentage error: {median(percent_errors):.2f}%")
else:
    print("  No manual predictions available for comparison")

# LLM predictions vs Static analysis
print("\n" + "-" * 80)
print("LLM PREDICTIONS vs STATIC ANALYSIS")
print("-" * 80)
if llm_diffs:
    print(f"  Number of comparisons: {len(llm_diffs)}")
    print(f"  Average difference (llm - static): {mean(llm_diffs):.6e}")
    print(f"  Median difference: {median(llm_diffs):.6e}")
    if len(llm_diffs) > 1:
        print(f"  Std deviation: {stdev(llm_diffs):.6e}")
    print(f"  Average absolute difference: {mean([abs(d) for d in llm_diffs]):.6e}")
    
    if llm_rel_diffs:
        print(f"\n  PERCENTAGE DIFFERENCES:")
        print(f"    Average percentage difference: {mean(llm_rel_diffs):.2f}%")
        print(f"    Median percentage difference: {median(llm_rel_diffs):.2f}%")
        if len(llm_rel_diffs) > 1:
            print(f"    Std deviation: {stdev(llm_rel_diffs):.2f}%")
        
        # Average percentage error (absolute value)
        percent_errors = [abs(d) for d in llm_rel_diffs]
        print(f"    Average percentage error: {mean(percent_errors):.2f}%")
        print(f"    Median percentage error: {median(percent_errors):.2f}%")
else:
    print("  No LLM predictions available for comparison")

# Save detailed comparison to CSV
output_file = "predictions_vs_static_comparison.csv"
print(f"\nSaving detailed comparison to: {output_file}")
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    fieldnames = ['filename', 'manual_time', 'llm_time', 'static_time',
                  'manual_diff', 'manual_abs_diff', 'manual_rel_diff', 'manual_percent_error',
                  'llm_diff', 'llm_abs_diff', 'llm_rel_diff', 'llm_percent_error']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for item in matched_data:
        writer.writerow({
            'filename': item['filename'],
            'manual_time': item['manual_time'],
            'llm_time': item['llm_time'],
            'static_time': item['static_time'],
            'manual_diff': item.get('manual_diff'),
            'manual_abs_diff': item.get('manual_abs_diff'),
            'manual_rel_diff': item.get('manual_rel_diff'),
            'manual_percent_error': item.get('manual_percent_error'),
            'llm_diff': item.get('llm_diff'),
            'llm_abs_diff': item.get('llm_abs_diff'),
            'llm_rel_diff': item.get('llm_rel_diff'),
            'llm_percent_error': item.get('llm_percent_error')
        })

print(f"  Saved {len(matched_data)} comparisons")

# Show top differences
if manual_rel_diffs:
    print("\n" + "-" * 80)
    print("TOP 10 LARGEST MANUAL PREDICTION DIFFERENCES (by absolute percentage error)")
    print("-" * 80)
    sorted_manual = sorted(matched_data, 
                          key=lambda x: x.get('manual_percent_error') or 0, 
                          reverse=True)
    for i, item in enumerate(sorted_manual[:10], 1):
        if item.get('manual_percent_error') is not None:
            print(f"  {i}. {item['filename']}")
            print(f"     Manual: {item['manual_time']:.6e}, Static: {item['static_time']:.6e}")
            print(f"     Difference: {item.get('manual_diff', 0):.6e}, Error: {item.get('manual_percent_error', 0):.2f}%")

if llm_rel_diffs:
    print("\n" + "-" * 80)
    print("TOP 10 LARGEST LLM PREDICTION DIFFERENCES (by absolute percentage error)")
    print("-" * 80)
    sorted_llm = sorted(matched_data, 
                       key=lambda x: x.get('llm_percent_error') or 0, 
                       reverse=True)
    for i, item in enumerate(sorted_llm[:10], 1):
        if item.get('llm_percent_error') is not None:
            print(f"  {i}. {item['filename']}")
            print(f"     LLM: {item['llm_time']:.6e}, Static: {item['static_time']:.6e}")
            print(f"     Difference: {item.get('llm_diff', 0):.6e}, Error: {item.get('llm_percent_error', 0):.2f}%")

# Create visualizations
create_visualizations(matched_data)

print("\n" + "=" * 80)
print("COMPARISON COMPLETE")
print("=" * 80)

