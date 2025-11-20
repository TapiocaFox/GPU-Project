# Kernel Execution Time Analysis

This repository contains scripts to analyze and compare kernel execution time predictions with actual measured execution times on TITAN V GPU.

## Overview

The analysis compares three types of execution time predictions:
1. **Static Analysis**: Time predictions from static code analysis
2. **Manual Predictions**: Manually calculated execution times
3. **LLM Predictions**: Execution times predicted by Large Language Models

These predictions are compared against **Actual Measured Times** from real GPU executions.

## Required Files

### Input Files

1. **`static_analysis_results_cuda3_us.csv`**
   - Contains static analysis predictions with execution times in microseconds
   - Columns include: `filename`, `execution_time`, `execution_time_us`, and various kernel metrics

2. **`kernel_results_b5_t15_cuda3 copy.csv`**
   - Contains actual measured execution times from GPU runs
   - Columns include: `filename`, `kernel_name`, `time_us`, `time_ms`, `compilation_success`, `execution_success`

### Optional Files (for full analysis)

3. **`kernel_predictions_cuda3.csv`**
   - Contains manual and LLM predictions (in seconds)
   - Used by `compare_times.py` for comprehensive comparison

4. **`time_comparison_results_us.csv`**
   - Generated output from time comparison analysis
   - Contains all predictions converted to microseconds

## Scripts

### 1. `compare_with_actual_times.py`

**Purpose**: Compares predicted execution times with actual measured times from GPU executions.

**Usage**:
```bash
python3 compare_with_actual_times.py
```

**What it does**:
- Reads actual measured times from `kernel_results_b5_t15_cuda3 copy.csv`
- Compares with static analysis predictions from `static_analysis_results_cuda3_us.csv`
- Compares with manual and LLM predictions from `time_comparison_results_us.csv`
- Calculates differences, relative differences, and percentage errors
- Identifies best and worst predictions

**Output**:
- Console output with summary statistics
- `actual_vs_predicted_comparison.csv` - Detailed comparison results

**Key Metrics**:
- Mean/Median differences (in microseconds)
- Average percentage error
- Median percentage error
- Standard deviation of errors

### 2. `compare_times.py`

**Purpose**: Compares different prediction methods (Manual, LLM, Static Analysis) against each other.

**Usage**:
```bash
python3 compare_times.py
```

**What it does**:
- Compares Manual vs Static Analysis predictions
- Compares LLM vs Static Analysis predictions
- Compares LLM vs Manual predictions
- Generates visualizations (if matplotlib is available)

**Output**:
- Console output with summary statistics
- `time_comparison_results.csv` - Detailed comparison results
- `time_comparison_visualizations.png` - Visualization plots (6 subplots)
- `time_comparison_visualizations_log.png` - Log-scale visualizations (4 subplots)

**Visualizations Include**:
- Scatter plots comparing prediction methods
- Error distribution histograms
- Box plots for error comparison
- Bar charts for top kernels
- Cumulative error distributions

### 3. `convert_to_microseconds.py`

**Purpose**: Converts execution times from seconds to microseconds for TITAN V GPU.

**Usage**:
```bash
python3 convert_to_microseconds.py
```

**What it does**:
- Converts `static_analysis_results_cuda3.csv` → `static_analysis_results_cuda3_us.csv`
- Converts `time_comparison_results.csv` → `time_comparison_results_us.csv`
- Adds `_us` columns with times in microseconds

**Output**:
- `static_analysis_results_cuda3_us.csv`
- `time_comparison_results_us.csv`

## Quick Start Guide

### Step 1: Ensure Required Files Exist

Check that you have:
- `static_analysis_results_cuda3_us.csv` (or run `convert_to_microseconds.py` first)
- `kernel_results_b5_t15_cuda3 copy.csv`

### Step 2: Install Dependencies (Optional, for visualizations)

```bash
pip3 install matplotlib numpy
```

### Step 3: Run the Analysis

**Option A: Compare predictions with actual measured times**
```bash
python3 compare_with_actual_times.py
```

**Option B: Compare different prediction methods**
```bash
python3 compare_times.py
```

**Option C: Run both analyses**
```bash
python3 compare_with_actual_times.py
python3 compare_times.py
```

## Output Files

### Generated CSV Files

1. **`actual_vs_predicted_comparison.csv`**
   - Columns: `filename`, `actual_time_us`, `static_time_us`, `manual_time_us`, `llm_time_us`
   - Error metrics: `diff_*_actual`, `rel_diff_*_actual`, `pct_error_*_actual`
   - One row per kernel with all comparisons

2. **`time_comparison_results.csv`**
   - Columns: `filename`, `manual_time`, `llm_time`, `static_time`
   - Error metrics: `diff_*`, `rel_diff_*`, `pct_error_*`
   - Compares predictions against each other (not actual times)

3. **`time_comparison_results_us.csv`**
   - Same as above but with times in microseconds

### Generated Visualization Files

1. **`time_comparison_visualizations.png`**
   - 6-panel visualization showing:
     - Scatter plots (Manual vs Static, LLM vs Static)
     - Error histograms
     - Box plots
     - Top 20 kernels bar chart

2. **`time_comparison_visualizations_log.png`**
   - 4-panel log-scale visualization showing:
     - Log-log scatter plots
     - Error distribution comparison
     - Cumulative error distribution

## Interpreting Results

### Percentage Error

- **0%**: Perfect prediction
- **< 10%**: Excellent prediction
- **10-50%**: Good prediction
- **50-100%**: Moderate prediction
- **> 100%**: Poor prediction (may indicate systematic issues)

### Key Statistics

- **Mean Difference**: Average absolute difference (in microseconds)
- **Median Difference**: Middle value of differences (less affected by outliers)
- **Average Percentage Error**: Mean of absolute percentage errors
- **Median Percentage Error**: Middle value of percentage errors (more robust to outliers)

### Understanding the Visualizations

1. **Scatter Plots**: Points on the diagonal line indicate perfect predictions
   - Points above the line: Over-prediction
   - Points below the line: Under-prediction

2. **Error Histograms**: Show distribution of prediction errors
   - Narrow distribution: Consistent predictions
   - Wide distribution: Variable prediction quality

3. **Box Plots**: Compare error distributions between methods
   - Lower median: Better method
   - Smaller box: More consistent predictions

4. **Cumulative Distribution**: Shows what percentage of predictions have error below a threshold
   - Steeper curve: More predictions with low error

## Example Workflow

```bash
# 1. Convert times to microseconds (if needed)
python3 convert_to_microseconds.py

# 2. Compare predictions with actual measured times
python3 compare_with_actual_times.py

# 3. Compare different prediction methods
python3 compare_times.py

# 4. Review outputs
# - Check console output for summary statistics
# - Open actual_vs_predicted_comparison.csv for detailed results
# - View time_comparison_visualizations.png for visual analysis
```

## TITAN V GPU Specifications

The analysis is based on TITAN V GPU specifications:
- **Base clock**: 1200 MHz (1.2 GHz)
- **Boost clock**: 1455 MHz (1.455 GHz)
- **Memory**: 12 GB HBM2
- **Memory bandwidth**: ~652.8 GB/s
- **Compute capability**: 7.0
- **SMs**: 80
- **CUDA cores**: 5120

## Troubleshooting

### Missing matplotlib

If you see: `Warning: matplotlib not available. Visualizations will be skipped.`

**Solution**:
```bash
pip3 install matplotlib numpy
```

### File Not Found Errors

Ensure all required CSV files are in the same directory as the scripts:
- `static_analysis_results_cuda3_us.csv`
- `kernel_results_b5_t15_cuda3 copy.csv`
- `kernel_predictions_cuda3.csv` (for full analysis)

### No Matching Files

If you see "Matched 0 kernels for comparison":
- Check that filenames match between files
- Ensure files contain data (not empty)
- Verify CSV file format is correct

## Notes

- Times are converted assuming input values are in seconds
- Scientific notation (e.g., `5.078125e-05`) is handled automatically
- Missing or invalid values are handled gracefully
- Visualizations are optional and require matplotlib

## Contact

For questions or issues with the analysis scripts, please check:
1. All required files are present
2. Python 3.6+ is installed
3. Dependencies are installed (for visualizations)



