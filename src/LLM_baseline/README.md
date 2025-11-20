# GPU Kernel Time Predictor Notebooks

Two notebooks that use GPT-4 to predict how long CUDA kernels will take to run. The main difference is whether GPU hardware specs get included in the analysis or not.

## The Two Versions

**gpu_kernel_time_predictor.ipynb** - This one includes GPU specs when analyzing kernels. Things like memory bandwidth, SM count, compute capability, etc. The idea is that giving GPT-4 more hardware context should help it make better predictions.

**gpu_kernel_time_predictor_no_gpu_spec.ipynb** - This version doesn't include any GPU specs. It just sends the kernel code and launch configuration to GPT-4. Made this to test whether the extra hardware info actually matters or if GPT-4 can figure things out from the code alone.

Otherwise they work the same way - scan for `.cu` files, analyze them, predict execution times, identify bottlenecks, suggest optimizations.

## Getting Started

You'll need an OpenAI API key. The notebooks will look for it in a few places:
- Google Colab secrets (if you're running in Colab)
- Environment variable `OPENAI_API_KEY`
- Or it'll just ask you to type it in

Install the packages in the first cell. It needs openai, pandas, matplotlib, and seaborn.

Set your kernel directory path. The default is `/content/drive/MyDrive/kernels_src` for Colab, or `/Users/james/GPU-Project/src/kernels_src` locally. You can change it with the `KERNELS_DIR` environment variable.

## How It Works

Run the cells in order. The notebook will:
1. Load your API key and set up the OpenAI client
2. Define GPU specs (if using the GPU spec version)
3. Scan your kernel directory for `.cu` files
4. For each kernel, send it to GPT-4 along with GPU specs (if applicable) and launch configuration
5. Parse the response to extract execution time predictions, bottlenecks, optimization suggestions
6. Save everything to a JSON file

You can configure the batch analysis parameters before running it:
- Grid and block dimensions
- Data size (optional)
- How many kernels to process (useful for testing)

## Output

Results get saved as JSON files with timestamps like `all_kernels_analysis_20251116_045342.json`. Each file contains analysis results for all the kernels processed in that batch.

Current prediction results for zero-shot prompting with and without GPU specs are output in the following json files:
1. kernel_analysis_gpu_spec_baseline.json
2. kernel_analysis_no_gpu_spec_baseline.json

The notebooks also have some visualization functions if you want to see charts comparing different kernels or GPU architectures.

## Notes

These are predictions, not actual measurements. Real execution times will be different. The notebooks are useful for getting quick estimates before actually running kernels, or for comparing different kernel implementations.

There's also a PTX version (`ptx_kernel_time_predictor.ipynb`) if you want to analyze assembly code instead of CUDA source code.

# GPU Kernel Time Predictor Notebooks

Two notebooks that use GPT-4 to predict how long CUDA kernels will take to run. The main difference is whether GPU hardware specs get included in the analysis or not.

## The Two Versions

**gpu_kernel_time_predictor.ipynb** - This one includes GPU specs when analyzing kernels. Things like memory bandwidth, SM count, compute capability, etc. The idea is that giving GPT-4 more hardware context should help it make better predictions.

**gpu_kernel_time_predictor_no_gpu_spec.ipynb** - This version doesn't include any GPU specs. It just sends the kernel code and launch configuration to GPT-4. Made this to test whether the extra hardware info actually matters or if GPT-4 can figure things out from the code alone.

Otherwise they work the same way - scan for `.cu` files, analyze them, predict execution times, identify bottlenecks, suggest optimizations.

## Getting Started

You'll need an OpenAI API key. The notebooks will look for it in a few places:
- Google Colab secrets (if you're running in Colab)
- Environment variable `OPENAI_API_KEY`
- Or it'll just ask you to type it in

Install the packages in the first cell. It needs openai, pandas, matplotlib, and seaborn.

Set your kernel directory path. The default is `/content/drive/MyDrive/kernels_src` for Colab, or `/Users/james/GPU-Project/src/kernels_src` locally. You can change it with the `KERNELS_DIR` environment variable.

## How It Works

Run the cells in order. The notebook will:
1. Load your API key and set up the OpenAI client
2. Define GPU specs (if using the GPU spec version)
3. Scan your kernel directory for `.cu` files
4. For each kernel, send it to GPT-4 along with GPU specs (if applicable) and launch configuration
5. Parse the response to extract execution time predictions, bottlenecks, optimization suggestions
6. Save everything to a JSON file

You can configure the batch analysis parameters before running it:
- Grid and block dimensions
- Data size (optional)
- How many kernels to process (useful for testing)

## Output

Results get saved as JSON files with timestamps like `all_kernels_analysis_20251116_045342.json`. Each file contains analysis results for all the kernels processed in that batch.

Current prediction results for zero-shot prompting with and without GPU specs are output in the following json files:
1. kernel_analysis_gpu_spec_baseline.json
2. kernel_analysis_no_gpu_spec_baseline.json

The notebooks also have some visualization functions if you want to see charts comparing different kernels or GPU architectures.

## Notes

These are predictions, not actual measurements. Real execution times will be different. The notebooks are useful for getting quick estimates before actually running kernels, or for comparing different kernel implementations.

There's also a PTX version (`ptx_kernel_time_predictor.ipynb`) if you want to analyze assembly code instead of CUDA source code.

## Accuracy Evaluation

We have an accuracy evaluation script (`accuracy_evaluation.py`) that compares LLM predictions against actual execution times. Here's how to reproduce the evaluation results:

### Prerequisites

Before running the evaluation, make sure you have:

1. **LLM baseline predictions**: The script expects `kernel_analysis_no_gpu_spec_baseline.json` in the `LLM_baseline` folder. This file contains predictions generated by running the notebook on all kernels.

2. **Actual execution results**: Execution results should be in the `kernel_executions` folder (one level up from `LLM_baseline`). The script looks for subdirectories like `test_cuda2/`, `validation_cuda2/`, etc., each containing an `execution_results.json` file with actual execution times.

3. **Python dependencies**: Make sure you have `numpy` installed (`pip install numpy`).

### Running the Evaluation

1. **Navigate to the LLM_baseline directory**:
   ```bash
   cd src/LLM_baseline
   ```

2. **Run the evaluation script**:
   ```bash
   python3 accuracy_evaluation.py
   ```

   The script will:
   - Load LLM predictions from `kernel_analysis_no_gpu_spec_baseline.json`
   - Load actual execution times from all `execution_results.json` files in `../kernel_executions/`
   - Match predictions with actual times by kernel ID (dataset, folder_id, kernel_id)
   - Calculate accuracy metrics (MAE, RMSE, MAPE, R², correlation, etc.)
   - Print a summary to the console
   - Save detailed results to `accuracy_evaluation_results.json`
   - Save summary metrics to `accuracy_evaluation_summary.json`

### What the Script Does

The evaluation process involves several steps:

1. **Loading LLM predictions**: Parses the baseline JSON file and extracts execution time estimates. If a kernel has multiple source files (e.g., `main.cu` and `kernel.cu`), it averages the predictions.

2. **Loading actual execution times**: Scans all execution result folders and extracts successful execution times. The script matches kernels using the tuple `(dataset, folder_id, kernel_id)`.

3. **Unit conversion**: Converts LLM predictions from microseconds to seconds for comparison.

4. **Error calculation**: For each matched kernel, computes:
   - Absolute error (in seconds)
   - Relative error (percentage)
   - Various error metrics across all kernels

5. **Metric computation**: Calculates overall metrics like:
   - Mean Absolute Error (MAE)
   - Root Mean Squared Error (RMSE)
   - Mean/Median Absolute Percentage Error (MAPE/MdAPE)
   - R² and correlation coefficient
   - Error percentiles and prediction accuracy thresholds

6. **Dataset-specific analysis**: Computes separate metrics for `test` and `validation` datasets.

### Output Files

After running the script, you'll get:

- **`accuracy_evaluation_results.json`**: Contains all individual kernel comparisons plus overall and dataset-specific metrics
- **`accuracy_evaluation_summary.json`**: A condensed version with just the summary metrics

### Interpreting Results

The console output shows:
- Number of kernels compared
- Overall error metrics (MAE, RMSE, MAPE, MdAPE)
- Correlation metrics (R², correlation coefficient)
- Percentage of predictions within error thresholds (10%, 25%, 50%, 2x)
- Error percentiles (p25, p50, p75, p90, p95, p99)
- Separate metrics for test and validation datasets

A typical run might show something like: "Found 47 matching kernels for comparison" followed by detailed accuracy metrics. The script handles cases where some kernels don't match (maybe they weren't in the LLM baseline or execution failed) gracefully.

### Notes

- The script only compares kernels that have both LLM predictions AND successful actual executions
- If a kernel was run multiple times, only the first successful execution is used
- If a kernel has multiple source files in the LLM baseline, predictions are averaged
- Make sure the file paths in the script match your directory structure if you've reorganized things

