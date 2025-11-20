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
