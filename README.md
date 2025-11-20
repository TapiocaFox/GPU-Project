For Three Experiment

1. Static Analysis Model - using microbenchmark:
   
   1. Please make sure you clone the folder kernels_src, which contains kernel source code and PTX code. It will be used in the static analysis model.
   2. First do microbenchmark on different GPUs. Please follow readme.md in src/microbenchmarking/readme.md
   3. Static Analysis Model Time Prediction. Please follow readme.md in src/static_analysis_model/readme.md

3. LLM Zero Shot:

   Template generated in Section 1 is in src/LLM/Data/ChatGPT/cuda_kernel_prediction_template.txt
   
   To reproduce zero-shot results, use the notebooks in src/LLM_baseline/. Set your OpenAI API key (via environment variable OPENAI_API_KEY or Colab secrets), install dependencies (openai, pandas, matplotlib, seaborn), configure the kernel directory path, and run the notebook cells in order. Results are saved to JSON files (kernel_analysis_gpu_spec_baseline.json and kernel_analysis_no_gpu_spec_baseline.json). 
   
   See src/LLM_baseline/README.md for details.

   To reproduce the accuracy evalution, just simply run accuracy_evaluation.py.
   
   After running the script, you'll get:
   
   - **`accuracy_evaluation_results.json`**: Contains all individual kernel comparisons plus overall and dataset-specific metrics
   - **`accuracy_evaluation_summary.json`**: A condensed version with just the summary metrics

   
4. LLM Few Shot: The data and code is in few shot directory, data come from intermediate result in few_shot_pre, which is execute and compiled on 5 NYU CUDA to get the result.
