How to do micro benchmarking?

Instructions:

**Before executing anything!**

Becauase we're going to execute python, please make sure you have installed the packages required.

Instructions:

1. create venv: execute "python3 -m venv microbenchmarking"
2. activate the environment (Linux/macOS): execute "source microbenchmarking/bin/activate"
3. install requirements: execute "pip install -r requirements.txt"

**To get <Latency, Throughput, PeakWarps>**

Output File: 

[For each cuda cluster]
1. ins_microbenchmark.csv

Instructions:

1. compile cu file by "nvcc -O3 cuda_microbenchmark_complete.cu -o cuda_microbenchmark_complete"
2. execute "./cuda_microbenchmark_complete" to get ins_microbenchmark.csv
3. rename the generated ins_microbenchmark.csv to ins_microbenchmark_cuda_{i}.csv
4. ssh to different cudas and do step 1~3 again.

**To get Launch Overhead**

Output File:

[For each cuda cluster]
1. launch_overhead_results.csv
2. overhead_formula_analysis.csv
3. launch_overhead_analysis.txt

Instructions:

1. compile cu file by "nvcc -O3 launch_overhead_test.cu -o launch_overhead_test"
2. execute "./launch_overhead_test" to two files: launch_overhead_results.csv, overhead_formula_analysis.csv
3. execute "python3 analyze_launch_overhead.py > launch_overhead_analysis.txt" to extract final fitting launch overhead formula for this cuda cluster.
4. ssh to different cuda and do step 1~3 again.

**After getting all you want**

Instructions:
1. execute "deactivate" to deactivate this python venv


Definition of latency, throughput, and peakwarps for instructions:
* Latency(cycles per instruction): Time for single instruction execution from start to completion in a dependent chain.
* Throughput(operations per cycle): Maximum instruction completion rate when many parallel threads execute simultaneously across warps.
* PeakWarps(minimum warps): Fewest warps needed to achieve 90% of maximum throughput; indicates required parallelism level.

In essence:
* Latency -> How **fast** one instruction executes
* Throughput -> How **many** instructions execute simultaneously
* PeakWarps -> How much **parallelism** is needed for peak performance
