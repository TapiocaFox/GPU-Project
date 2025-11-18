1. compile cu file by "nvcc -O3 cuda_microbenchmark_complete.cu -o cuda_microbenchmark_complete"
2. execute "./cuda_microbenchmark_complete" to get ins_microbenchmark.csv


Definition of latency, throughput, and peakwarps for instructions:
1. Latency(cycles per instruction): Time for single instruction execution from start to completion in a dependent chain.
2. Throughput(operations per cycle): Maximum instruction completion rate when many parallel threads execute simultaneously across warps.
3. PeakWarps(minimum warps): Fewest warps needed to achieve 90% of maximum throughput; indicates required parallelism level.

In essence:
* Latency -> How *fast* one instruction executes
* Throughput -> How *many* instructions execute simultaneously
* PeakWarps -> How much *parallelism* is needed for peak performance
