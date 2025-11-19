How to let this static analysis model to predict execution time of a kernel?

Instructions:

**Before executing anything!**

Becauase we're going to execute python, please make sure you have installed the packages required.

Instructions:

1. create venv: execute "python3 -m venv static_analysis_model"
2. activate the environment (Linux/macOS): execute "source static_analysis_model/bin/activate"
3. install requirements: execute "pip install -r requirements.txt"

**To get static analysis model predict kernel execution time**

Output File: 

1. multi_gpu_simulation_results.csv
2. eval_result.txt

Instructions:

1. execute "python3 multi_gpu_batch_simulator.py --repo-root ../" to get multi_gpu_simulation_results.csv
2. execute "python3 eval.py > eval_result.txt"


**After getting all you want**

Instructions:
1. execute "deactivate" to deactivate this python venv

