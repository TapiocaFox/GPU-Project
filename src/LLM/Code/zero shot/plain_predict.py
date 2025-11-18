#!/usr/bin/env python3

import os
import re
import csv
import time
from pathlib import Path

try:
    import openai
except ImportError:
    print("Error: openai package not installed. Install with: pip install openai")
    exit(1)


# GPU configurations
GPUS = [
    "GeForce GTX TITAN Black (6 GB memory)",
    "GeForce RTX 2080 Ti (11 GB memory)",
    "TITAN V (12 GB memory)",
    "GeForce GTX TITAN X (12 GB memory)",
    "GeForce GTX 4070 (12 GB memory)"
]

GPU_COLUMN_NAMES = [
    "GTX_TITAN_Black",
    "RTX_2080_Ti",
    "TITAN_V",
    "GTX_TITAN_X",
    "GTX_4070"
]


def predict_execution_time(file_path, client, gpu_name):
    """Predict execution time using OpenAI for a specific GPU"""
    # Read CUDA code
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        cuda_code = f.read()
    
    # Build prompt with specific GPU
    prompt = f"""predict CUDA kernel execution time. The kernel is running on {gpu_name}. Just give me the execution time in milliseconds.

CUDA Kernel Code:
```cuda
{cuda_code}
```
"""
    
    try:
        # Basic API call - simplest version
        response = client.chat.completions.create(
            model="gpt-5.1",
            messages=[
                {
                    "role": "system",
                    "content": "You are correct, and safe. Analyze the provided CUDA kernel code and predict its execution time. Provide only the execution time prediction."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            max_completion_tokens=200
        )
        
        # Extract response
        response_text = response.choices[0].message.content.strip()
        
        # Try to extract number from response (handles "5.2 ms", "5.2", etc.)
        numbers = re.findall(r'\d+\.?\d*', response_text)
        if numbers:
            prediction = float(numbers[0])
        else:
            prediction = response_text
        
        return prediction, (prompt, response_text)
        
    except Exception as e:
        print(f"Error: {e}")
        return None, None


def save_to_csv(filename, predictions_dict, csv_file="predictions.csv"):
    """Save predictions to CSV file with GPU columns"""
    file_exists = Path(csv_file).exists()
    
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # Write header if file is new
        if not file_exists:
            header = ['filename'] + GPU_COLUMN_NAMES
            writer.writerow(header)
        
        # Write data row
        row = [filename] + [predictions_dict.get(gpu, '') for gpu in GPU_COLUMN_NAMES]
        writer.writerow(row)


def save_computation_process(filename, gpu_name, prompt, response, txt_file="computation_process.txt"):
    """Save computation process (prompt and response) to text file"""
    with open(txt_file, 'a', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"File: {filename} | GPU: {gpu_name}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("PROMPT:\n")
        f.write("-" * 80 + "\n")
        f.write(prompt)
        f.write("\n\n")
        
        f.write("RESPONSE:\n")
        f.write("-" * 80 + "\n")
        f.write(response)
        f.write("\n\n\n")


def main():
    """Main function - processes all .cu files in data_b5_t15"""
    api_key = #mask
    if not api_key:
        print("Error: API key not set")
        return
    
    client = openai.OpenAI(api_key=api_key)
    
    data_dir = Path("data_b5_t15")
    
    if not data_dir.exists():
        print(f"Error: Directory not found: {data_dir}")
        return
    
    cu_files = sorted(list(data_dir.glob("*.cu")))
    total_files = len(cu_files)
    
    if total_files == 0:
        print(f"No .cu files found in {data_dir}")
        return
    
    print(f"Found {total_files} files to process")
    print(f"Predicting for {len(GPUS)} GPUs: {', '.join(GPU_COLUMN_NAMES)}")
    print("Starting predictions...\n")
    
    for idx, file_path in enumerate(cu_files, 1):
        filename = file_path.name
        print(f"[{idx}/{total_files}] Processing {filename}...")
        
        predictions_dict = {}
        
        for gpu_idx, gpu_name in enumerate(GPUS, 1):
            gpu_column = GPU_COLUMN_NAMES[gpu_idx - 1]
            print(f"  [{gpu_idx}/{len(GPUS)}] {gpu_column}...", end=" ", flush=True)
            
            # Predict
            prediction, computation = predict_execution_time(file_path, client, gpu_name)
            
            if prediction and computation:
                prompt, response = computation
                predictions_dict[gpu_column] = prediction
                print(f"✓ {prediction} ms")
                
                # Save computation process
                save_computation_process(filename, gpu_name, prompt, response)
            else:
                print("✗ Failed")
                predictions_dict[gpu_column] = ''
            
            # Rate limiting - small delay between requests
            if gpu_idx < len(GPUS) or idx < total_files:
                time.sleep(0.5)

        save_to_csv(filename, predictions_dict)
        print()
    
    print(f"\nCompleted! Processed {total_files} files for {len(GPUS)} GPUs.")
    print(f"Results saved to predictions.csv and computation_process.txt")


if __name__ == "__main__":
    main()

