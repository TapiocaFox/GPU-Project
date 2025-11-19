import pandas as pd
import numpy as np

def prepare_true_values(df):
    """
    Extracts the join key and renames the true value column.
    The key is 'kernel_i_kernel_j' extracted from the 'filename'.
    """
    df['key_id'] = df['filename'].apply(
        lambda x: '_'.join(x.split('_')[:2])
    )
    return df[['key_id', 'time_ms']].rename(columns={'time_ms': 'true_value'})

def calculate_mape(true_values, predictions):
    """Calculates the Mean Absolute Percentage Error (MAPE)."""
    # Filter out true values that are zero to avoid division by zero
    non_zero_mask = true_values != 0
    true_values = true_values[non_zero_mask]
    predictions = predictions[non_zero_mask]

    # Calculate Absolute Percentage Error (APE)
    ape = np.abs((true_values - predictions) / true_values)

    # Calculate MAPE (in percent)
    mape = ape.mean() * 100
    return mape

# Define the prediction file and the true measurement files
PREDICTION_FILE = "multi_gpu_simulation_results.csv"
TRUE_VALUE_FILES = {
    'cuda2': "kernel_results_b5_t15_cuda2.csv",
    'cuda3': "kernel_results_b5_t15_cuda3.csv",
    'cuda4': "kernel_results_b5_t15_cuda4.csv",
    'cuda5': "kernel_results_b5_t15_cuda5.csv"
}

# Load the prediction data
predictions_df = pd.read_csv(PREDICTION_FILE)
predictions_df = predictions_df.rename(columns={'predicted_time_ms': 'prediction'})

# List to hold the MAPE results for each architecture
results = []

for gpu_id, true_file in TRUE_VALUE_FILES.items():
    # Load and prepare true values
    true_df = pd.read_csv(true_file)
    true_df_clean = prepare_true_values(true_df)

    # Get the GPU architecture name for the result table
    gpu_architecture = predictions_df[predictions_df['gpu_id'] == gpu_id]['gpu_architecture'].iloc[0]

    # Filter predictions for the current gpu_id
    predictions_filtered = predictions_df[predictions_df['gpu_id'] == gpu_id]

    # Merge predictions and true values using the kernel ID ('key_id' in true_df_clean and 'kernel_id' in predictions_df)
    merged_df = pd.merge(
        predictions_filtered,
        true_df_clean,
        left_on='kernel_id',
        right_on='key_id',
        how='inner'
    ).dropna(subset=['true_value', 'prediction'])

    # Calculate MAPE
    mape_value = calculate_mape(merged_df['true_value'], merged_df['prediction'])

    # Store results
    results.append({
        'GPU_Architecture': gpu_architecture,
        'GPU_ID': gpu_id,
        'MAPE': mape_value
    })

# Convert results to a DataFrame for clean presentation
results_df = pd.DataFrame(results)

# Sort results by MAPE and print
results_df_sorted = results_df.sort_values(by='MAPE')
print(results_df_sorted.to_markdown(index=False, floatfmt=".2f"))