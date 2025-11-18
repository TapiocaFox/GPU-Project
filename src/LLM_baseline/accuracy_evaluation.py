#!/usr/bin/env python3
"""
Accuracy Evaluation Script
Compares LLM baseline predictions with actual execution times.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
from typing import Dict, List, Tuple, Optional

# Paths
LLM_BASELINE_PATH = Path(__file__).parent / "all_kernels_analysis_20251116_045342.json"
KERNEL_EXECUTIONS_DIR = Path(__file__).parent.parent / "kernel_executions"
OUTPUT_DIR = Path(__file__).parent


def load_llm_baseline(baseline_path: Path) -> Dict:
    """Load LLM baseline predictions."""
    print(f"Loading LLM baseline from {baseline_path}...")
    with open(baseline_path, 'r') as f:
        data = json.load(f)
    
    # Extract analyses if wrapped in metadata
    if "analyses" in data:
        analyses = data["analyses"]
    else:
        analyses = data
    
    print(f"Loaded {len(analyses)} kernel analyses from LLM baseline")
    return analyses


def parse_kernel_path(path_key: str) -> Optional[Tuple[str, str, str]]:
    """
    Parse kernel path like "test/10615/4/main.cu" or "validation/10073/327/softmax_gradient_kernel.cu"
    Returns (dataset, folder_id, kernel_id) or None if invalid
    """
    parts = path_key.split('/')
    if len(parts) < 3:
        return None
    
    dataset = parts[0]  # test or validation
    folder_id = parts[1]
    kernel_id = parts[2]
    
    return (dataset, folder_id, kernel_id)


def aggregate_llm_predictions(analyses: Dict) -> Dict[Tuple[str, str, str], Dict]:
    """
    Aggregate LLM predictions by (dataset, folder_id, kernel_id).
    If multiple files exist for the same kernel, average the predictions.
    """
    aggregated = defaultdict(list)
    
    for path_key, analysis in analyses.items():
        if path_key.startswith('_'):
            continue
        
        parsed = parse_kernel_path(path_key)
        if parsed is None:
            continue
        
        key = parsed
        if "execution_time_estimate" in analysis:
            estimate = analysis["execution_time_estimate"]
            aggregated[key].append(estimate)
    
    # Average predictions for kernels with multiple files
    result = {}
    for key, estimates in aggregated.items():
        if not estimates:
            continue
        
        # Calculate average of typical values
        typical_values = [e.get("microseconds_typical", 0) for e in estimates if "microseconds_typical" in e]
        min_values = [e.get("microseconds_min", 0) for e in estimates if "microseconds_min" in e]
        max_values = [e.get("microseconds_max", 0) for e in estimates if "microseconds_max" in e]
        
        if typical_values:
            result[key] = {
                "microseconds_typical": np.mean(typical_values),
                "microseconds_min": np.mean(min_values) if min_values else None,
                "microseconds_max": np.mean(max_values) if max_values else None,
                "num_files": len(estimates)
            }
    
    print(f"Aggregated {len(result)} unique kernels from LLM baseline")
    return result


def load_execution_results(executions_dir: Path) -> Dict[Tuple[str, str, str], float]:
    """
    Load actual execution times from all execution folders.
    Returns dict mapping (dataset, folder_id, kernel_id) -> execution_time_seconds
    """
    execution_times = {}
    
    # Find all execution_results.json files
    for execution_folder in executions_dir.iterdir():
        if not execution_folder.is_dir():
            continue
        
        results_file = execution_folder / "execution_results.json"
        if not results_file.exists():
            continue
        
        print(f"Loading execution results from {execution_folder.name}...")
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        for result in results:
            if result.get("status") != "success":
                continue
            
            dataset = result.get("dataset")
            folder_id = result.get("folder_id")
            kernel_id = result.get("kernel_id")
            exec_time = result.get("execution_time_seconds")
            
            if dataset and folder_id and kernel_id and exec_time is not None:
                key = (dataset, str(folder_id), str(kernel_id))
                # If multiple executions exist, keep the first one (or could average)
                if key not in execution_times:
                    execution_times[key] = exec_time
    
    print(f"Loaded {len(execution_times)} actual execution times")
    return execution_times


def match_and_compare(
    llm_predictions: Dict[Tuple[str, str, str], Dict],
    actual_times: Dict[Tuple[str, str, str], float]
) -> List[Dict]:
    """
    Match LLM predictions with actual execution times and calculate errors.
    Returns list of comparison results.
    """
    comparisons = []
    
    # Find matching kernels
    matched_keys = set(llm_predictions.keys()) & set(actual_times.keys())
    print(f"\nFound {len(matched_keys)} matching kernels for comparison")
    
    for key in matched_keys:
        llm_pred = llm_predictions[key]
        actual_time_seconds = actual_times[key]
        
        # Convert LLM prediction from microseconds to seconds
        llm_pred_seconds = llm_pred["microseconds_typical"] / 1_000_000
        
        # Calculate errors
        absolute_error = abs(llm_pred_seconds - actual_time_seconds)
        relative_error = absolute_error / actual_time_seconds if actual_time_seconds > 0 else float('inf')
        percentage_error = relative_error * 100
        
        comparisons.append({
            "dataset": key[0],
            "folder_id": key[1],
            "kernel_id": key[2],
            "llm_prediction_seconds": llm_pred_seconds,
            "llm_prediction_microseconds": llm_pred["microseconds_typical"],
            "actual_time_seconds": actual_time_seconds,
            "absolute_error_seconds": absolute_error,
            "relative_error": relative_error,
            "percentage_error": percentage_error,
            "num_files_in_llm": llm_pred.get("num_files", 1)
        })
    
    return comparisons


def calculate_accuracy_metrics(comparisons: List[Dict]) -> Dict:
    """Calculate overall accuracy metrics."""
    if not comparisons:
        return {}
    
    llm_preds = np.array([c["llm_prediction_seconds"] for c in comparisons])
    actual_times = np.array([c["actual_time_seconds"] for c in comparisons])
    errors = np.array([c["absolute_error_seconds"] for c in comparisons])
    relative_errors = np.array([c["relative_error"] for c in comparisons])
    
    # Mean Absolute Error (MAE)
    mae = np.mean(errors)
    
    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(np.mean(errors ** 2))
    
    # Mean Absolute Percentage Error (MAPE)
    mape = np.mean(relative_errors) * 100
    
    # Median Absolute Percentage Error (MdAPE)
    mdape = np.median(relative_errors) * 100
    
    # R² (Coefficient of Determination)
    ss_res = np.sum((actual_times - llm_preds) ** 2)
    ss_tot = np.sum((actual_times - np.mean(actual_times)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Correlation coefficient
    correlation = np.corrcoef(llm_preds, actual_times)[0, 1] if len(llm_preds) > 1 else 0
    
    # Percentiles of relative error
    error_percentiles = {
        "p25": np.percentile(relative_errors, 25) * 100,
        "p50": np.percentile(relative_errors, 50) * 100,
        "p75": np.percentile(relative_errors, 75) * 100,
        "p90": np.percentile(relative_errors, 90) * 100,
        "p95": np.percentile(relative_errors, 95) * 100,
        "p99": np.percentile(relative_errors, 99) * 100,
    }
    
    # Count predictions within certain error thresholds
    within_10pct = np.sum(relative_errors <= 0.10) / len(comparisons) * 100
    within_25pct = np.sum(relative_errors <= 0.25) / len(comparisons) * 100
    within_50pct = np.sum(relative_errors <= 0.50) / len(comparisons) * 100
    within_2x = np.sum(relative_errors <= 1.0) / len(comparisons) * 100
    
    return {
        "num_kernels": len(comparisons),
        "mean_absolute_error_seconds": float(mae),
        "root_mean_squared_error_seconds": float(rmse),
        "mean_absolute_percentage_error": float(mape),
        "median_absolute_percentage_error": float(mdape),
        "r_squared": float(r_squared),
        "correlation_coefficient": float(correlation),
        "relative_error_percentiles": error_percentiles,
        "predictions_within_threshold": {
            "within_10_percent": float(within_10pct),
            "within_25_percent": float(within_25pct),
            "within_50_percent": float(within_50pct),
            "within_2x": float(within_2x)
        },
        "min_actual_time_seconds": float(np.min(actual_times)),
        "max_actual_time_seconds": float(np.max(actual_times)),
        "mean_actual_time_seconds": float(np.mean(actual_times)),
        "min_llm_prediction_seconds": float(np.min(llm_preds)),
        "max_llm_prediction_seconds": float(np.max(llm_preds)),
        "mean_llm_prediction_seconds": float(np.mean(llm_preds))
    }


def main():
    """Main execution function."""
    print("=" * 80)
    print("LLM Baseline Accuracy Evaluation")
    print("=" * 80)
    
    # Load data
    llm_analyses = load_llm_baseline(LLM_BASELINE_PATH)
    llm_predictions = aggregate_llm_predictions(llm_analyses)
    actual_times = load_execution_results(KERNEL_EXECUTIONS_DIR)
    
    # Match and compare
    comparisons = match_and_compare(llm_predictions, actual_times)
    
    if not comparisons:
        print("\n❌ No matching kernels found! Check that the kernel IDs match.")
        return
    
    # Calculate metrics
    metrics = calculate_accuracy_metrics(comparisons)
    
    # Print summary
    print("\n" + "=" * 80)
    print("ACCURACY METRICS SUMMARY")
    print("=" * 80)
    print(f"Number of kernels compared: {metrics['num_kernels']}")
    print(f"\nError Metrics:")
    print(f"  Mean Absolute Error (MAE): {metrics['mean_absolute_error_seconds']:.6f} seconds")
    print(f"  Root Mean Squared Error (RMSE): {metrics['root_mean_squared_error_seconds']:.6f} seconds")
    print(f"  Mean Absolute Percentage Error (MAPE): {metrics['mean_absolute_percentage_error']:.2f}%")
    print(f"  Median Absolute Percentage Error (MdAPE): {metrics['median_absolute_percentage_error']:.2f}%")
    print(f"\nCorrelation Metrics:")
    print(f"  R² (Coefficient of Determination): {metrics['r_squared']:.4f}")
    print(f"  Correlation Coefficient: {metrics['correlation_coefficient']:.4f}")
    print(f"\nPrediction Accuracy:")
    print(f"  Within 10% error: {metrics['predictions_within_threshold']['within_10_percent']:.2f}%")
    print(f"  Within 25% error: {metrics['predictions_within_threshold']['within_25_percent']:.2f}%")
    print(f"  Within 50% error: {metrics['predictions_within_threshold']['within_50_percent']:.2f}%")
    print(f"  Within 2x error: {metrics['predictions_within_threshold']['within_2x']:.2f}%")
    print(f"\nRelative Error Percentiles:")
    for percentile, value in metrics['relative_error_percentiles'].items():
        print(f"  {percentile}: {value:.2f}%")
    
    # Calculate metrics by dataset
    test_comparisons = [c for c in comparisons if c["dataset"] == "test"]
    validation_comparisons = [c for c in comparisons if c["dataset"] == "validation"]
    
    test_metrics = calculate_accuracy_metrics(test_comparisons) if test_comparisons else {}
    validation_metrics = calculate_accuracy_metrics(validation_comparisons) if validation_comparisons else {}
    
    # Print dataset-specific summaries
    if test_metrics:
        print(f"\n--- Test Dataset ({test_metrics['num_kernels']} kernels) ---")
        print(f"  MAPE: {test_metrics['mean_absolute_percentage_error']:.2f}%")
        print(f"  MdAPE: {test_metrics['median_absolute_percentage_error']:.2f}%")
        print(f"  R²: {test_metrics['r_squared']:.4f}")
    
    if validation_metrics:
        print(f"\n--- Validation Dataset ({validation_metrics['num_kernels']} kernels) ---")
        print(f"  MAPE: {validation_metrics['mean_absolute_percentage_error']:.2f}%")
        print(f"  MdAPE: {validation_metrics['median_absolute_percentage_error']:.2f}%")
        print(f"  R²: {validation_metrics['r_squared']:.4f}")
    
    # Save results
    output_data = {
        "summary_metrics": metrics,
        "test_dataset_metrics": test_metrics,
        "validation_dataset_metrics": validation_metrics,
        "individual_comparisons": comparisons
    }
    
    output_file = OUTPUT_DIR / "accuracy_evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✅ Results saved to {output_file}")
    
    # Save summary only
    summary_file = OUTPUT_DIR / "accuracy_evaluation_summary.json"
    summary_data = {
        "overall": metrics,
        "test": test_metrics,
        "validation": validation_metrics
    }
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"✅ Summary saved to {summary_file}")


if __name__ == "__main__":
    main()

