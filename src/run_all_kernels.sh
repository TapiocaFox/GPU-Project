#!/usr/bin/env bash

set -uo pipefail

usage() {
    cat <<'EOF'
run_all_kernels.sh - Execute compiled GPU kernels and capture timing data

Usage:
  ./run_all_kernels.sh [options]

Options:
  -d, --dataset <name>   Dataset to run (test, validation, all). Default: all
  -n, --node <name>      Restrict to executables under the given node directory (e.g., cuda2)
  -m, --max <count>      Limit the number of kernels to run
  -k, --kernels-dir <p>  Override the kernels directory (default: ./kernels_src)
  -o, --output-dir <p>   Directory to store logs (default: ./kernel_execution_runs_<timestamp>)
      --dry-run          Print which kernels would run without executing them
  -h, --help             Show this help

Environment:
  KERNELS_DIR            Same as --kernels-dir when the flag is not provided
EOF
}

info()  { echo "[INFO]  $1"; }
warn()  { echo "[WARN]  $1" >&2; }
error() { echo "[ERROR] $1" >&2; }

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_KERNELS_DIR="$SCRIPT_DIR/kernels_src"
KERNELS_DIR="${KERNELS_DIR:-$DEFAULT_KERNELS_DIR}"
DATASET_FILTER="all"
NODE_FILTER=""
MAX_KERNELS=0
OUTPUT_DIR=""
DRY_RUN=0
TIME_CMD="${TIME_CMD:-/usr/bin/time}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--dataset)
            [[ $# -lt 2 ]] && { error "Missing value for $1"; usage; exit 1; }
            DATASET_FILTER="$2"
            shift 2
            ;;
        -n|--node)
            [[ $# -lt 2 ]] && { error "Missing value for $1"; usage; exit 1; }
            NODE_FILTER="$2"
            shift 2
            ;;
        -m|--max)
            [[ $# -lt 2 ]] && { error "Missing value for $1"; usage; exit 1; }
            MAX_KERNELS="$2"
            shift 2
            ;;
        -k|--kernels-dir)
            [[ $# -lt 2 ]] && { error "Missing value for $1"; usage; exit 1; }
            KERNELS_DIR="$2"
            shift 2
            ;;
        -o|--output-dir)
            [[ $# -lt 2 ]] && { error "Missing value for $1"; usage; exit 1; }
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            error "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

if [[ ! -d "$KERNELS_DIR" ]]; then
    error "Kernels directory not found: $KERNELS_DIR"
    exit 1
fi

if [[ -n "$TIME_CMD" && ! -x "$TIME_CMD" ]]; then
    warn "Configured time command '$TIME_CMD' is not executable; falling back to internal timer"
    TIME_CMD=""
fi

if ! command -v python3 >/dev/null 2>&1; then
    error "python3 is required to measure execution time"
    exit 1
fi

case "$DATASET_FILTER" in
    all|"")
        DATASETS=("test" "validation")
        ;;
    test|validation)
        DATASETS=("$DATASET_FILTER")
        ;;
    *)
        error "Unsupported dataset: $DATASET_FILTER"
        exit 1
        ;;
esac

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_ROOT="${OUTPUT_DIR:-$SCRIPT_DIR/kernel_execution_runs_$TIMESTAMP}"
mkdir -p "$RESULTS_ROOT"
JSON_FILE="$RESULTS_ROOT/execution_results.json"

run_kernel() {
    local exe_path="$1"
    local stdout_path="$2"
    local stderr_path="$3"
    local time_path="$4"
    local workdir="$5"
    shift 5
    local exe_args=("$@")

    mkdir -p "$(dirname "$stdout_path")" "$(dirname "$stderr_path")" "$(dirname "$time_path")"

    local started_at
    local ended_at
    local exit_code
    local duration="0"

    started_at=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    if [[ -n "$TIME_CMD" ]]; then
        if (cd "$workdir" && "$TIME_CMD" -f "%e" -o "$time_path" "$exe_path" "${exe_args[@]}") \
            >"$stdout_path" 2>"$stderr_path"; then
            exit_code=0
        else
            exit_code=$?
        fi
        if [[ -f "$time_path" ]]; then
            duration=$(tr -d $'\r\n' < "$time_path")
        fi
    else
        python3 - "$exe_path" "$stdout_path" "$stderr_path" "$workdir" "$time_path" "$(IFS=$'\t'; echo "${exe_args[*]}")" <<'PY'
import os
import subprocess
import sys
import time

exe_path, stdout_path, stderr_path, workdir, time_path, args_str = sys.argv[1:7]
args = args_str.split("\t") if args_str else []
os.makedirs(os.path.dirname(stdout_path), exist_ok=True)
os.makedirs(os.path.dirname(stderr_path), exist_ok=True)
start = time.time()
with open(stdout_path, "w") as out_file, open(stderr_path, "w") as err_file:
    result = subprocess.run([exe_path] + args, cwd=workdir, stdout=out_file, stderr=err_file)
end = time.time()
with open(time_path, "w") as f:
    f.write(f"{end-start:.6f}")
sys.exit(result.returncode)
PY
        exit_code=$?
        if [[ -f "$time_path" ]]; then
            duration=$(tr -d $'\r\n' < "$time_path")
        fi
    fi

    ended_at=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    echo "$started_at|$ended_at|$duration|$exit_code"
}

info "Scanning executables under $KERNELS_DIR"

declare -a EXECUTABLES=()

for dataset in "${DATASETS[@]}"; do
    DATASET_DIR="$KERNELS_DIR/$dataset"
    if [[ ! -d "$DATASET_DIR" ]]; then
        warn "Dataset directory not found: $DATASET_DIR"
        continue
    fi

    FIND_ARGS=( "$DATASET_DIR" -type f -name "benchmark_*" -perm -111 )
    if [[ -n "$NODE_FILTER" ]]; then
        FIND_ARGS+=( -path "*/$NODE_FILTER/benchmark_*" )
    fi

    while IFS= read -r exe_path; do
        EXECUTABLES+=("$exe_path")
    done < <(find "${FIND_ARGS[@]}" 2>/dev/null | LC_ALL=C sort)
done

TOTAL_FOUND=${#EXECUTABLES[@]}
if (( TOTAL_FOUND == 0 )); then
    error "No executable kernels found with the current filters."
    exit 1
fi

if (( MAX_KERNELS > 0 && MAX_KERNELS < TOTAL_FOUND )); then
    info "Limiting execution to $MAX_KERNELS kernels (out of $TOTAL_FOUND)"
    TOTAL_TARGET=$MAX_KERNELS
else
    TOTAL_TARGET=$TOTAL_FOUND
fi

SUCCESS_COUNT=0
FAIL_COUNT=0
SKIPPED_COUNT=0
JSON_LINES_FILE="$RESULTS_ROOT/.json_lines.tmp"
> "$JSON_LINES_FILE"

for (( idx=0; idx<TOTAL_TARGET; idx++ )); do
    exe="${EXECUTABLES[$idx]}"
    rel_path="${exe#$KERNELS_DIR/}"
    IFS='/' read -r dataset folder_id kernel_id node_dir exe_name <<< "$rel_path"

    if [[ -z "$dataset" || -z "$folder_id" || -z "$kernel_id" || -z "$node_dir" || -z "$exe_name" ]]; then
        warn "Unable to parse kernel metadata from $exe"
        continue
    fi

    run_label="${dataset}_${folder_id}_${kernel_id}_${node_dir}"
    run_dir="$RESULTS_ROOT/$run_label"
    mkdir -p "$run_dir"
    stdout_log="$run_dir/stdout.log"
    stderr_log="$run_dir/stderr.log"
    time_log="$run_dir/time.log"

    printf "[%d/%d] %s\n" "$((idx + 1))" "$TOTAL_TARGET" "$rel_path"

    if (( DRY_RUN )); then
        started_at=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
        ended_at="$started_at"
        duration="0.0"
        exit_code="0"
        status="skipped"
        ((SKIPPED_COUNT++))
    else
        run_result=$(run_kernel "$exe" "$stdout_log" "$stderr_log" "$time_log" "$(dirname "$exe")" "$folder_id")
        IFS='|' read -r started_at ended_at duration exit_code <<< "$run_result"
        if [[ "$exit_code" == "0" ]]; then
            status="success"
            ((SUCCESS_COUNT++))
        else
            status="failed"
            ((FAIL_COUNT++))
            warn "Kernel failed (exit $exit_code): $rel_path"
        fi
    fi

    # Collect JSON data (silently fail if Python script has issues)
    python3 - "$dataset" "$folder_id" "$kernel_id" "$node_dir" "$exe" "$started_at" "$ended_at" "$duration" "$exit_code" "$status" "$stdout_log" "$stderr_log" "$JSON_LINES_FILE" 2>/dev/null <<'PY' || true
import sys
import json

try:
    dataset, folder_id, kernel_id, node_dir, exe, started_at, ended_at, duration, exit_code, status, stdout_log, stderr_log, json_lines_file = sys.argv[1:14]
    
    result = {
        "dataset": dataset,
        "folder_id": folder_id,
        "kernel_id": kernel_id,
        "node": node_dir,
        "executable": exe,
        "started_at": started_at,
        "ended_at": ended_at,
        "execution_time_seconds": float(duration),
        "exit_code": int(exit_code),
        "status": status,
        "stdout_log": stdout_log,
        "stderr_log": stderr_log
    }
    
    with open(json_lines_file, 'a') as f:
        f.write(json.dumps(result) + '\n')
except Exception as e:
    # Silently fail - don't block execution
    pass
PY
done

# Convert JSON lines to JSON array
if [[ -f "$JSON_LINES_FILE" ]] && [[ -s "$JSON_LINES_FILE" ]]; then
    if ! python3 - "$JSON_LINES_FILE" "$JSON_FILE" <<'PY' 2>/dev/null; then
        warn "Failed to convert JSON lines to JSON array, creating empty JSON file"
        echo "[]" > "$JSON_FILE"
    fi
import sys
import json

json_lines_file, output_file = sys.argv[1:3]

results = []
try:
    with open(json_lines_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    # Skip invalid JSON lines
                    pass
except FileNotFoundError:
    pass
except Exception as e:
    # If anything fails, just create empty array
    results = []

try:
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
except Exception as e:
    # Fallback: create empty JSON
    with open(output_file, 'w') as f:
        f.write("[]\n")
    sys.exit(1)

# Clean up temp file
import os
try:
    os.remove(json_lines_file)
except:
    pass
PY
else
    # Create empty JSON array if no results file or file is empty
    echo "[]" > "$JSON_FILE"
    [[ -f "$JSON_LINES_FILE" ]] && rm -f "$JSON_LINES_FILE"
fi

info "Execution summary:"
info "  Success: $SUCCESS_COUNT"
info "  Failed : $FAIL_COUNT"
info "  Skipped: $SKIPPED_COUNT"
info "Results stored in: $RESULTS_ROOT"
info "Results JSON: $JSON_FILE"

if (( FAIL_COUNT > 0 )); then
    exit 2
fi

