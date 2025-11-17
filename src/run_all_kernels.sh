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
  -t, --timeout <sec>    Timeout for each kernel execution in seconds (default: 60)
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
TIMEOUT_SECONDS=60
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
        -t|--timeout)
            [[ $# -lt 2 ]] && { error "Missing value for $1"; usage; exit 1; }
            TIMEOUT_SECONDS="$2"
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

# Check if timeout command is available
if ! command -v timeout >/dev/null 2>&1; then
    warn "timeout command not found; timeout feature will use Python subprocess timeout instead"
    # Force use of Python fallback if timeout is not available
    if [[ -n "$TIME_CMD" ]]; then
        warn "Falling back to Python timer since timeout command is not available"
        TIME_CMD=""
    fi
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
        # Quote arguments properly for sh -c
        local quoted_args=""
        for arg in "${exe_args[@]}"; do
            quoted_args+=" $(printf %q "$arg")"
        done
        if timeout "$TIMEOUT_SECONDS" sh -c "cd $(printf %q "$workdir") && $(printf %q "$TIME_CMD") -f '%e' -o $(printf %q "$time_path") $(printf %q "$exe_path")$quoted_args" \
            >"$stdout_path" 2>"$stderr_path"; then
            exit_code=0
        else
            exit_code=$?
            # Check if timeout occurred (exit code 124 means timeout)
            if (( exit_code == 124 )) || (( exit_code >= 128 && exit_code < 128 + 15 )); then
                echo "Kernel execution timed out after ${TIMEOUT_SECONDS}s" >> "$stderr_path"
                # If timeout occurred, set duration to timeout value
                if [[ ! -f "$time_path" ]] || [[ ! -s "$time_path" ]]; then
                    echo "$TIMEOUT_SECONDS" > "$time_path"
                fi
            fi
        fi
        if [[ -f "$time_path" ]]; then
            duration=$(tr -d $'\r\n' < "$time_path")
        fi
    else
        python3 - "$exe_path" "$stdout_path" "$stderr_path" "$workdir" "$time_path" "$TIMEOUT_SECONDS" "$(IFS=$'\t'; echo "${exe_args[*]}")" <<'PYTIMER' 2>/dev/null
import os
import subprocess
import sys
import time
import re

exe_path, stdout_path, stderr_path, workdir, time_path, timeout_sec, args_str = sys.argv[1:8]
args = args_str.split("\t") if args_str else []
timeout_float = float(timeout_sec) if timeout_sec else None
os.makedirs(os.path.dirname(stdout_path), exist_ok=True)
os.makedirs(os.path.dirname(stderr_path), exist_ok=True)
start = time.time()
with open(stdout_path, "w") as out_file, open(stderr_path, "w") as err_file:
    try:
        result = subprocess.run([exe_path] + args, cwd=workdir, stdout=out_file, stderr=err_file, timeout=timeout_float)
        exit_code = result.returncode
    except subprocess.TimeoutExpired:
        exit_code = 124
        err_file.write(f"Kernel execution timed out after {timeout_sec}s\n")
end = time.time()
with open(time_path, "w") as f:
    f.write(f"{end-start:.6f}")
sys.exit(exit_code)
PYTIMER
        exit_code=$?
        if [[ -f "$time_path" ]]; then
            duration=$(tr -d $'\r\n' < "$time_path")
        fi
    fi

    # Parse execution time from kernel stdout output
    # Format: [time_in_microseconds,(blockx,blocky),(xsize,ysize)]
    # Extract the first number from the bracket pattern (prefer last match if multiple lines)
    if [[ -f "$stdout_path" ]] && [[ -s "$stdout_path" ]]; then
        parsed_time=$(python3 <<PYEOF 2>/dev/null
import sys
import re

stdout_file = "$stdout_path"
last_match = None
try:
    with open(stdout_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Match pattern [number, (e.g., [2197.64,(8,8),(240,240)])
            match = re.search(r'\[([0-9]+\.?[0-9]*),', line)
            if match:
                # Convert microseconds to seconds
                time_us = float(match.group(1))
                time_sec = time_us / 1000000.0
                last_match = time_sec
    # Use the last match if found, otherwise try alternative patterns
    if last_match is not None:
        print(f"{last_match:.6f}")
    else:
        # Fallback: try to find any number pattern
        with open(stdout_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                match = re.search(r'^([0-9]+\.?[0-9]*)', line)
                if match:
                    time_val = float(match.group(1))
                    # Assume microseconds if > 1000 (likely microseconds), otherwise assume seconds
                    if time_val > 1000:
                        time_val = time_val / 1000000.0
                    print(f"{time_val:.6f}")
                    break
except Exception:
    pass
PYEOF
        )
        # Set to empty if parsing failed
        parsed_time="${parsed_time:-}"
        
        # Validate parsed time is a valid positive number
        if [[ -n "$parsed_time" ]]; then
            # Check if it's a valid positive float using python
            if python3 -c "float('$parsed_time') > 0" 2>/dev/null; then
                duration="$parsed_time"
            fi
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
        elif [[ "$exit_code" == "124" ]]; then
            status="timeout"
            ((FAIL_COUNT++))
            warn "Kernel timed out after ${TIMEOUT_SECONDS}s: $rel_path"
        else
            status="failed"
            ((FAIL_COUNT++))
            warn "Kernel failed (exit $exit_code): $rel_path"
        fi
    fi

    # Collect JSON data (silently fail if Python script has issues)
    python3 - "$dataset" "$folder_id" "$kernel_id" "$node_dir" "$exe" "$started_at" "$ended_at" "$duration" "$exit_code" "$status" "$stdout_log" "$stderr_log" "$JSON_LINES_FILE" 2>/dev/null <<'PYJSON' || true
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
PYJSON
done

# Convert JSON lines to JSON array
if [[ -f "$JSON_LINES_FILE" ]] && [[ -s "$JSON_LINES_FILE" ]]; then
    python3 - "$JSON_LINES_FILE" "$JSON_FILE" 2>/dev/null <<'PYEOF'
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
PYEOF
    if [[ $? -ne 0 ]]; then
        warn "Failed to convert JSON lines to JSON array, creating empty JSON file"
        echo "[]" > "$JSON_FILE"
    fi
else
    # Create empty JSON array if no results file or file is empty
    echo "[]" > "$JSON_FILE"
    [[ -f "$JSON_LINES_FILE" ]] && rm -f "$JSON_LINES_FILE"
fi

# Generate simplified JSON with filename as key and execution time as value
SIMPLIFIED_JSON_FILE="$RESULTS_ROOT/execution_times.json"
if [[ -f "$JSON_FILE" ]] && [[ -s "$JSON_FILE" ]]; then
    python3 - "$JSON_FILE" "$SIMPLIFIED_JSON_FILE" 2>/dev/null <<'PYSIMPLIFY'
import sys
import json
import os

input_file, output_file = sys.argv[1:3]

result_dict = {}
try:
    with open(input_file, 'r') as f:
        results = json.load(f)
    
    for item in results:
        # Extract filename from executable path
        executable_path = item.get('executable', '')
        filename = os.path.basename(executable_path)
        
        # Get execution time
        exec_time = item.get('execution_time_seconds', 0.0)
        
        # Use filename as key, execution time as value
        if filename:
            result_dict[filename] = exec_time
    
    with open(output_file, 'w') as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)
except Exception as e:
    # If anything fails, create empty dict
    with open(output_file, 'w') as f:
        json.dump({}, f, indent=2)
PYSIMPLIFY
    if [[ $? -ne 0 ]]; then
        warn "Failed to generate simplified JSON file"
        echo "{}" > "$SIMPLIFIED_JSON_FILE"
    fi
else
    echo "{}" > "$SIMPLIFIED_JSON_FILE"
fi

info "Execution summary:"
info "  Success: $SUCCESS_COUNT"
info "  Failed : $FAIL_COUNT"
info "  Skipped: $SKIPPED_COUNT"
info "Results stored in: $RESULTS_ROOT"
info "Results JSON: $JSON_FILE"
info "Execution times JSON: $SIMPLIFIED_JSON_FILE"

if (( FAIL_COUNT > 0 )); then
    exit 2
fi

