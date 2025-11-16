#!/bin/bash

# compile_gpu_clusters_fixed.sh - Fixed version with proper PTX generation

DATASET=${1:-test}
MAX_KERNELS=${2:-0}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() { echo -e "${BLUE}ℹ $1${NC}"; }
print_success() { echo -e "${GREEN}✓ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠ $1${NC}"; }
print_error() { echo -e "${RED}✗ $1${NC}"; }

# Detect current GPU cluster (use short hostname)
CURRENT_NODE=$(hostname -s)
print_info "Current cluster: $CURRENT_NODE"

# GPU-specific configurations
case "$CURRENT_NODE" in
    cuda2*)
        GPU_NAME="RTX_2080_Ti"
        COMPUTE_CAP="75"
        GPU_ARCH="sm_75"
        ;;
    cuda3*)
        GPU_NAME="TITAN_V"
        COMPUTE_CAP="70"
        GPU_ARCH="sm_70"
        ;;
    cuda4*)
        GPU_NAME="GTX_TITAN_X"
        COMPUTE_CAP="52"
        GPU_ARCH="sm_52"
        ;;
    cuda5*)
        GPU_NAME="RTX_4070"
        COMPUTE_CAP="89"
        GPU_ARCH="sm_89"
        ;;
    *)
        print_warning "Unknown cluster: $CURRENT_NODE, using default settings"
        GPU_NAME="Unknown"
        COMPUTE_CAP="75"
        GPU_ARCH="sm_75"
        ;;
esac

print_info "GPU: $GPU_NAME (Compute Capability: $COMPUTE_CAP)"

# Validate inputs
if [[ "$DATASET" != "test" && "$DATASET" != "validation" ]]; then
    print_error "Invalid dataset: $DATASET. Must be 'test' or 'validation'"
    exit 1
fi

KERNELS_DIR="kernels_src"
DATASET_DIR="$KERNELS_DIR/$DATASET"
RESULTS_DIR="compile_results_${DATASET}_${CURRENT_NODE}_${TIMESTAMP}"

if [ ! -d "$DATASET_DIR" ]; then
    print_error "Dataset directory not found: $DATASET_DIR"
    exit 1
fi

mkdir -p "$RESULTS_DIR"

# Setup result files with absolute paths
WORK_DIR="$(pwd)"
SUCCESS_LOG="$WORK_DIR/$RESULTS_DIR/successful_kernels.txt"
FAILED_LOG="$WORK_DIR/$RESULTS_DIR/failed_kernels.txt"
BUILD_ERRORS="$WORK_DIR/$RESULTS_DIR/build_errors.log"
SUMMARY_FILE="$WORK_DIR/$RESULTS_DIR/compilation_summary.txt"

# Clear log files
> "$SUCCESS_LOG"
> "$FAILED_LOG"
> "$BUILD_ERRORS"

# Find all kernels
ALL_KERNEL_DIRS=($(find "$DATASET_DIR" -name "main.cu" -exec dirname {} \; | sort))
TOTAL_AVAILABLE=${#ALL_KERNEL_DIRS[@]}

if [ $MAX_KERNELS -eq 0 ] || [ $MAX_KERNELS -gt $TOTAL_AVAILABLE ]; then
    KERNEL_DIRS=("${ALL_KERNEL_DIRS[@]}")
    TOTAL_KERNELS=$TOTAL_AVAILABLE
else
    KERNEL_DIRS=("${ALL_KERNEL_DIRS[@]:0:$MAX_KERNELS}")
    TOTAL_KERNELS=$MAX_KERNELS
fi

echo "============================================"
echo "GPU Cluster-Aware Kernel Compilation"
echo "============================================"
echo "Cluster: $CURRENT_NODE"
echo "GPU: $GPU_NAME"
echo "Compute Capability: $COMPUTE_CAP"
echo "Dataset: $DATASET"
echo "Kernels to compile: $TOTAL_KERNELS out of $TOTAL_AVAILABLE available"
echo "Results directory: $RESULTS_DIR"
echo ""

# CUDA compiler info
print_info "CUDA Compiler: $(nvcc --version | grep release)"
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
    print_info "Detected GPU: $GPU_INFO"
fi

# Initialize counters
SUCCESSFUL_BUILDS=0
FAILED_BUILDS=0
PTX_GENERATED=0
START_TIME=$(date +%s)

print_info "Starting GPU-specific compilation process..."
echo ""

# Process each kernel
for i in "${!KERNEL_DIRS[@]}"; do
    KERNEL_DIR="${KERNEL_DIRS[$i]}"
    KERNEL_NUM=$((i + 1))
    
    # Extract folder and kernel IDs
    RELATIVE_PATH=${KERNEL_DIR#$DATASET_DIR/}
    IFS='/' read -ra PATH_PARTS <<< "$RELATIVE_PATH"
    FOLDER_ID=${PATH_PARTS[0]}
    KERNEL_ID=${PATH_PARTS[1]}
    
    echo "[$KERNEL_NUM/$TOTAL_KERNELS] Processing Kernel $FOLDER_ID/$KERNEL_ID"
    
    cd "$KERNEL_DIR"
    
    # Create cluster-specific directory (use short hostname)
    mkdir -p "$CURRENT_NODE"
    
    # Output files
    BUILD_OUTPUT="$CURRENT_NODE/benchmark_${FOLDER_ID}_${KERNEL_ID}_${GPU_NAME}"
    PTX_OUTPUT="kernel_${FOLDER_ID}_${KERNEL_ID}_${GPU_NAME}.ptx"
    
    print_info "  Building for $GPU_NAME..."
    
    # Step 1: Build executable
    NVCC_EXEC_CMD="nvcc -arch=$GPU_ARCH -O3 -o $BUILD_OUTPUT main.cu"
    
    if timeout 60 $NVCC_EXEC_CMD 2>"compile_error.tmp"; then
        print_success "  Executable build successful"
        
        # Step 2: Generate PTX file
        NVCC_PTX_CMD="nvcc --ptx -arch=$GPU_ARCH -o $PTX_OUTPUT main.cu"
        
        if timeout 30 $NVCC_PTX_CMD 2>"ptx_error.tmp"; then
            print_success "  PTX generation successful"
            ((PTX_GENERATED++))
        else
            print_warning "  PTX generation failed"
        fi
        
        ((SUCCESSFUL_BUILDS++))
        
        # Log success with detailed info
        EXEC_SIZE=$(stat -f%z "$BUILD_OUTPUT" 2>/dev/null || stat -c%s "$BUILD_OUTPUT" 2>/dev/null || echo "unknown")
        PTX_SIZE=$(stat -f%z "$PTX_OUTPUT" 2>/dev/null || stat -c%s "$PTX_OUTPUT" 2>/dev/null || echo "unknown")
        
        echo "$DATASET,$FOLDER_ID,$KERNEL_ID,$GPU_NAME,$BUILD_OUTPUT,$PTX_OUTPUT,$(pwd)" >> "$SUCCESS_LOG"
        
        print_success "  Executable: $BUILD_OUTPUT (${EXEC_SIZE} bytes)"
        if [ -f "$PTX_OUTPUT" ]; then
            print_success "  PTX file: $PTX_OUTPUT (${PTX_SIZE} bytes)"
        fi
        
    else
        print_error "  Build failed"
        ((FAILED_BUILDS++))
        
        # Log failure
        echo "$DATASET,$FOLDER_ID,$KERNEL_ID,$GPU_NAME,FAILED,FAILED,$(pwd)" >> "$FAILED_LOG"
        
        # Save error details
        echo "=== Kernel $FOLDER_ID/$KERNEL_ID ($GPU_NAME) Build Error ===" >> "$BUILD_ERRORS"
        echo "Cluster: $CURRENT_NODE" >> "$BUILD_ERRORS"
        echo "GPU: $GPU_NAME (compute_$COMPUTE_CAP)" >> "$BUILD_ERRORS"
        echo "Directory: $(pwd)" >> "$BUILD_ERRORS"
        echo "Command: $NVCC_EXEC_CMD" >> "$BUILD_ERRORS"
        if [ -f "compile_error.tmp" ]; then
            cat "compile_error.tmp" >> "$BUILD_ERRORS"
        fi
        echo "" >> "$BUILD_ERRORS"
    fi
    
    # Clean up temp files
    rm -f "compile_error.tmp" "ptx_error.tmp"
    
    cd "$WORK_DIR"
    
    # Progress update every 20 kernels
    if [ $((KERNEL_NUM % 20)) -eq 0 ] || [ $KERNEL_NUM -eq $TOTAL_KERNELS ]; then
        SUCCESS_RATE=$((SUCCESSFUL_BUILDS * 100 / KERNEL_NUM))
        print_info "Progress: $KERNEL_NUM/$TOTAL_KERNELS | Success: $SUCCESSFUL_BUILDS | Failed: $FAILED_BUILDS | PTX: $PTX_GENERATED | Rate: $SUCCESS_RATE%"
    fi
done

# Final statistics
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
SUCCESS_RATE=$((SUCCESSFUL_BUILDS * 100 / TOTAL_KERNELS))

echo ""
echo "============================================"
echo "GPU-SPECIFIC COMPILATION COMPLETE"
echo "============================================"
print_success "Cluster: $CURRENT_NODE"
print_success "GPU: $GPU_NAME"
print_success "Total kernels processed: $TOTAL_KERNELS"
print_success "Successful builds: $SUCCESSFUL_BUILDS"
print_success "PTX files generated: $PTX_GENERATED"
print_error "Failed builds: $FAILED_BUILDS"
print_success "Success rate: $SUCCESS_RATE%"
print_info "Total time: $((TOTAL_TIME / 60))m $((TOTAL_TIME % 60))s"

# Generate summary
cat > "$SUMMARY_FILE" << EOF
GPU Cluster-Specific Compilation Summary
=======================================
Generated: $(date)
Cluster: $CURRENT_NODE
GPU: $GPU_NAME
Compute Capability: $COMPUTE_CAP
Architecture: $GPU_ARCH
Dataset: $DATASET
Kernels Processed: $TOTAL_KERNELS of $TOTAL_AVAILABLE available
Total Time: $((TOTAL_TIME / 60))m $((TOTAL_TIME % 60))s

Results:
--------
Successful builds:        $SUCCESSFUL_BUILDS
Failed builds:            $FAILED_BUILDS
PTX files generated:      $PTX_GENERATED
Success rate:             $SUCCESS_RATE%

File Organization:
-----------------
- <kernel_dir>/$CURRENT_NODE/benchmark_*  # GPU-optimized executable
- <kernel_dir>/kernel_*.ptx               # PTX intermediate code

Architecture Details:
--------------------
GPU Architecture: $GPU_ARCH
Compute Capability: sm_$COMPUTE_CAP
Optimization Level: -O3

Usage:
------
Run benchmarks: cd <kernel_dir> && ./$CURRENT_NODE/benchmark_* <matrix_count>
Analyze PTX: examine kernel_*.ptx files for instruction differences
Compare across clusters: run same script on cuda2, cuda3, cuda4, cuda5
EOF

echo ""
print_info "Results summary: $SUMMARY_FILE"

# Show file organization example
if [ $SUCCESSFUL_BUILDS -gt 0 ]; then
    echo ""
    print_info "Example organization:"
    echo "  kernels_src/$DATASET/1/0/$CURRENT_NODE/benchmark_1_0_$GPU_NAME"
    echo "  kernels_src/$DATASET/1/0/kernel_1_0_$GPU_NAME.ptx"
fi

echo ""
print_success "Compilation completed for $GPU_NAME on $CURRENT_NODE!"
print_info "Next: Run this script on other clusters for comparison"
