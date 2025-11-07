#!/bin/bash

# Script to compile all kernels in the LS-CAT repository
# Usage: ./compile_all_kernels.sh [options]
# Example: bash \compile_all_kernels.sh --max-sucess 300
# Successful kernels will be stored in bin/compilation_success.txt
# Options:
#   -d, --kernels-dir DIR    Directory containing kernels (default: LS-CAT/data/kernels)
#   -o, --output-dir DIR     Output directory for compiled binaries (default: bin)
#   -f, --flags FLAGS        Additional nvcc flags (default: -gencode=arch=compute_52,code=sm_52)
#   -j, --jobs N             Number of parallel compilation jobs (default: 1)
#   -v, --verbose            Verbose output
#   -h, --help               Show this help message

# Default values
KERNELS_DIR="LS-CAT/data/kernels"
OUTPUT_DIR="bin"
NVCC_FLAGS="-gencode=arch=compute_52,code=sm_52"
JOBS=1
VERBOSE=false
SKIP_EXISTING=false
WORKING_DIR=""
MAX_SUCCESS=0
STOPPED_EARLY=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--kernels-dir)
            KERNELS_DIR="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -f|--flags)
            NVCC_FLAGS="$2"
            shift 2
            ;;
        -j|--jobs)
            JOBS="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -s|--skip-existing)
            SKIP_EXISTING=true
            shift
            ;;
        -W|--working-dir)
            WORKING_DIR="$2"
            shift 2
            ;;
        -M|--max-success)
            MAX_SUCCESS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  -d, --kernels-dir DIR    Directory containing kernels (default: LS-CAT/data/kernels)"
            echo "  -o, --output-dir DIR     Output directory for compiled binaries (default: bin)"
            echo "  -f, --flags FLAGS       Additional nvcc flags (default: -gencode=arch=compute_52,code=sm_52)"
            echo "  -j, --jobs N            Number of parallel compilation jobs (default: 1)"
            echo "  -s, --skip-existing     Skip kernels that already have compiled binaries"
            echo "  -W, --working-dir DIR   Copy kernels that compile successfully to DIR"
            echo "  -M, --max-success N     Stop after N successful kernels (sequential mode only)"
            echo "  -v, --verbose           Verbose output"
            echo "  -h, --help              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Trap to handle Ctrl+C gracefully
trap 'echo ""; echo -e "${YELLOW}Interrupted. Cleaning up...${NC}"; exit 130' INT

# Check if nvcc is available
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}Error: nvcc not found. Please ensure CUDA toolkit is installed and in PATH.${NC}"
    exit 1
fi

# Check if kernels directory exists
if [ ! -d "$KERNELS_DIR" ]; then
    echo -e "${RED}Error: Kernels directory '$KERNELS_DIR' does not exist.${NC}"
    exit 1
fi

if ! [[ "$MAX_SUCCESS" =~ ^[0-9]+$ ]]; then
    echo -e "${RED}Error: --max-success must be a non-negative integer.${NC}"
    exit 1
fi
if [ "$MAX_SUCCESS" -gt 0 ] && [ "$JOBS" -gt 1 ]; then
    echo -e "${YELLOW}Warning:${NC} --max-success requires sequential mode. Forcing --jobs 1."
    JOBS=1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Statistics
TOTAL=0
SUCCESS=0
FAILED=0
SKIPPED=0
FAILED_KERNELS=()
SUCCESS_LIST=""

# Log file
LOG_FILE="$OUTPUT_DIR/compilation_log.txt"
ERROR_LOG="$OUTPUT_DIR/compilation_errors.txt"

# Initialize log files
echo "Compilation started at $(date)" > "$LOG_FILE"
echo "" > "$ERROR_LOG"
SUCCESS_LIST="$OUTPUT_DIR/compilation_success.txt"
echo "" > "$SUCCESS_LIST"

# Function to compile a single kernel
compile_kernel() {
    local kernel_dir="$1"
    local relative_path="${kernel_dir#$KERNELS_DIR/}"
    local output_path="$OUTPUT_DIR/$relative_path"
    local output_binary="$output_path/kernel"
    
    # Skip if already compiled and SKIP_EXISTING is true
    if [ "$SKIP_EXISTING" = true ] && [ -f "$output_binary" ]; then
        echo -e "${YELLOW}Skipping${NC} $relative_path (already compiled)"
        ((SKIPPED++))
        return 0
    fi
    
    # Create output directory
    mkdir -p "$output_path"
    
    # Find all source files in the kernel directory
    local source_files=()
    if [ -f "$kernel_dir/main.cu" ]; then
        source_files+=("$kernel_dir/main.cu")
        
        # Check if main.cu includes other .cu files
        # If it does, we should NOT pass those .cu files separately to nvcc
        # (they'll be processed by the preprocessor when included)
        local included_cu_files=()
        if [ -f "$kernel_dir/main.cu" ]; then
            # Find all .cu files that are included in main.cu
            # Look for lines with #include and .cu
            while IFS= read -r line; do
                # Extract filename from #include "filename.cu" or #include 'filename.cu'
                local included_file=""
                # Try double quotes first
                if [[ "$line" =~ include[[:space:]]+\"([^\"]+\.cu)\" ]]; then
                    included_file="${BASH_REMATCH[1]}"
                # Try single quotes
                elif [[ "$line" =~ include[[:space:]]+\'([^\']+\.cu)\' ]]; then
                    included_file="${BASH_REMATCH[1]}"
                fi
                # Check if file exists in the kernel directory
                if [ -n "$included_file" ] && [ -f "$kernel_dir/$included_file" ]; then
                    included_cu_files+=("$included_file")
                fi
            done < <(grep -i 'include.*\.cu' "$kernel_dir/main.cu" 2>/dev/null || true)
        fi
        
        # Find other .cu files that are NOT included in main.cu
        while IFS= read -r -d '' file; do
            if [ "$file" != "$kernel_dir/main.cu" ]; then
                local filename=$(basename "$file")
                local is_included=false
                for included in "${included_cu_files[@]}"; do
                    if [ "$filename" == "$included" ]; then
                        is_included=true
                        break
                    fi
                done
                # Only add .cu files that are NOT included
                if [ "$is_included" = false ]; then
                    source_files+=("$file")
                fi
            fi
        done < <(find "$kernel_dir" -maxdepth 1 -name "*.cu" -type f -print0)
        
        # Always include .c and .cpp files separately (they're usually not included)
        while IFS= read -r -d '' file; do
            source_files+=("$file")
        done < <(find "$kernel_dir" -maxdepth 1 -name "*.c" -type f -print0)
        
        while IFS= read -r -d '' file; do
            source_files+=("$file")
        done < <(find "$kernel_dir" -maxdepth 1 -name "*.cpp" -type f -print0)
    else
        if [ "$VERBOSE" = true ]; then
            echo -e "${YELLOW}Warning:${NC} No main.cu found in $relative_path"
        fi
        return 1
    fi
    
    # Compile
    if [ "$VERBOSE" = true ]; then
        echo -e "${BLUE}Compiling${NC} $relative_path..."
    fi
    
    # Convert source files to relative paths from kernel directory
    local relative_sources=()
    for src_file in "${source_files[@]}"; do
        local filename=$(basename "$src_file")
        relative_sources+=("$filename")
    done
    
    # Convert output path to absolute path for use when cd'd into kernel directory
    local output_binary_abs=$(cd "$(dirname "$output_binary")" && pwd)/$(basename "$output_binary")
    
    # Add kernel directory to include path to ensure headers can be found
    # Compile from kernel directory to ensure relative includes work
    local compile_output
    local compile_exit_code
    
    if compile_output=$(cd "$kernel_dir" && nvcc -I"$kernel_dir" "${relative_sources[@]}" -o "$output_binary_abs" $NVCC_FLAGS 2>&1); then
        compile_exit_code=0
    else
        compile_exit_code=$?
    fi
    
    if [ $compile_exit_code -eq 0 ]; then
        echo -e "${GREEN}Success${NC} $relative_path" | tee -a "$LOG_FILE"
        echo "$relative_path" >> "$SUCCESS_LIST"
        ((SUCCESS++))

        if [ -n "$WORKING_DIR" ]; then
            local working_path="$WORKING_DIR/$relative_path"
            mkdir -p "$(dirname "$working_path")"
            if command -v rsync &> /dev/null; then
                if ! rsync -a --delete "$kernel_dir/" "$working_path/"; then
                    echo -e "${YELLOW}Warning:${NC} Failed to copy $relative_path to working directory" | tee -a "$LOG_FILE"
                fi
            else
                if ! cp -a "$kernel_dir/." "$working_path/"; then
                    echo -e "${YELLOW}Warning:${NC} Failed to copy $relative_path to working directory" | tee -a "$LOG_FILE"
                fi
            fi
        fi

        return 0
    else
        echo -e "${RED}Failed${NC}  $relative_path" | tee -a "$LOG_FILE"
        echo "=== $relative_path ===" >> "$ERROR_LOG"
        echo "$compile_output" >> "$ERROR_LOG"
        echo "" >> "$ERROR_LOG"
        FAILED_KERNELS+=("$relative_path")
        ((FAILED++))
        return 1
    fi
}

# Export function for parallel execution
export -f compile_kernel
export KERNELS_DIR OUTPUT_DIR NVCC_FLAGS VERBOSE SKIP_EXISTING WORKING_DIR SUCCESS_LIST MAX_SUCCESS
export RED GREEN YELLOW BLUE NC

# Find all kernel directories (directories containing main.cu)
echo -e "${BLUE}Scanning for kernels in $KERNELS_DIR...${NC}"
kernel_dirs=()
while IFS= read -r dir; do
    if [ -n "$dir" ]; then
        kernel_dirs+=("$dir")
        ((TOTAL++))
    fi
done < <(find "$KERNELS_DIR" -type f -name "main.cu" -exec dirname {} \; | sort -u)

echo -e "${BLUE}Found $TOTAL kernels${NC}"
if [ -n "$WORKING_DIR" ]; then
    mkdir -p "$WORKING_DIR"
    echo -e "${BLUE}Working kernels will be copied to:${NC} $WORKING_DIR"
fi
echo -e "${BLUE}Starting compilation...${NC}"
echo ""

# Compile kernels
if [ "$JOBS" -gt 1 ]; then
    # Parallel compilation
    printf '%s\0' "${kernel_dirs[@]}" | xargs -0 -n 1 -P "$JOBS" -I {} bash -c 'compile_kernel "$@"' _ {}
else
    # Sequential compilation
    for kernel_dir in "${kernel_dirs[@]}"; do
        compile_kernel "$kernel_dir"
        if [ "$MAX_SUCCESS" -gt 0 ] && [ "$SUCCESS" -ge "$MAX_SUCCESS" ]; then
            STOPPED_EARLY=true
            echo -e "${YELLOW}Reached maximum successful kernels (${MAX_SUCCESS}). Stopping early.${NC}" | tee -a "$LOG_FILE"
            break
        fi
    done
fi

# Print summary
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Compilation Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Total kernels:    ${BLUE}$TOTAL${NC}"
echo -e "Successful:       ${GREEN}$SUCCESS${NC}"
echo -e "Failed:           ${RED}$FAILED${NC}"
if [ "$SKIP_EXISTING" = true ]; then
    echo -e "Skipped:          ${YELLOW}$SKIPPED${NC}"
fi
if [ "$STOPPED_EARLY" = true ]; then
    echo -e "Stopped early:    ${YELLOW}Yes (max success ${MAX_SUCCESS})${NC}"
fi
if [ -s "$SUCCESS_LIST" ]; then
    echo ""
    echo -e "${GREEN}Working kernels:${NC}"
    while IFS= read -r kernel; do
        [ -n "$kernel" ] && echo "  $kernel"
    done < "$SUCCESS_LIST"
    if [ -n "$WORKING_DIR" ]; then
        echo ""
        echo -e "${GREEN}Copied working kernels to:${NC} $WORKING_DIR"
    fi
fi
echo ""
echo -e "Log file:         $LOG_FILE"
echo -e "Error log:        $ERROR_LOG"
echo -e "Success list:     $SUCCESS_LIST"

if [ $FAILED -gt 0 ]; then
    echo ""
    echo -e "${RED}Failed kernels (first 20):${NC}"
    for i in "${!FAILED_KERNELS[@]}"; do
        if [ $i -lt 20 ]; then
            echo "  ${FAILED_KERNELS[$i]}"
        fi
    done
    if [ ${#FAILED_KERNELS[@]} -gt 20 ]; then
        echo "  ... and $(( ${#FAILED_KERNELS[@]} - 20 )) more"
    fi
    echo ""
    echo -e "${YELLOW}Check $ERROR_LOG for detailed error messages${NC}"
    exit 1
else
    exit 0
fi
