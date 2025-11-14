#!/bin/bash

# Script to disassemble all kernel binaries using cuobjdump and nvdisasm

# Define paths
SOURCE_DIR="../kernel_compliation_on_demand/bin"
OUTPUT_DIR="./output"

# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory $SOURCE_DIR does not exist"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Starting disassembly process..."
echo "Source: $SOURCE_DIR"
echo "Output: $OUTPUT_DIR"
echo ""

# Counter for progress
total=0
processed=0

# Count total kernels
total=$(find "$SOURCE_DIR" -type f -name "kernel" | wc -l)
echo "Found $total kernel binaries to process"
echo ""

# Find all kernel files and process them
find "$SOURCE_DIR" -type f -name "kernel" | while read kernel_path; do
    # Extract the relative path structure (e.g., "1/0" from "bin/1/0/kernel")
    relative_path=$(echo "$kernel_path" | sed "s|$SOURCE_DIR/||" | sed 's|/kernel$||')
    
    # Create corresponding output directory
    output_subdir="$OUTPUT_DIR/$relative_path"
    mkdir -p "$output_subdir"
    
    # Define output file path
    output_file="$output_subdir/nvdisasm.txt"
    
    # Convert kernel_path to absolute path
    kernel_abs_path=$(realpath "$kernel_path")
    
    echo "Processing: $relative_path"
    
    # Change to output subdirectory (cuobjdump creates files in current directory)
    cd "$output_subdir"
    
    # Step 1: Extract ELF from the kernel binary
    # cuobjdump creates files with names like "kernel.sm_60.cubin.elf" in current directory
    cuobjdump --extract-elf all "$kernel_abs_path" 2>&1
    
    if [ $? -eq 0 ]; then
        # Find the extracted cubin file(s) - cuobjdump extracts .cubin files
        cubin_file=$(find . -maxdepth 1 -name "*.cubin" | head -n 1)
        
        if [ -n "$cubin_file" ]; then
            # Step 2: Disassemble the cubin file
            nvdisasm "$cubin_file" > "nvdisasm.txt" 2>&1
            
            if [ $? -eq 0 ]; then
                echo "  ✓ Successfully disassembled to $output_file"
            else
                echo "  ✗ Error: nvdisasm failed for $relative_path"
                echo "  See $output_file for error details"
            fi
            
            # Clean up all cubin files
            rm -f *.cubin
        else
            echo "  ✗ Error: No cubin file extracted for $relative_path"
            echo "No cubin file extracted" > "nvdisasm.txt"
        fi
    else
        echo "  ✗ Error: cuobjdump failed to extract ELF for $relative_path"
        echo "cuobjdump error" > "nvdisasm.txt"
    fi
    
    # Return to script directory
    cd - > /dev/null
    
    # Increment counter
    ((processed++))
    echo "  Progress: $processed/$total"
    echo ""
done

echo "Disassembly process completed!"
echo "Output directory: $OUTPUT_DIR"