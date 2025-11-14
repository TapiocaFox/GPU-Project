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
    
    # Temporary ELF file
    temp_elf="$output_subdir/temp.elf"
    
    echo "Processing: $relative_path"
    
    # Step 1: Extract ELF from the kernel binary
    cuobjdump --extract-elf all "$kernel_path" -o "$temp_elf" 2>&1
    
    if [ $? -eq 0 ] && [ -f "$temp_elf" ]; then
        # Step 2: Disassemble the ELF file
        nvdisasm "$temp_elf" > "$output_file" 2>&1
        
        if [ $? -eq 0 ]; then
            echo "  ✓ Successfully disassembled to $output_file"
            # Clean up temporary ELF file
            rm -f "$temp_elf"
        else
            echo "  ✗ Error: nvdisasm failed for $relative_path"
            echo "  See $output_file for error details"
        fi
    else
        echo "  ✗ Error: cuobjdump failed to extract ELF for $relative_path"
        echo "cuobjdump error" > "$output_file"
    fi
    
    # Increment counter
    ((processed++))
    echo "  Progress: $processed/$total"
    echo ""
done

echo "Disassembly process completed!"
echo "Output directory: $OUTPUT_DIR"