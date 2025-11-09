#!/bin/bash

# Directory containing .bin files
BIN_DIR="src/kernel_compliation_on_demand"

# Choose your disassembler here.
# For demonstration, we use objdump. Replace with cuobjdump or llvm-objdump if needed.
DISASSEMBLER="objdump"
DISASSEMBLER_ARGS="-d"

for bin_file in "$BIN_DIR"/*.bin; do
    if [ -f "$bin_file" ]; then
        asm_file="${bin_file%.bin}.asm"
        echo "Disassembling $bin_file -> $asm_file"
        $DISASSEMBLER $DISASSEMBLER_ARGS "$bin_file" > "$asm_file"
    fi
done
