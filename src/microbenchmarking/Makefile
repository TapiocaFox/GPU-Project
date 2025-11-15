# Makefile for CUDA Microbenchmarking Tool

# Compiler
NVCC = nvcc

# Target executable name
TARGET = cuda_microbenchmark

# Source file
SRC = cuda_microbenchmark.cu

# Compiler flags
# -O3: Maximum optimization
# -arch: GPU architecture (adjust for your GPU)
# -lineinfo: Include line number information for profiling
# NVCCFLAGS = -O3 -arch=sm_70 -lineinfo

# Common architectures (uncomment the one for your GPU):
# NVCCFLAGS = -O3 -arch=sm_60 -lineinfo  # Pascal (GTX 1080, P100)
# NVCCFLAGS = -O3 -arch=sm_70 -lineinfo  # Volta (V100)
# NVCCFLAGS = -O3 -arch=sm_75 -lineinfo  # Turing (RTX 2080, T4)
# NVCCFLAGS = -O3 -arch=sm_80 -lineinfo  # Ampere (A100)
# NVCCFLAGS = -O3 -arch=sm_86 -lineinfo  # Ampere (RTX 3090)
# NVCCFLAGS = -O3 -arch=sm_89 -lineinfo  # Ada Lovelace (RTX 4090)

# NVCCFLAGS = -O3 -arch=sm_70 -lineinfo -std=c++11  # Titan V (add -std=c++11 for C++ features)
NVCCFLAGS = -O0 -arch=sm_75 -lineinfo -std=c++11

# Default target
all: $(TARGET)

# Compile the CUDA program
$(TARGET): $(SRC)
	@echo "Compiling $(SRC)..."
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $(TARGET)
	@echo "Build complete: $(TARGET)"

# Run the benchmark
run: $(TARGET)
	@echo "Running microbenchmarks..."
	./$(TARGET)

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -f $(TARGET)
	rm -f *.o
	@echo "Clean complete"

# Help target
help:
	@echo "Available targets:"
	@echo "  make          - Compile the program (default)"
	@echo "  make run      - Compile and run the program"
	@echo "  make clean    - Remove compiled files"
	@echo "  make help     - Show this help message"
	@echo ""
	@echo "To change GPU architecture, edit NVCCFLAGS in Makefile"

.PHONY: all run clean help
