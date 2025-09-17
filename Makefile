# TinyServe Optimized vLLM PagedAttention CUDA Kernels Makefile
# Advanced Page Allocation and Memory Management for Large Language Models

# Compiler and flags
NVCC = nvcc
CXX = g++

# CUDA architecture (adjust based on your GPU)
CUDA_ARCH = -arch=sm_80

# Compilation flags
NVCC_FLAGS = $(CUDA_ARCH) -O3 -std=c++17 -Xcompiler -fPIC
NVCC_FLAGS += -use_fast_math -maxrregcount=255
NVCC_FLAGS += -Xptxas -O3,-v
NVCC_FLAGS += -lcudart -lcublas -lcurand

CXX_FLAGS = -O3 -std=c++17 -fPIC
CXX_FLAGS += -I/usr/local/cuda/include
CXX_FLAGS += -L/usr/local/cuda/lib64 -lcudart

# Directories
SRC_DIR = .
BUILD_DIR = build
INCLUDE_DIR = .

# Source files
KERNEL_SRC = vllm_kernels.cu
EXAMPLE_SRC = tinyserve_example.cu
HEADER_SRC = tinyserve_kernels.h

# Object files
KERNEL_OBJ = $(BUILD_DIR)/vllm_kernels.o
EXAMPLE_OBJ = $(BUILD_DIR)/tinyserve_example.o

# Executables
KERNEL_LIB = $(BUILD_DIR)/libtinyserve_kernels.a
EXAMPLE_EXE = $(BUILD_DIR)/tinyserve_example

# Default target
all: $(KERNEL_LIB) $(EXAMPLE_EXE)

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Compile CUDA kernels
$(KERNEL_OBJ): $(KERNEL_SRC) $(HEADER_SRC) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $(KERNEL_SRC) -o $@

# Compile example
$(EXAMPLE_OBJ): $(EXAMPLE_SRC) $(HEADER_SRC) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $(EXAMPLE_SRC) -o $@

# Create static library
$(KERNEL_LIB): $(KERNEL_OBJ)
	ar rcs $@ $<

# Link example executable
$(EXAMPLE_EXE): $(EXAMPLE_OBJ) $(KERNEL_LIB)
	$(NVCC) $(NVCC_FLAGS) -o $@ $< -L$(BUILD_DIR) -ltinyserve_kernels

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)

# Install headers and library
install: $(KERNEL_LIB)
	@echo "Installing TinyServe kernels..."
	@mkdir -p /usr/local/include/tinyserve
	@mkdir -p /usr/local/lib
	cp $(HEADER_SRC) /usr/local/include/tinyserve/
	cp $(KERNEL_LIB) /usr/local/lib/
	@echo "Installation complete!"

# Uninstall
uninstall:
	@echo "Uninstalling TinyServe kernels..."
	rm -f /usr/local/include/tinyserve/$(HEADER_SRC)
	rm -f /usr/local/lib/$(KERNEL_LIB)
	@echo "Uninstallation complete!"

# Run example
run: $(EXAMPLE_EXE)
	@echo "Running TinyServe example..."
	./$(EXAMPLE_EXE)

# Benchmark
benchmark: $(EXAMPLE_EXE)
	@echo "Running TinyServe benchmark..."
	./$(EXAMPLE_EXE) --benchmark

# Test compilation
test: $(KERNEL_LIB)
	@echo "Testing TinyServe kernel compilation..."
	@echo "✓ CUDA kernels compiled successfully"
	@echo "✓ Static library created: $(KERNEL_LIB)"
	@echo "✓ Ready for integration!"

# Check CUDA installation
check-cuda:
	@echo "Checking CUDA installation..."
	@nvcc --version || (echo "❌ CUDA not found. Please install CUDA toolkit." && exit 1)
	@echo "✓ CUDA installation found"
	@nvidia-smi || (echo "❌ NVIDIA driver not found." && exit 1)
	@echo "✓ NVIDIA driver found"

# Performance profiling
profile: $(EXAMPLE_EXE)
	@echo "Running performance profiling..."
	nsys profile --output=profile_report ./$(EXAMPLE_EXE)
	@echo "Profile report generated: profile_report.nsys-rep"

# Memory usage analysis
memcheck: $(EXAMPLE_EXE)
	@echo "Running memory usage analysis..."
	cuda-memcheck ./$(EXAMPLE_EXE)

# Help
help:
	@echo "TinyServe Optimized vLLM PagedAttention CUDA Kernels"
	@echo ""
	@echo "Available targets:"
	@echo "  all          - Build everything (default)"
	@echo "  clean        - Remove build artifacts"
	@echo "  install      - Install headers and library system-wide"
	@echo "  uninstall    - Remove installed files"
	@echo "  run          - Run the example program"
	@echo "  benchmark    - Run performance benchmark"
	@echo "  test         - Test compilation"
	@echo "  check-cuda   - Check CUDA installation"
	@echo "  profile      - Run performance profiling with nsys"
	@echo "  memcheck     - Run memory usage analysis"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "Key Features:"
	@echo "  • FlashAttention integration"
	@echo "  • Advanced memory coalescing"
	@echo "  • Optimized block allocation strategies"
	@echo "  • Enhanced kernel fusion"
	@echo "  • Dynamic workload balancing"
	@echo "  • LRU cache management"
	@echo "  • Intelligent memory compaction"

.PHONY: all clean install uninstall run benchmark test check-cuda profile memcheck help
