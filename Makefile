# GPU Database Metal Benchmark Makefile
# Builds C++ project using metal-cpp library

# Project Configuration
PROJECT_NAME = MetalGenericDBEngine
SOURCE_DIR = src
KERNEL_DIR = kernels
METAL_CPP_DIR = third_party/metal-cpp
DATA_DIR = data
BUILD_DIR = build
BIN_DIR = $(BUILD_DIR)/bin
OBJ_DIR = $(BUILD_DIR)/obj

# Compiler and flags
# Use Apple toolchain (avoids Homebrew libc++/SDK mismatches)
# Note: GNU Make defines a default `CXX = c++`, so we must override explicitly.
# Use the xcrun wrapper so the correct SDK/sysroot is selected.
CXX = xcrun -sdk macosx clang++
# macOS deployment target
MACOSX_MIN ?= 14.0
# Use -O2 instead of -O3 to avoid Apple clang optimizer bug causing segfault
CXXFLAGS = -std=c++20 -Wall -Wextra -O2 -mmacosx-version-min=$(MACOSX_MIN)

# Include paths
INCLUDES = -I$(METAL_CPP_DIR) \
           -Ithird_party \
           -I$(SOURCE_DIR)/Engine/Executor \
           -I$(SOURCE_DIR)/Engine/Planner \
           -I$(SOURCE_DIR)/Engine/Storage \
           -I$(SOURCE_DIR)/Engine/Storage/CPU \
           -I$(SOURCE_DIR)/Engine/Storage/GPU \
           -I$(SOURCE_DIR)/Engine/Utils


# Framework flags for macOS
FRAMEWORKS = -framework Metal -framework Foundation -framework QuartzCore

# Source files
SOURCES = $(shell find $(SOURCE_DIR) -name '*.cpp')
OBJECTS = $(SOURCES:$(SOURCE_DIR)/%.cpp=$(OBJ_DIR)/%.o)
KERNELS = $(wildcard $(KERNEL_DIR)/*.metal)

# Target executable
TARGET = $(BIN_DIR)/$(PROJECT_NAME)

# Metal compiler tools
METAL = xcrun -sdk macosx metal -std=macos-metal2.4
METALLIB = xcrun -sdk macosx metallib
KERNEL_AIR = $(BUILD_DIR)/kernels.air
KERNEL_METALLIB = $(BUILD_DIR)/kernels.metallib
BUILD_SENTINEL = $(BUILD_DIR)/.dir

# Auto-detect Metal shader compiler (requires full Xcode, not just CommandLineTools)
HAS_METAL := $(shell xcrun -sdk macosx -f metal 2>/dev/null && echo 1 || echo 0)

# Default target: always build C++ binary; optionally pre-compile metallib
.PHONY: all
ifeq ($(HAS_METAL),1)
all: $(TARGET) $(KERNEL_METALLIB)
	@echo "Built with pre-compiled Metal library (GPU)"
else
all: $(TARGET)
	@echo "Built without metal CLI — shaders will be compiled at runtime from kernels/Operators.metal"
endif

# Create target executable
$(TARGET): $(OBJECTS) | $(BIN_DIR)
	@echo "Linking $(PROJECT_NAME)..."
	$(CXX) $(CXXFLAGS) $(OBJECTS) $(FRAMEWORKS) -o $@
	@echo "Build complete: $@"

# Build Metal kernels (only when metal CLI is available)
$(KERNEL_AIR): $(KERNELS) | $(BUILD_SENTINEL)
	@echo "Compiling Metal kernels (.air)..."
	$(METAL) -c $(KERNEL_DIR)/Operators.metal -o $(KERNEL_AIR)

$(KERNEL_METALLIB): $(KERNEL_AIR)
	@echo "Linking Metal library (.metallib)..."
	$(METALLIB) $(KERNEL_AIR) -o $(KERNEL_METALLIB)
	@cp $(KERNEL_METALLIB) default.metallib

# Compile source files
$(OBJ_DIR)/%.o: $(SOURCE_DIR)/%.cpp | $(OBJ_DIR)
	@echo "Compiling $<..."
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Create directories
$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

$(OBJ_DIR):
	@mkdir -p $(OBJ_DIR)

$(BUILD_SENTINEL):
	@mkdir -p $(BUILD_DIR)
	@touch $(BUILD_SENTINEL)

# Clean build artifacts
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(BUILD_DIR)
	@echo "Clean complete"

# Run the program
.PHONY: run
run: $(TARGET)
	@echo "Running $(PROJECT_NAME)..."
	@$(TARGET)

# Build target
.PHONY: build
build: all

# Explicit metallib target (for users with full Xcode)
.PHONY: metallib
metallib: $(KERNEL_METALLIB)
