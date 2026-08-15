# simple makefile so you can build without remembering cmake flags
CXX ?= c++
CXXFLAGS ?= -std=c++17 -O2 -Iinclude
BUILD = build
BIN = $(BUILD)/mlir_nn

SRCS = src/graph.cpp src/passes.cpp src/cpu_kernels.cpp src/runtime.cpp src/main.cpp

.PHONY: all clean cuda cpu test demo

all: cpu

$(BUILD):
	mkdir -p $(BUILD)

cpu: $(BUILD)
	# cpu only binary, works without an nvidia toolchain
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(BIN)
	@echo "built $(BIN)"

cuda: $(BUILD)
	# needs nvcc on path
	nvcc -std=c++17 -O2 -Iinclude -DMLIR_HAS_CUDA=1 \
		$(SRCS) cuda/kernels.cu -o $(BIN) -lcudart
	@echo "built $(BIN) with cuda"

clean:
	rm -rf $(BUILD)

demo:
	python3 examples/run_demo.py

test: cpu
	$(BIN) compile --emit $(BUILD)/test.mlir
	$(BIN) run --cpu
	$(BIN) bench --cpu --iters 20
	$(BIN) profile --cpu
