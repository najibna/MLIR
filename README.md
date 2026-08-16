# MLIR Neural Network Compiler & CUDA Inference Runtime

**End-to-end inference compiler stack** in C++, Python, and CUDA: graph IR → dialect lowering → operator fusion → int8 quantization → cost-model scheduling → GPU/CPU execution → latency profiling.

Built to demonstrate the same systems problems that show up in production AI compilers—graph rewriting, kernel fusion, quantized execution, and measured performance—not just model training wrappers.

```
PyTorch MLP ──► nn dialect (MLIR-style IR)
                      │
                      ▼
              ┌───────────────┐
              │  Lower        │
              │  Fuse         │  compiler passes
              │  Quantize     │
              │  Schedule     │
              └───────┬───────┘
                      ▼
              CUDA / CPU runtime ──► latency + throughput reports
```

---

## Why this project

Modern inference is a **compiler problem**: models arrive as graphs, performance comes from how you lower, fuse, and schedule them onto hardware. This repo implements that loop in a small, readable codebase so the design is inspectable end-to-end.

| Compiler concern | Implementation |
|---|---|
| Intermediate representation | Custom `nn` dialect with SSA-style `%` values, tensor types, and MLIR-like textual form |
| Lowering | High-level graph → runtime-ready ops (`src/passes.cpp`) |
| Peephole / fusion | `matmul` + `relu` → `fused_matmul_relu` with single-user safety checks |
| Quantization | Absmax int8 scales + zero-points on weight constants |
| Scheduling | Topological order + element-count cost model |
| Codegen / kernels | CUDA kernels for matmul, fused matmul-relu, relu, add (`cuda/kernels.cu`) |
| Runtime | Host orchestration, device fallback, buffer management (`src/runtime.cpp`) |
| Measurement | Warmup + timed iters → latency (ms) and throughput (QPS) |
| Frontend | PyTorch export path into the dialect (`python/export_graph.py`) |

---

## Architecture

### 1. Graph IR (`include/graph.h`, `src/graph.cpp`)

- Typed ops: `input`, `constant`, `matmul`, `add`, `relu`, `softmax`, `quantize`, `dequantize`, `fused_matmul_relu`, `output`
- Shape-carrying nodes and weight payloads
- Round-trip textual dump/parse in an MLIR-inspired module/func layout

### 2. Pass pipeline (`include/passes.h`, `src/passes.cpp`)

```
Graph
  → lower_graph()          # normalize names / shapes for the runtime
  → fuse_operators()       # pattern match matmul→relu, rewrite users
  → quantize_tensors()     # int8 scale on constants
  → schedule_graph()       # topo order + estimated ms
```

Fusion only fires when the producer has a single consumer—same class of legality check used in real graph optimizers before rewriting.

### 3. CUDA inference runtime (`cuda/kernels.cu`, `src/runtime.cpp`)

- Device kernels: blocked matmul, epilogue-fused matmul+ReLU, elementwise relu/add
- Host runtime selects CUDA when available, otherwise CPU reference kernels
- Shared schedule drives execution order on both backends
- Softmax kept on host (typical for tiny classifier heads)

### 4. Python tooling (PyTorch bridge)

| Script | Role |
|---|---|
| `python/export_graph.py` | Export a TinyMLP to dialect text + weight dump |
| `python/compile.py` | Mirror fusion/quantize stages from Python |
| `python/benchmark.py` | Compare PyTorch eager latency vs native runtime |
| `python/profile_runtime.py` | Dump schedule + measured latency |

---

## Example: dialect before / after fusion

**Before**

```mlir
%2 = nn.matmul (%0, %1) : tensor<1x8xf32>  // fc1
%3 = nn.relu (%2) : tensor<1x8xf32>        // relu1
%5 = nn.matmul (%3, %4) : tensor<1x3xf32>  // fc2
```

**After compiler passes**

```mlir
%2 = nn.fused_matmul_relu (%0, %1) : tensor<1x8xf32>  // fused_fc1
%5 = nn.matmul (%2, %4) : tensor<1x3xf32>             // fc2
```

Weights are rewritten with int8 scales (`*_q8`); the scheduler emits an execution order and estimated cost before the runtime measures wall-clock latency.

---

## Skills this demonstrates (AI compiler lens)

- **IR design** — dialects, tensor types, SSA-like value uses
- **Graph optimization** — legality-aware fusion, quantization passes
- **Lowering & scheduling** — multi-pass pipeline into a cost-aware order
- **GPU systems** — CUDA kernels, host/device sync, CPU fallback path
- **Performance culture** — benchmark harnesses, profiling dumps, QPS reporting
- **ML ↔ systems bridge** — PyTorch frontends into a custom compiler stack
- **Engineering hygiene** — CMake + Makefile dual build, CLI surface, smoke tests

Stack: **C++17 · CUDA · Python · PyTorch · CMake**

---

## Quick start

### Build

```bash
# CPU reference path
make cpu

# GPU path (requires nvcc)
make cuda

# or CMake
cmake -S . -B build -DMLIR_NN_ENABLE_CUDA=ON
cmake --build build
```

### Compile → run → measure

```bash
./build/mlir_nn compile --emit build/graph.mlir
./build/mlir_nn run --cpu          # or --cuda
./build/mlir_nn bench --cpu --iters 50
./build/mlir_nn profile --cpu
```

### PyTorch export + end-to-end demo

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python3 python/export_graph.py --out build/model.mlir
python3 examples/run_demo.py
```

---

## Repository layout

```
include/     Graph IR, pass API, runtime, kernel declarations
src/         Compiler driver, passes, CPU kernels, inference runtime, CLI
cuda/        CUDA kernels (matmul, fused matmul-relu, relu, add)
python/      PyTorch export, compile helpers, benchmark, profile
examples/    Full pipeline demo
tests/       Fusion smoke tests
```

---

## Design notes

This is a **focused compiler systems project**: a complete, inspectable path from graph IR to fused CUDA execution. It intentionally uses a compact dialect rather than linking a full LLVM/MLIR tree, so every pass, rewrite, and kernel stays readable in one sitting—useful for discussing tradeoffs in fusion legality, quantization error, and host/device scheduling in interviews.

The same ideas scale to larger stacks (dialect hierarchies, memory planning, autotuning, multi-stream execution); this repo is the vertical slice that proves the pipeline.
