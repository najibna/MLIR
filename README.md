# mlir nn compiler + cuda runtime

c++ / python / cuda toolkit for compiling neural nets down to a small
inference runtime. the front end speaks a tiny `nn` dialect that looks
like mlir text. passes lower the graph, fuse ops, quantize weights, then
a cpu or cuda runtime schedules and runs the model.

## what it does

- lower high level graph ops into a runtime friendly form
- fuse `matmul` + `relu` into one kernel
- quantize constant tensors with a simple int8 scale
- schedule ops with a cheap cost model
- run inference on cpu, or cuda when available
- profile and benchmark latency from both c++ and python
- export a pytorch mlp into the mlir-ish text format

## layout

```
include/          headers for graph, passes, runtime
src/              compiler + cpu path + cli
cuda/kernels.cu   matmul / fused matmul-relu / relu / add
python/           pytorch export, compile, bench, profile
examples/         end to end demo script
```

## build

cpu only (no gpu needed):

```bash
make cpu
```

with cuda (needs `nvcc`):

```bash
make cuda
```

or cmake:

```bash
cmake -S . -B build -DMLIR_NN_ENABLE_CUDA=ON
cmake --build build
```

## python setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## quick run

```bash
# c++ cli
./build/mlir_nn compile --emit build/graph.mlir
./build/mlir_nn run --cpu
./build/mlir_nn bench --cpu --iters 50
./build/mlir_nn profile --cpu

# pytorch export + passes + bench
python3 python/export_graph.py --out build/model.mlir
python3 python/compile.py --input build/model.mlir
python3 python/benchmark.py --iters 30
python3 python/profile_runtime.py --python-only

# full demo
python3 examples/run_demo.py
```

## sample mlir dialect

```
module {
  func @main() {
    %0 = nn.input : tensor<1x4xf32>  // x
    %1 = nn.constant : tensor<4x8xf32>  // w1
    %2 = nn.matmul (%0, %1) : tensor<1x8xf32>  // fc1
    %3 = nn.relu (%2) : tensor<1x8xf32>  // relu1
    %4 = nn.constant : tensor<8x3xf32>  // w2
    %5 = nn.matmul (%3, %4) : tensor<1x3xf32>  // fc2
    %6 = nn.softmax (%5) : tensor<1x3xf32>  // probs
    %7 = nn.output (%6) : tensor<1x3xf32>  // y
    return
  }
}
```

after fusion the first matmul/relu pair becomes `nn.fused_matmul_relu`.

## notes

the dialect and passes are kept small on purpose so the path from graph
to fused kernels stays easy to follow. it is not a replacement for full
mlir or a production inference engine.
