#include "graph.h"
#include "passes.h"
#include "runtime.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// build a tiny mlp graph by hand so we can demo without python
Graph make_demo_graph() {
  Graph g;
  int x = g.add_node(OpKind::Input, "x", {}, {1, 4});

  int w1 = g.add_node(OpKind::Constant, "w1", {}, {4, 8});
  g.get(w1)->data = {
      0.2f, -0.1f, 0.4f, 0.3f, -0.2f, 0.1f, 0.05f, -0.3f,
      0.1f, 0.2f, -0.15f, 0.25f, 0.3f, -0.05f, 0.1f, 0.2f,
      -0.2f, 0.1f, 0.3f, -0.1f, 0.2f, 0.15f, -0.25f, 0.05f,
      0.05f, -0.2f, 0.1f, 0.3f, -0.15f, 0.2f, 0.1f, -0.05f};

  int h = g.add_node(OpKind::MatMul, "fc1", {x, w1}, {1, 8});
  int h_relu = g.add_node(OpKind::Relu, "relu1", {h}, {1, 8});

  int w2 = g.add_node(OpKind::Constant, "w2", {}, {8, 3});
  g.get(w2)->data = {
      0.1f, -0.2f, 0.15f,
      -0.1f, 0.25f, 0.05f,
      0.2f, 0.1f, -0.15f,
      0.05f, -0.1f, 0.2f,
      -0.15f, 0.2f, 0.1f,
      0.1f, 0.05f, -0.2f,
      0.2f, -0.05f, 0.1f,
      -0.1f, 0.15f, 0.05f};

  int logits = g.add_node(OpKind::MatMul, "fc2", {h_relu, w2}, {1, 3});
  int probs = g.add_node(OpKind::Softmax, "probs", {logits}, {1, 3});
  g.add_node(OpKind::Output, "y", {probs}, {1, 3});
  return g;
}

static void write_file(const std::string& path, const std::string& text) {
  std::ofstream out(path);
  out << text;
}

static std::string read_file(const std::string& path) {
  std::ifstream in(path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

static void usage() {
  std::cout
      << "usage:\n"
      << "  mlir_nn compile [--quantize] [--emit path]\n"
      << "  mlir_nn run [--cuda] [--input vals...]\n"
      << "  mlir_nn bench [--cuda] [--iters n]\n"
      << "  mlir_nn profile [--cuda]\n";
}

int main(int argc, char** argv) {
  if (argc < 2) {
    usage();
    return 1;
  }

  std::string cmd = argv[1];
  bool do_quantize = true;
  bool use_cuda = true;
  std::string emit_path = "build/graph.mlir";
  int iters = 50;
  std::vector<float> input = {0.5f, -0.25f, 0.75f, 0.1f};

  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--quantize") do_quantize = true;
    else if (a == "--no-quantize") do_quantize = false;
    else if (a == "--cuda") use_cuda = true;
    else if (a == "--cpu") use_cuda = false;
    else if (a == "--emit" && i + 1 < argc) emit_path = argv[++i];
    else if (a == "--iters" && i + 1 < argc) iters = std::stoi(argv[++i]);
    else if (a == "--input") {
      input.clear();
      while (i + 1 < argc && argv[i + 1][0] != '-') {
        input.push_back(std::stof(argv[++i]));
      }
    }
  }

  Graph g = make_demo_graph();

  if (cmd == "compile") {
    std::cout << "before passes:\n" << g.to_mlir() << "\n";
    run_compiler_pipeline(g, do_quantize);
    auto plan = schedule_graph(g);
    std::cout << "after passes:\n" << g.to_mlir() << "\n";
    std::cout << "schedule:";
    for (int id : plan.order) std::cout << " %" << id;
    std::cout << "\nestimated_ms: " << plan.estimated_ms << "\n";
    write_file(emit_path, g.to_mlir());
    std::cout << "wrote " << emit_path << "\n";
    return 0;
  }

  run_compiler_pipeline(g, do_quantize);

  if (cmd == "run") {
    InferenceRuntime rt(use_cuda);
    rt.load(g);
    auto out = rt.run(input);
    std::cout << "device: " << (rt.using_cuda() ? "cuda" : "cpu") << "\n";
    std::cout << "output:";
    for (float v : out) std::cout << " " << v;
    std::cout << "\n";
    return 0;
  }

  if (cmd == "bench") {
    InferenceRuntime rt(use_cuda);
    rt.load(g);
    auto br = rt.benchmark(input, 5, iters);
    std::cout << "device: " << br.device << "\n";
    std::cout << "latency_ms: " << br.latency_ms << "\n";
    std::cout << "throughput_qps: " << br.throughput << "\n";
    return 0;
  }

  if (cmd == "profile") {
    InferenceRuntime rt(use_cuda);
    rt.load(g);
    std::cout << rt.profile(input);
    return 0;
  }

  // optional: load an mlir file if someone passes one
  if (cmd == "from-mlir" && argc >= 3) {
    Graph loaded = Graph::from_mlir(read_file(argv[2]));
    run_compiler_pipeline(loaded, do_quantize);
    std::cout << loaded.to_mlir();
    return 0;
  }

  usage();
  return 1;
}
