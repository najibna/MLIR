#pragma once

#include "graph.h"
#include <string>
#include <vector>

// result from one timed run
struct BenchResult {
  double latency_ms = 0.0;
  double throughput = 0.0;
  std::string device;
};

// holds buffers and runs the compiled graph
class InferenceRuntime {
public:
  explicit InferenceRuntime(bool use_cuda = true);
  ~InferenceRuntime();

  // load weights / shapes from the graph
  bool load(const Graph& g);

  // run one forward pass, returns output floats
  std::vector<float> run(const std::vector<float>& input);

  // time a few runs and report avg latency
  BenchResult benchmark(const std::vector<float>& input, int warmup = 5, int iters = 50);

  // dump simple profile stats for each op
  std::string profile(const std::vector<float>& input);

  bool using_cuda() const { return use_cuda_; }

private:
  Graph graph_;
  bool use_cuda_ = false;
  bool loaded_ = false;

  // host side scratch
  std::vector<std::vector<float>> host_bufs_;

  void run_cpu(const std::vector<float>& input, std::vector<float>& output);
  void run_cuda(const std::vector<float>& input, std::vector<float>& output);
};
