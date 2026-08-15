#include "runtime.h"
#include "kernels.h"
#include "passes.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <sstream>
#include <unordered_map>

InferenceRuntime::InferenceRuntime(bool use_cuda) {
  use_cuda_ = use_cuda && cuda_available();
}

InferenceRuntime::~InferenceRuntime() = default;

bool InferenceRuntime::load(const Graph& g) {
  graph_ = g;
  host_bufs_.clear();
  // store one buffer per node id (ids can have gaps after fusion)
  int max_id = 0;
  for (const auto& n : graph_.nodes()) max_id = std::max(max_id, n.id);
  host_bufs_.assign(static_cast<size_t>(max_id + 1), {});

  for (const auto& n : graph_.nodes()) {
    long count = 1;
    for (int d : n.shape) count *= std::max(d, 1);
    host_bufs_[n.id].assign(static_cast<size_t>(count), 0.0f);
    if (n.kind == OpKind::Constant && !n.data.empty()) {
      host_bufs_[n.id] = n.data;
    }
  }
  loaded_ = true;
  return true;
}

static int elems(const Node& n) {
  int e = 1;
  for (int d : n.shape) e *= std::max(d, 1);
  return e;
}

static void exec_matmul(bool use_cuda, bool fused,
                        const std::vector<float>& a,
                        const std::vector<float>& b,
                        std::vector<float>& buf,
                        const Node& n) {
  int m = n.shape.size() > 0 ? n.shape[0] : 1;
  int out_n = n.shape.size() > 1 ? n.shape[1] : 1;
  int k = static_cast<int>(a.size() / std::max(m, 1));
  buf.assign(static_cast<size_t>(m * out_n), 0.0f);
  if (use_cuda) {
    if (fused) cuda_fused_matmul_relu(a.data(), b.data(), buf.data(), m, out_n, k);
    else cuda_matmul(a.data(), b.data(), buf.data(), m, out_n, k);
  } else {
    if (fused) cpu_fused_matmul_relu(a.data(), b.data(), buf.data(), m, out_n, k);
    else cpu_matmul(a.data(), b.data(), buf.data(), m, out_n, k);
  }
}

void InferenceRuntime::run_cpu(const std::vector<float>& input,
                               std::vector<float>& output) {
  run_cuda(input, output); // shared path, use_cuda_ gates kernels
}

void InferenceRuntime::run_cuda(const std::vector<float>& input,
                                std::vector<float>& output) {
  auto plan = schedule_graph(graph_);
  bool gpu = use_cuda_;

  for (int id : plan.order) {
    Node* n = graph_.get(id);
    if (!n) continue;
    auto& buf = host_bufs_[id];

    switch (n->kind) {
      case OpKind::Input:
        buf = input;
        if (static_cast<int>(buf.size()) < elems(*n))
          buf.resize(elems(*n), 0.0f);
        break;
      case OpKind::Constant:
        break;
      case OpKind::MatMul: {
        if (n->inputs.size() < 2) break;
        exec_matmul(gpu, false, host_bufs_[n->inputs[0]],
                    host_bufs_[n->inputs[1]], buf, *n);
        break;
      }
      case OpKind::FusedMatMulRelu: {
        if (n->inputs.size() < 2) break;
        exec_matmul(gpu, true, host_bufs_[n->inputs[0]],
                    host_bufs_[n->inputs[1]], buf, *n);
        break;
      }
      case OpKind::Add: {
        if (n->inputs.size() < 2) break;
        const auto& a = host_bufs_[n->inputs[0]];
        const auto& b = host_bufs_[n->inputs[1]];
        buf.resize(a.size());
        if (gpu) cuda_add(a.data(), b.data(), buf.data(), static_cast<int>(a.size()));
        else cpu_add(a.data(), b.data(), buf.data(), static_cast<int>(a.size()));
        break;
      }
      case OpKind::Relu: {
        if (n->inputs.empty()) break;
        buf = host_bufs_[n->inputs[0]];
        if (gpu) cuda_relu(buf.data(), static_cast<int>(buf.size()));
        else cpu_relu(buf.data(), static_cast<int>(buf.size()));
        break;
      }
      case OpKind::Softmax: {
        if (n->inputs.empty()) break;
        buf = host_bufs_[n->inputs[0]];
        cpu_softmax(buf.data(), static_cast<int>(buf.size()));
        break;
      }
      case OpKind::Output: {
        if (!n->inputs.empty()) buf = host_bufs_[n->inputs[0]];
        output = buf;
        break;
      }
      default:
        if (!n->inputs.empty()) buf = host_bufs_[n->inputs[0]];
        break;
    }
  }

  if (output.empty() && !graph_.nodes().empty()) {
    output = host_bufs_[graph_.nodes().back().id];
  }
}

std::vector<float> InferenceRuntime::run(const std::vector<float>& input) {
  std::vector<float> output;
  if (!loaded_) return output;
  run_cuda(input, output);
  return output;
}

BenchResult InferenceRuntime::benchmark(const std::vector<float>& input,
                                        int warmup, int iters) {
  BenchResult r;
  r.device = use_cuda_ ? "cuda" : "cpu";

  for (int i = 0; i < warmup; ++i) (void)run(input);

  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) (void)run(input);
  auto t1 = std::chrono::steady_clock::now();

  double total_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();
  r.latency_ms = total_ms / std::max(iters, 1);
  r.throughput = 1000.0 / std::max(r.latency_ms, 1e-9);
  return r;
}

std::string InferenceRuntime::profile(const std::vector<float>& input) {
  std::ostringstream out;
  out << "device: " << (use_cuda_ ? "cuda" : "cpu") << "\n";

  auto plan = schedule_graph(graph_);
  out << "schedule order:";
  for (int id : plan.order) out << " " << id;
  out << "\n";
  out << "estimated_ms: " << plan.estimated_ms << "\n";

  for (const auto& n : graph_.nodes()) {
    out << "op %" << n.id << " " << op_name(n.kind)
        << " name=" << n.name << "\n";
  }

  auto br = benchmark(input, 2, 10);
  out << "measured_latency_ms: " << br.latency_ms << "\n";
  out << "throughput_qps: " << br.throughput << "\n";
  return out.str();
}
