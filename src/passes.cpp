#include "passes.h"

#include <algorithm>
#include <cmath>
#include <unordered_set>

void lower_graph(Graph& g) {
  // for now lowering mostly renames things and cleans shapes
  // keeps the graph in a form the runtime likes
  for (auto& n : g.nodes()) {
    if (n.kind == OpKind::Softmax && n.shape.empty()) {
      // fill a default shape if someone forgot
      n.shape = {1, 10};
    }
    if (n.name.empty()) {
      n.name = op_name(n.kind) + "_" + std::to_string(n.id);
    }
  }
}

void fuse_operators(Graph& g) {
  // look for matmul followed by relu and merge them
  auto& nodes = g.nodes();
  std::unordered_set<int> remove;

  for (size_t i = 0; i < nodes.size(); ++i) {
    if (nodes[i].kind != OpKind::Relu) continue;
    if (nodes[i].inputs.size() != 1) continue;

    int src = nodes[i].inputs[0];
    Node* mat = g.get(src);
    if (!mat || mat->kind != OpKind::MatMul) continue;

    // make sure this matmul only feeds the relu
    int users = 0;
    for (const auto& o : nodes) {
      for (int in : o.inputs) {
        if (in == mat->id) users++;
      }
    }
    if (users != 1) continue;

    mat->kind = OpKind::FusedMatMulRelu;
    mat->name = "fused_" + mat->name;
    // point anything that used the relu at the fused op
    for (auto& o : nodes) {
      for (int& in : o.inputs) {
        if (in == nodes[i].id) in = mat->id;
      }
    }
    remove.insert(nodes[i].id);
  }

  nodes.erase(
      std::remove_if(nodes.begin(), nodes.end(),
                     [&](const Node& n) { return remove.count(n.id) > 0; }),
      nodes.end());
}

void quantize_tensors(Graph& g, float scale) {
  // simple absmax style scale on constant weights
  for (auto& n : g.nodes()) {
    if (n.kind != OpKind::Constant) continue;
    if (n.data.empty()) continue;

    float max_abs = 0.0f;
    for (float v : n.data) max_abs = std::max(max_abs, std::fabs(v));
    if (max_abs < 1e-8f) max_abs = 1.0f;

    n.scale = max_abs / scale;
    n.zero_point = 0;
    // store dequantized-ish values so cpu path still works
    // real int8 would live in a separate buffer
    for (float& v : n.data) {
      int q = static_cast<int>(std::round(v / n.scale));
      q = std::max(-128, std::min(127, q));
      v = q * n.scale;
    }
    n.name = n.name + "_q8";
  }
}

static double op_cost_ms(const Node& n) {
  // hand wavy costs so scheduling has something to sort on
  long elems = 1;
  for (int d : n.shape) elems *= std::max(d, 1);

  switch (n.kind) {
    case OpKind::MatMul:
    case OpKind::FusedMatMulRelu:
      return 0.002 * elems;
    case OpKind::Softmax:
      return 0.001 * elems;
    case OpKind::Relu:
    case OpKind::Add:
      return 0.0002 * elems;
    default:
      return 0.0001 * elems;
  }
}

SchedulePlan schedule_graph(const Graph& g) {
  // topological order with a cheap cost estimate
  SchedulePlan plan;
  std::unordered_set<int> done;
  std::vector<Node> pending = g.nodes();

  while (plan.order.size() < g.nodes().size()) {
    bool progressed = false;
    for (const auto& n : pending) {
      if (done.count(n.id)) continue;
      bool ready = true;
      for (int in : n.inputs) {
        if (!done.count(in)) {
          // constants and inputs may not be in done yet if missing
          const Node* p = g.get(in);
          if (p && !done.count(in)) {
            ready = false;
            break;
          }
        }
      }
      if (!ready) continue;
      plan.order.push_back(n.id);
      plan.estimated_ms += op_cost_ms(n);
      done.insert(n.id);
      progressed = true;
    }
    if (!progressed) {
      // just append leftovers so we never hang
      for (const auto& n : pending) {
        if (!done.count(n.id)) {
          plan.order.push_back(n.id);
          plan.estimated_ms += op_cost_ms(n);
          done.insert(n.id);
        }
      }
    }
  }
  return plan;
}

void run_compiler_pipeline(Graph& g, bool do_quantize) {
  lower_graph(g);
  fuse_operators(g);
  if (do_quantize) quantize_tensors(g);
  // schedule is computed by the caller / runtime when needed
  (void)schedule_graph(g);
}
