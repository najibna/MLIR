#include "graph.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

std::string op_name(OpKind kind) {
  switch (kind) {
    case OpKind::Input: return "input";
    case OpKind::Constant: return "constant";
    case OpKind::MatMul: return "matmul";
    case OpKind::Add: return "add";
    case OpKind::Relu: return "relu";
    case OpKind::Softmax: return "softmax";
    case OpKind::Quantize: return "quantize";
    case OpKind::Dequantize: return "dequantize";
    case OpKind::FusedMatMulRelu: return "fused_matmul_relu";
    case OpKind::Output: return "output";
  }
  return "unknown";
}

OpKind op_from_name(const std::string& name) {
  if (name == "input") return OpKind::Input;
  if (name == "constant") return OpKind::Constant;
  if (name == "matmul") return OpKind::MatMul;
  if (name == "add") return OpKind::Add;
  if (name == "relu") return OpKind::Relu;
  if (name == "softmax") return OpKind::Softmax;
  if (name == "quantize") return OpKind::Quantize;
  if (name == "dequantize") return OpKind::Dequantize;
  if (name == "fused_matmul_relu") return OpKind::FusedMatMulRelu;
  if (name == "output") return OpKind::Output;
  throw std::runtime_error("unknown op: " + name);
}

int Graph::add_node(OpKind kind, const std::string& name,
                    const std::vector<int>& inputs,
                    const std::vector<int>& shape) {
  Node n;
  n.id = next_id_++;
  n.kind = kind;
  n.name = name;
  n.inputs = inputs;
  n.shape = shape;
  nodes_.push_back(n);
  return n.id;
}

Node* Graph::get(int id) {
  for (auto& n : nodes_) {
    if (n.id == id) return &n;
  }
  return nullptr;
}

const Node* Graph::get(int id) const {
  for (const auto& n : nodes_) {
    if (n.id == id) return &n;
  }
  return nullptr;
}

std::string Graph::to_mlir() const {
  // write something that looks a bit like mlir dialect text
  std::ostringstream out;
  out << "module {\n";
  out << "  func @main() {\n";
  for (const auto& n : nodes_) {
    out << "    %" << n.id << " = nn." << op_name(n.kind) << " ";
    if (!n.inputs.empty()) {
      out << "(";
      for (size_t i = 0; i < n.inputs.size(); ++i) {
        if (i) out << ", ";
        out << "%" << n.inputs[i];
      }
      out << ") ";
    }
    out << ": tensor<";
    for (size_t i = 0; i < n.shape.size(); ++i) {
      if (i) out << "x";
      out << n.shape[i];
    }
    out << "xf32>  // " << n.name << "\n";
  }
  out << "    return\n";
  out << "  }\n";
  out << "}\n";
  return out.str();
}

Graph Graph::from_mlir(const std::string& text) {
  // very small parser, just enough for our demo graphs
  Graph g;
  std::istringstream in(text);
  std::string line;
  std::unordered_map<int, int> id_map;

  while (std::getline(in, line)) {
    auto pos = line.find("nn.");
    if (pos == std::string::npos) continue;

    // find %id
    auto pct = line.find('%');
    if (pct == std::string::npos) continue;
    int old_id = std::stoi(line.substr(pct + 1));

    auto op_start = pos + 3;
    auto op_end = line.find(' ', op_start);
    std::string op = line.substr(op_start, op_end - op_start);
    OpKind kind = op_from_name(op);

    std::vector<int> inputs;
    auto paren = line.find('(');
    auto close = line.find(')');
    if (paren != std::string::npos && close != std::string::npos && close > paren) {
      std::string inside = line.substr(paren + 1, close - paren - 1);
      std::istringstream pin(inside);
      std::string tok;
      while (std::getline(pin, tok, ',')) {
        auto p = tok.find('%');
        if (p != std::string::npos) {
          int ref = std::stoi(tok.substr(p + 1));
          inputs.push_back(id_map.count(ref) ? id_map[ref] : ref);
        }
      }
    }

    std::vector<int> shape;
    auto tpos = line.find("tensor<");
    if (tpos != std::string::npos) {
      auto tend = line.find('>', tpos);
      std::string shape_s = line.substr(tpos + 7, tend - tpos - 7);
      // drop trailing xf32
      auto x = shape_s.find('x');
      auto f = shape_s.find('f');
      if (f != std::string::npos) shape_s = shape_s.substr(0, f);
      std::replace(shape_s.begin(), shape_s.end(), 'x', ' ');
      std::istringstream sin(shape_s);
      int d;
      while (sin >> d) shape.push_back(d);
    }

    std::string name = op;
    auto cmt = line.find("//");
    if (cmt != std::string::npos) {
      name = line.substr(cmt + 2);
      while (!name.empty() && name[0] == ' ') name.erase(name.begin());
    }

    int nid = g.add_node(kind, name, inputs, shape);
    id_map[old_id] = nid;
  }
  return g;
}
