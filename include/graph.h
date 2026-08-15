#pragma once

#include <string>
#include <vector>

// simple op kinds we care about for inference
enum class OpKind {
  Input,
  Constant,
  MatMul,
  Add,
  Relu,
  Softmax,
  Quantize,
  Dequantize,
  FusedMatMulRelu,
  Output
};

// one node in the graph
struct Node {
  int id;
  OpKind kind;
  std::string name;
  std::vector<int> inputs;
  std::vector<int> shape;
  // for constants / weights
  std::vector<float> data;
  // scale used after we quantize
  float scale = 1.0f;
  // zero point for int8 path
  int zero_point = 0;
};

// tiny mlir-ish graph container
class Graph {
public:
  int add_node(OpKind kind, const std::string& name,
               const std::vector<int>& inputs,
               const std::vector<int>& shape);

  Node* get(int id);
  const Node* get(int id) const;

  const std::vector<Node>& nodes() const { return nodes_; }
  std::vector<Node>& nodes() { return nodes_; }

  // dump a rough mlir looking text form
  std::string to_mlir() const;

  // load from that same text form
  static Graph from_mlir(const std::string& text);

private:
  std::vector<Node> nodes_;
  int next_id_ = 0;
};

std::string op_name(OpKind kind);
OpKind op_from_name(const std::string& name);
