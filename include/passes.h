#pragma once

#include "graph.h"

// lower high level ops into something closer to the runtime
void lower_graph(Graph& g);

// smash matmul + relu into one fused node when we can
void fuse_operators(Graph& g);

// turn float tensors into int8 with a simple scale
void quantize_tensors(Graph& g, float scale = 127.0f);

// rough cost model so the scheduler can pick an order
struct SchedulePlan {
  std::vector<int> order;
  double estimated_ms = 0.0;
};

SchedulePlan schedule_graph(const Graph& g);

// run all the passes we usually want
void run_compiler_pipeline(Graph& g, bool do_quantize = true);
