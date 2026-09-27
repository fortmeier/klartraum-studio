#pragma once

#include "studio/graph_model.hpp"

namespace kstudio {

// The value of a CPU number output (Number, Time or Sine node) `time`
// seconds after the studio started. Throws std::runtime_error if an input is
// not connected.
float evaluateNumber(const Graph& graph, int node, double time);

} // namespace kstudio
