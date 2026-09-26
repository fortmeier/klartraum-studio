#pragma once

#include <filesystem>
#include <string>

#include "studio/graph_model.hpp"

namespace kstudio {

// Graphs are stored as JSON ("*.ktgraph.json"): a format version, the nodes
// with their parameters and editor positions, and the links.
std::string toJson(const Graph& graph);

// Throws std::runtime_error with a readable message on malformed input.
Graph fromJson(const std::string& text);

// A node's parameters as they are stored, e.g. for comparing them.
std::string paramsToString(const NodeParams& params);

void saveGraph(const Graph& graph, const std::filesystem::path& path);
Graph loadGraph(const std::filesystem::path& path);

} // namespace kstudio
