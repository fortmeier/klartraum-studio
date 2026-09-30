#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "studio/graph_model.hpp"

namespace kstudio {

// Graphs are stored as JSON ("*.ktgraph.json"): a format version, the nodes
// with their parameters and editor positions, the links and the graph's meta
// node definitions.
std::string toJson(const Graph& graph);

// Throws std::runtime_error with a readable message on malformed input. Files
// from before Upload Gaussians existed, with scenes linked straight into
// Gaussian Splatting, get an Upload Gaussians node on each such link.
Graph fromJson(const std::string& text);

// A node's parameters as they are stored, e.g. for comparing them.
std::string paramsToString(const NodeParams& params);

// Parameters by their key in graph files, e.g. for the parameters a meta node
// definition exposes. Values are JSON text.
std::vector<std::string> paramKeys(const NodeParams& params);
std::optional<std::string> paramValue(const NodeParams& params, std::string_view key);
// Throws std::runtime_error if the kind has no such parameter or the value
// does not fit it.
void setParamValue(NodeParams& params, NodeKind kind, std::string_view key, const std::string& value);

void saveGraph(const Graph& graph, const std::filesystem::path& path);
Graph loadGraph(const std::filesystem::path& path);

} // namespace kstudio
