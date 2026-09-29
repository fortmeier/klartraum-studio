#pragma once

#include <algorithm>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "studio/graph_model.hpp"
#include "studio/tensor_shapes.hpp"

namespace klartraum {
struct GsplatConfig;
enum class GsplatBackend;
} // namespace klartraum

namespace kstudio {

// What runs every frame: the Present node and everything feeding it.
struct LivePlan {
    int presentNode = 0;
    // Depth first from Present, inputs in slot order. Two plans with the same
    // signature list corresponding nodes at the same positions.
    std::vector<int> nodes;
    int cameraNode = 0;  // the camera the window's orbit camera drives; 0: none
    // The nodes' kinds, parameters and links, without node ids and live
    // parameters (NodeKindInfo::liveParams): those apply while it runs.
    std::string signature;

    // Whether switching from `other` to this plan needs new pipelines.
    bool needsRebuildFrom(const LivePlan& other) const { return signature != other.signature; }
    bool contains(int node) const { return std::find(nodes.begin(), nodes.end(), node) != nodes.end(); }
};

// What Run executes: every node that feeds a sink, in dependency order.
struct RunPlan {
    std::vector<int> nodes;
    std::vector<int> sinks;
    std::map<OutputPin, TensorType> types;  // tensor types, where known (see inferTensorShapes)
};

struct CompilePlan {
    std::optional<LivePlan> live;  // the live part, if any and valid
    std::optional<RunPlan> run;    // the run part, if any and valid
    std::vector<Diagnostic> diagnostics;

    bool ok() const { return live.has_value(); }
};

// Tells whether an input file (scene, image) stored in the graph exists.
using InputExists = std::function<bool(const std::string& path)>;

// Validates the graph and extracts the plans. Each plan is only produced when
// none of the nodes it depends on has an error, so an error in the run part
// does not stop live rendering and vice versa. With `onnxInfo`, tensor shapes
// are checked as well; with `inputExists`, scene and image files.
CompilePlan planGraph(const Graph& graph, const OnnxInfoProvider& onnxInfo = {}, const InputExists& inputExists = {});

klartraum::GsplatConfig toGsplatConfig(const SplattingParams& params);
klartraum::GsplatBackend toGsplatBackend(SplattingBackend backend);

} // namespace kstudio
