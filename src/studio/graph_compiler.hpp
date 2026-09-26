#pragma once

#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "studio/graph_model.hpp"
#include "studio/tensor_shapes.hpp"

namespace klartraum {
class ComputeGraphElement;
class GaussianDataStandard;
class KlartraumEngine;
struct GsplatConfig;
enum class GsplatBackend;
} // namespace klartraum

namespace kstudio {

// What the authoring graph asks klartraum to build: the chain that feeds the
// Present node. The klartraum backends render one Gaussian-splatting pass
// straight into the swapchain, so that is the only shape the plan describes.
struct SplattingPlan {
    int presentNode = 0;
    int splattingNode = 0;
    int sceneNode = 0;
    int cameraNode = 0;
    int targetNode = 0;
    std::string scenePath;
    SplattingParams params;

    // Whether switching from `other` to this plan needs new pipelines. Camera
    // parameters are applied live, and node ids only affect the UI.
    bool needsRebuildFrom(const SplattingPlan& other) const {
        return scenePath != other.scenePath || !(params == other.params);
    }
};

// What Run executes: every node that feeds a sink, in dependency order.
struct RunPlan {
    std::vector<int> nodes;
    std::vector<int> sinks;
    std::map<int, TensorShape> shapes;  // tensor shapes, when ONNX info was available
};

struct CompilePlan {
    std::optional<SplattingPlan> splatting;  // the live part, if any and valid
    std::optional<RunPlan> run;              // the run part, if any and valid
    std::vector<Diagnostic> diagnostics;

    bool ok() const { return splatting.has_value(); }
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

// The klartraum elements built for a plan.
struct BuiltGraph {
    std::shared_ptr<klartraum::ComputeGraphElement> root;
    // Maps the elements that stand for an authoring node's output to that
    // node (scene buffers, camera UBO, swapchain images); everything else
    // belongs to the splatting node.
    std::map<const klartraum::ComputeGraphElement*, int> owners;
};

// Builds the plan's elements for the engine's current swapchain and adds them
// to the engine. Meant to run inside a KlartraumEngine graph builder.
BuiltGraph buildGraph(klartraum::KlartraumEngine& engine, const SplattingPlan& plan,
                      const std::shared_ptr<klartraum::GaussianDataStandard>& model);

} // namespace kstudio
