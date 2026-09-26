#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "studio/graph_model.hpp"

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

struct CompilePlan {
    std::optional<SplattingPlan> splatting;
    std::vector<Diagnostic> diagnostics;

    bool ok() const { return splatting.has_value(); }
};

// Validates the graph and extracts the plan. The plan is empty if the graph
// has errors.
CompilePlan planGraph(const Graph& graph);

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
