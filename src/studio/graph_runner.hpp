#pragma once

#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "studio/gaussian_sources.hpp"
#include "studio/graph_compiler.hpp"
#include "studio/graph_introspection.hpp"
#include "studio/image_io.hpp"
#include "studio/tensor_shapes.hpp"

namespace klartraum {
class ComputeGraphElement;
class GaussianDataStandard;
class KlartraumEngine;
class VulkanContext;
} // namespace klartraum

namespace kstudio {

// How a run or the live graph finds its inputs and places its outputs; paths
// are the ones stored in the graph.
struct RunContext {
    // Resolves an input file (image, ONNX model); throws if it is missing.
    std::function<std::filesystem::path(const std::string&)> resolveInput;
    // Resolves where an output file goes.
    std::function<std::filesystem::path(const std::string&)> resolveOutput;
    // Uploads (or returns cached) Gaussians assembled from scene files; see
    // gaussianParts().
    std::function<std::shared_ptr<klartraum::GaussianDataStandard>(const std::vector<GaussianPart>&)> loadGaussians;
    OnnxInfoProvider onnxInfo;
};

struct RunResult {
    std::map<int, ImageRGBA8> images;             // per sink node
    std::vector<std::filesystem::path> written;   // files written by Image File Writers
    ElementGraph compiled;                        // the klartraum graph that ran
    std::map<std::string, float> timings;         // GPU milliseconds per element label
    double milliseconds = 0.0;                    // build, execution and readback
};

// Builds the plan's nodes as a standalone klartraum compute graph (one
// path), executes it once and reads the sinks back. Throws
// std::runtime_error naming the failing node.
RunResult runGraph(klartraum::VulkanContext& vulkanContext, const Graph& graph, const RunPlan& plan,
                   const RunContext& context);

// The klartraum elements built for the live plan.
struct BuiltGraph {
    std::shared_ptr<klartraum::ComputeGraphElement> root;
    // Maps elements to the authoring node they were built for (see
    // introspect()).
    std::map<const klartraum::ComputeGraphElement*, int> owners;
};

// Builds the live plan's nodes for the engine's current swapchain, one path
// per swapchain image, and adds them to the engine; the plan's camera gets
// the engine's camera UBO. The Present node shows a rendering into the
// swapchain as it is and stretches any other image to the window. Meant to
// run inside a KlartraumEngine graph builder. Throws std::runtime_error
// naming the failing node.
BuiltGraph buildLiveGraph(klartraum::KlartraumEngine& engine, const Graph& graph, const LivePlan& plan,
                          const RunContext& context);

} // namespace kstudio
