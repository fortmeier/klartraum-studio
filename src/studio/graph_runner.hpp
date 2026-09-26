#pragma once

#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "studio/graph_compiler.hpp"
#include "studio/graph_introspection.hpp"
#include "studio/image_io.hpp"
#include "studio/tensor_shapes.hpp"

namespace klartraum {
class GaussianDataStandard;
class VulkanContext;
} // namespace klartraum

namespace kstudio {

// How a run finds its inputs and places its outputs; paths are the ones
// stored in the graph.
struct RunContext {
    // Resolves an input file (image, ONNX model); throws if it is missing.
    std::function<std::filesystem::path(const std::string&)> resolveInput;
    // Resolves where an output file goes.
    std::function<std::filesystem::path(const std::string&)> resolveOutput;
    // Loads (or returns a cached) scene for a Scene node's path.
    std::function<std::shared_ptr<klartraum::GaussianDataStandard>(const std::string&)> loadScene;
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

} // namespace kstudio
