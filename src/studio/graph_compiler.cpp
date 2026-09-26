#include "studio/graph_compiler.hpp"

#include <algorithm>
#include <format>
#include <functional>
#include <set>

#include "klartraum/gaussian_splatting_factory.hpp"

#include "studio/graph_serialization.hpp"

namespace kstudio {

CompilePlan planGraph(const Graph& graph, const OnnxInfoProvider& onnxInfo, const InputExists& inputExists) {
    CompilePlan plan;
    plan.diagnostics = graph.validate();
    ShapeInference shapes = inferTensorShapes(graph, onnxInfo);
    plan.diagnostics.insert(plan.diagnostics.end(), shapes.diagnostics.begin(), shapes.diagnostics.end());
    if (inputExists) {
        for (const auto& node : graph.nodes()) {
            const std::string* path = nullptr;
            if (node.kind == NodeKind::Scene) {
                path = &node.as<SceneParams>().path;
            } else if (node.kind == NodeKind::ImageFile) {
                path = &node.as<ImageFileParams>().path;
            }
            if (path && !path->empty() && !inputExists(*path)) {
                plan.diagnostics.push_back(Diagnostic{Severity::Error, node.id, "File not found: " + *path});
            }
        }
    }

    std::set<int> errorNodes;
    bool graphError = false;
    for (const auto& d : plan.diagnostics) {
        if (d.severity != Severity::Error) {
            continue;
        }
        if (d.node < 0) {
            graphError = true;
        } else {
            errorNodes.insert(d.node);
        }
    }
    auto clean = [&](const std::vector<int>& nodes) {
        return !graphError && std::none_of(nodes.begin(), nodes.end(), [&](int id) { return errorNodes.contains(id); });
    };

    // Run part: everything upstream of the sinks.
    std::vector<int> sinks;
    std::set<int> runNodes;
    for (const auto& node : graph.nodes()) {
        if (isSink(node.kind)) {
            sinks.push_back(node.id);
            for (int id : graph.upstreamOf(node.id)) {
                runNodes.insert(id);
            }
        }
    }
    if (!sinks.empty() && clean(std::vector<int>(runNodes.begin(), runNodes.end()))) {
        RunPlan run;
        for (int id : graph.topologicalOrder()) {
            if (runNodes.contains(id)) {
                run.nodes.push_back(id);
            }
        }
        run.sinks = sinks;
        run.shapes = std::move(shapes.shapes);
        plan.run = std::move(run);
    }

    // Live part: validate() allows at most one Present.
    const auto present = std::find_if(graph.nodes().begin(), graph.nodes().end(),
                                      [](const Node& n) { return n.kind == NodeKind::Present; });
    if (present == graph.nodes().end() || !clean(graph.upstreamOf(present->id))) {
        return plan;
    }
    LivePlan live;
    live.presentNode = present->id;
    std::map<int, size_t> visited;
    std::function<void(int)> visit = [&](int id) {
        if (auto it = visited.find(id); it != visited.end()) {
            live.signature += std::format("#{}", it->second);
            return;
        }
        visited[id] = live.nodes.size();
        live.nodes.push_back(id);
        const Node& node = *graph.findNode(id);
        live.signature += kindInfo(node.kind).name;
        if (node.kind == NodeKind::Camera) {
            live.cameraNode = id;
        } else {
            live.signature += paramsToString(node.params);
        }
        live.signature += "(";
        const auto& inputs = kindInfo(node.kind).inputs;
        for (int slot = 0; slot < static_cast<int>(inputs.size()); ++slot) {
            const Link* link = graph.inputLink(id, slot);
            live.signature += std::format("{}{}:", slot == 0 ? "" : ",", link->fromSlot);
            visit(link->fromNode);
        }
        live.signature += ")";
    };
    visit(present->id);
    plan.live = std::move(live);
    return plan;
}

klartraum::GsplatConfig toGsplatConfig(const SplattingParams& params) {
    klartraum::GsplatConfig config;
    config.spreadMultiplier = params.spreadMultiplier;
    config.maxMod = params.maxMod;
    config.numSortWGsCap = params.numSortWGsCap;
    config.splatTileX = params.splatTileX;
    config.splatTileY = params.splatTileY;
    config.shDegree = params.shDegree;
    config.alphaCullThreshold = params.alphaCullThreshold;
    config.useMeshShader = params.useMeshShader;
    return config;
}

klartraum::GsplatBackend toGsplatBackend(SplattingBackend backend) {
    return backend == SplattingBackend::Raster ? klartraum::GsplatBackend::Raster : klartraum::GsplatBackend::Compute;
}

} // namespace kstudio
