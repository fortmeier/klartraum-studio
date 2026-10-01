#include "studio/graph_compiler.hpp"

#include <algorithm>
#include <format>
#include <functional>
#include <set>
#include <tuple>

#include "klartraum/gaussian_splatting_factory.hpp"

#include "studio/graph_serialization.hpp"

namespace kstudio {

namespace {

CompilePlan planFlatGraph(const Graph& graph, const OnnxInfoProvider& onnxInfo, const InputExists& inputExists) {
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
            } else if (node.kind == NodeKind::Prompt) {
                path = &node.as<PromptParams>().vocabulary;
            } else if (node.kind == NodeKind::LatentNoise) {
                path = &node.as<LatentNoiseParams>().path;  // optional
            }
            if (path && !path->empty() && !inputExists(*path)) {
                plan.diagnostics.push_back(Diagnostic{Severity::Error, node.id, "File not found: " + *path});
            }
        }
    }

    // Run-only nodes: those that depend on a staged node (which needs
    // several submissions), unless they also depend on something that changes
    // every frame. A live graph reads their results from the last Run.
    std::set<int> varying;
    std::set<int> afterStaged;
    for (int id : graph.topologicalOrder()) {
        const Node& node = *graph.findNode(id);
        bool isVarying = node.kind == NodeKind::Camera || node.kind == NodeKind::SwapchainTarget ||
                         node.kind == NodeKind::Time;
        bool isAfterStaged = isStaged(node.kind);
        for (const auto& link : graph.links()) {
            if (link.toNode == id) {
                isVarying = isVarying || varying.contains(link.fromNode);
                isAfterStaged = isAfterStaged || afterStaged.contains(link.fromNode);
            }
        }
        if (isVarying) {
            varying.insert(id);
        }
        if (isAfterStaged) {
            afterStaged.insert(id);
        }
        if (isStaged(node.kind) && isVarying) {
            plan.diagnostics.push_back(
                Diagnostic{Severity::Error, id,
                           "Runs only with Run, so it cannot depend on a camera, the swapchain or time."});
        }
    }
    auto runOnly = [&](int id) { return afterStaged.contains(id) && !varying.contains(id); };

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

    // Live part: Present and what it needs, down to run-only nodes, whose
    // outputs it reads as retained results. validate() allows at most one
    // Present.
    const auto present = std::find_if(graph.nodes().begin(), graph.nodes().end(),
                                      [](const Node& n) { return n.kind == NodeKind::Present; });
    LivePlan live;
    if (present != graph.nodes().end()) {
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
            }
            // Live parameters apply without rebuilding (camera, CPU numbers,
            // Make Transform values).
            if (!kindInfo(node.kind).liveParams) {
                live.signature += paramsToString(node.params);
            }
            live.signature += "(";
            const auto inputs = graph.inputPins(node);
            for (int slot = 0; slot < static_cast<int>(inputs.size()); ++slot) {
                const Link* link = graph.inputLink(id, slot);
                if (!link) {
                    // An optional input left unconnected.
                    live.signature += std::format("{}-", slot == 0 ? "" : ",");
                    continue;
                }
                if (runOnly(link->fromNode)) {
                    const OutputPin pin{link->fromNode, link->fromSlot};
                    if (std::find(live.retained.begin(), live.retained.end(), pin) == live.retained.end()) {
                        live.retained.push_back(pin);
                    }
                    live.signature += std::format("{}retained {}.{}", slot == 0 ? "" : ",",
                                                  kindInfo(graph.findNode(link->fromNode)->kind).name,
                                                  link->fromSlot);
                    continue;
                }
                live.signature += std::format("{}{}:", slot == 0 ? "" : ",", link->fromSlot);
                visit(link->fromNode);
            }
            live.signature += ")";
        };
        visit(present->id);
    }

    // Run part: everything upstream of the sinks, and of the results the live
    // part retains.
    std::vector<int> sinks;
    std::set<int> runNodes;
    auto addUpstream = [&](int node) {
        for (int id : graph.upstreamOf(node)) {
            runNodes.insert(id);
        }
    };
    for (const auto& node : graph.nodes()) {
        if (isSink(node.kind)) {
            sinks.push_back(node.id);
            addUpstream(node.id);
        }
    }
    for (const OutputPin& pin : live.retained) {
        addUpstream(pin.first);
    }
    if ((!sinks.empty() || !live.retained.empty()) && clean(std::vector<int>(runNodes.begin(), runNodes.end()))) {
        RunPlan run;
        for (int id : graph.topologicalOrder()) {
            if (runNodes.contains(id)) {
                run.nodes.push_back(id);
            }
        }
        run.sinks = sinks;
        run.retained = live.retained;
        run.types = std::move(shapes.types);
        plan.run = std::move(run);
    }

    // The live part needs the run-only nodes it reads from to be free of
    // errors, too: they make its retained results.
    if (present != graph.nodes().end() && clean(graph.upstreamOf(present->id))) {
        plan.live = std::move(live);
    }
    return plan;
}

} // namespace

CompilePlan planGraph(const Graph& graph, const OnnxInfoProvider& onnxInfo, const InputExists& inputExists) {
    FlatGraph flat = flatten(graph);
    CompilePlan plan = planFlatGraph(flat.graph, onnxInfo, inputExists);
    // A meta node that could not be expanded is missing from the flat graph,
    // so the nodes it feeds report unconnected inputs, which keeps the plans
    // that need it from being made.
    std::vector<Diagnostic> diagnostics = flat.diagnostics;
    std::set<std::tuple<int, Severity, std::string>> seen;
    for (auto& d : plan.diagnostics) {
        if (d.node >= 0) {
            const int top = flat.top(d.node);
            // Inner nodes are titled "<meta node> / <inner node>".
            if (top != d.node && d.severity != Severity::Info) {
                d.message = flat.graph.findNode(d.node)->title + ": " + d.message;
            }
            d.node = top;
        }
        if (seen.insert({d.node, d.severity, d.message}).second) {
            diagnostics.push_back(std::move(d));
        }
    }
    plan.diagnostics = std::move(diagnostics);
    plan.flat = std::move(flat);
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
