#include "studio/graph_introspection.hpp"

#include <algorithm>
#include <functional>

#include "klartraum/computegraph/computegraphelement.hpp"

namespace kstudio {

namespace {
constexpr int kUnassigned = -2;
}

ElementCategory categorize(std::string_view type) {
    auto starts = [&](std::string_view prefix) { return type.starts_with(prefix); };
    if (starts("BufferElement") || starts("TensorElement")) {
        return ElementCategory::Buffer;
    }
    if (starts("UniformBufferObject") || starts("HostValues")) {
        return ElementCategory::Uniform;
    }
    if (starts("ImageViewSrcTransition")) {
        return ElementCategory::Sync;
    }
    if (starts("ImageViewSrc") || starts("ImageSrc") || starts("OffscreenTarget")) {
        return ElementCategory::Image;
    }
    if (starts("GeneralComputation") || starts("BufferTransformation") || starts("CopyBuffer") ||
        starts("OnnxNetwork") || starts("ImageResample") || starts("TransformBuffer") ||
        starts("GaussianTransform") || starts("GaussianMerge")) {
        return ElementCategory::Compute;
    }
    if (starts("RenderPass")) {
        return ElementCategory::Graphics;
    }
    if (starts("BufferToGraphicsBarrier") || starts("NoOp")) {
        return ElementCategory::Sync;
    }
    if (starts("GaussianSplatting") || starts("ComputeGraphGroup")) {
        return ElementCategory::Group;
    }
    return ElementCategory::Other;
}

std::string_view categoryName(ElementCategory category) {
    switch (category) {
    case ElementCategory::Buffer: return "Buffer";
    case ElementCategory::Uniform: return "Uniform";
    case ElementCategory::Image: return "Image";
    case ElementCategory::Compute: return "Compute";
    case ElementCategory::Graphics: return "Graphics";
    case ElementCategory::Sync: return "Sync";
    case ElementCategory::Group: return "Group";
    case ElementCategory::Other: return "Other";
    }
    return "?";
}

const ElementNode* ElementGraph::find(int id) const {
    auto it = std::find_if(nodes.begin(), nodes.end(), [id](const ElementNode& n) { return n.id == id; });
    return it == nodes.end() ? nullptr : &*it;
}

ElementGraph introspect(const std::shared_ptr<klartraum::ComputeGraphElement>& root,
                        const std::map<const klartraum::ComputeGraphElement*, int>& owners, int defaultOwner,
                        const std::set<const klartraum::ComputeGraphElement*>& inserted) {
    ElementGraph graph;
    if (!root) {
        return graph;
    }

    std::map<const klartraum::ComputeGraphElement*, int> ids;

    // Post-order DFS, so every producer gets its id before its consumers.
    std::function<int(const std::shared_ptr<klartraum::ComputeGraphElement>&)> visit =
        [&](const std::shared_ptr<klartraum::ComputeGraphElement>& element) -> int {
        if (auto it = ids.find(element.get()); it != ids.end()) {
            return it->second;
        }
        ids[element.get()] = -1;  // in progress; klartraum graphs are acyclic

        std::vector<std::pair<int, int>> inputs;  // (slot, id)
        for (const auto& [slot, input] : element->getInputs()) {
            if (input) {
                inputs.emplace_back(slot, visit(input));
            }
        }

        ElementNode node;
        node.id = static_cast<int>(graph.nodes.size());
        node.name = element->getName();
        node.type = element->getType();
        node.category = categorize(node.type);
        node.inserted = inserted.contains(element.get());
        if (auto it = owners.find(element.get()); it != owners.end()) {
            node.owner = it->second;
        } else {
            node.owner = kUnassigned;
        }
        for (const auto& [slot, inputId] : inputs) {
            node.inputs.push_back(inputId);
            graph.edges.push_back(ElementEdge{static_cast<int>(graph.edges.size()), inputId, node.id, slot});
        }
        ids[element.get()] = node.id;
        graph.nodes.push_back(std::move(node));
        return graph.nodes.back().id;
    };
    graph.root = visit(root);

    for (const auto& edge : graph.edges) {
        auto& outputs = graph.nodes[edge.from].outputs;
        if (std::find(outputs.begin(), outputs.end(), edge.to) == outputs.end()) {
            outputs.push_back(edge.to);
        }
    }

    // Unlisted elements (e.g. inside a group) belong to what consumes them.
    // Consumers come after producers, so walking backwards resolves chains.
    for (auto it = graph.nodes.rbegin(); it != graph.nodes.rend(); ++it) {
        if (it->owner != kUnassigned) {
            continue;
        }
        it->owner = defaultOwner;
        for (int consumer : it->outputs) {
            if (graph.nodes[consumer].owner != kUnassigned) {
                it->owner = graph.nodes[consumer].owner;
                break;
            }
        }
    }
    return graph;
}

} // namespace kstudio
