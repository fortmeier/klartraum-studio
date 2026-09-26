#pragma once

#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace klartraum {
class ComputeGraphElement;
}

namespace kstudio {

// A read-only snapshot of a compiled klartraum compute graph, taken by
// walking ComputeGraphElement::getInputs() from the root element. That is the
// traversal ComputeGraph::compileFrom() uses, so the snapshot holds exactly
// the elements klartraum records, including those inside groups such as the
// Gaussian-splatting backends.

enum class ElementCategory { Buffer, Uniform, Image, Compute, Graphics, Sync, Group, Other };

ElementCategory categorize(std::string_view type);
std::string_view categoryName(ElementCategory category);

struct ElementNode {
    int id = 0;
    std::string name;    // may be empty
    std::string type;
    ElementCategory category = ElementCategory::Other;
    int owner = -1;      // authoring node this element was built for, or -1
    std::vector<int> inputs;   // element ids, in input-slot order
    std::vector<int> outputs;  // element ids consuming this one

    // The label klartraum uses in profiling results: the name, or the type.
    std::string label() const { return name.empty() ? type : name; }
};

struct ElementEdge {
    int id = 0;
    int from = 0;  // producer (input of `to`)
    int to = 0;
    int slot = 0;  // input index on `to`
};

struct ElementGraph {
    std::vector<ElementNode> nodes;  // producers before consumers
    std::vector<ElementEdge> edges;
    int root = -1;

    const ElementNode* find(int id) const;
    bool empty() const { return nodes.empty(); }
};

// `owners` maps elements to the authoring node they were built for; elements
// not listed get `defaultOwner`.
ElementGraph introspect(const std::shared_ptr<klartraum::ComputeGraphElement>& root,
                        const std::map<const klartraum::ComputeGraphElement*, int>& owners = {},
                        int defaultOwner = -1);

} // namespace kstudio
