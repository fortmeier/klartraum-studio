#include "studio/graph_model.hpp"

#include <algorithm>
#include <format>
#include <set>
#include <stdexcept>

namespace kstudio {

namespace {

constexpr PinDesc kSceneOutputs[] = {{"Gaussians", PinType::Gaussians}};
constexpr PinDesc kCameraOutputs[] = {{"Camera", PinType::Camera}};
constexpr PinDesc kTargetOutputs[] = {{"Image", PinType::Image}};
constexpr PinDesc kSplattingInputs[] = {
    {"Gaussians", PinType::Gaussians},
    {"Camera", PinType::Camera},
    {"Target", PinType::Image},
};
constexpr PinDesc kSplattingOutputs[] = {{"Image", PinType::Image}};
constexpr PinDesc kPresentInputs[] = {{"Image", PinType::Image}};

const NodeKindInfo kKinds[] = {
    {NodeKind::Scene, "scene", "Scene", "Loads a 3D Gaussian model from an .spz file.", {}, kSceneOutputs},
    {NodeKind::Camera, "camera", "Orbit Camera", "Camera uniform buffer driven by an orbit camera.", {}, kCameraOutputs},
    {NodeKind::SwapchainTarget, "swapchain_target", "Swapchain Target",
     "The window's swapchain images, rendered into directly.", {}, kTargetOutputs},
    {NodeKind::GaussianSplatting, "gaussian_splatting", "Gaussian Splatting",
     "Renders the Gaussians into the target image (klartraum::createGaussianSplatting).", kSplattingInputs,
     kSplattingOutputs},
    {NodeKind::Present, "present", "Present", "Presents the image in the window.", kPresentInputs, {}},
};

} // namespace

const NodeKindInfo& kindInfo(NodeKind kind) {
    for (const auto& info : kKinds) {
        if (info.kind == kind) {
            return info;
        }
    }
    throw std::logic_error("unknown node kind");
}

std::span<const NodeKindInfo> allKinds() {
    return kKinds;
}

std::optional<NodeKind> kindFromName(std::string_view name) {
    for (const auto& info : kKinds) {
        if (info.name == name) {
            return info.kind;
        }
    }
    return std::nullopt;
}

std::string_view pinTypeName(PinType type) {
    switch (type) {
    case PinType::Gaussians: return "Gaussians";
    case PinType::Camera: return "Camera";
    case PinType::Image: return "Image";
    }
    return "?";
}

std::string_view backendName(SplattingBackend backend) {
    return backend == SplattingBackend::Raster ? "raster" : "compute";
}

NodeParams defaultParams(NodeKind kind) {
    switch (kind) {
    case NodeKind::Scene: return SceneParams{};
    case NodeKind::Camera: return CameraParams{};
    case NodeKind::SwapchainTarget: return SwapchainTargetParams{};
    case NodeKind::GaussianSplatting: return SplattingParams{};
    case NodeKind::Present: return PresentParams{};
    }
    throw std::logic_error("unknown node kind");
}

int pinId(const PinRef& pin) {
    const int base = pin.direction == PinDirection::Input ? 0 : kMaxPinsPerDirection;
    return pin.node * (2 * kMaxPinsPerDirection) + base + pin.slot;
}

PinRef pinFromId(int id) {
    constexpr int stride = 2 * kMaxPinsPerDirection;
    const int local = id % stride;
    PinRef pin;
    pin.node = id / stride;
    pin.direction = local < kMaxPinsPerDirection ? PinDirection::Input : PinDirection::Output;
    pin.slot = local % kMaxPinsPerDirection;
    return pin;
}

int Graph::addNode(NodeKind kind, Vec2 position) {
    Node node;
    node.id = nextNodeId_;
    node.kind = kind;
    node.title = std::string(kindInfo(kind).title);
    node.params = defaultParams(kind);
    node.position = position;
    addNodeWithId(std::move(node));
    return nextNodeId_ - 1;
}

bool Graph::addNodeWithId(Node node) {
    if (node.id <= 0 || findNode(node.id)) {
        return false;
    }
    nextNodeId_ = std::max(nextNodeId_, node.id + 1);
    nodes_.push_back(std::move(node));
    touch();
    return true;
}

bool Graph::removeNode(int id) {
    auto it = std::find_if(nodes_.begin(), nodes_.end(), [id](const Node& n) { return n.id == id; });
    if (it == nodes_.end()) {
        return false;
    }
    nodes_.erase(it);
    std::erase_if(links_, [id](const Link& l) { return l.fromNode == id || l.toNode == id; });
    touch();
    return true;
}

std::optional<std::string> Graph::connect(PinRef from, PinRef to) {
    if (auto error = checkConnection(from, to)) {
        return error;
    }
    // Links always run output -> input.
    if (from.direction == PinDirection::Input) {
        std::swap(from, to);
    }
    std::erase_if(links_, [&](const Link& l) { return l.toNode == to.node && l.toSlot == to.slot; });
    links_.push_back(Link{nextLinkId_++, from.node, from.slot, to.node, to.slot});
    touch();
    return std::nullopt;
}

std::optional<std::string> Graph::checkConnection(PinRef from, PinRef to) const {
    // Accept the pins in either order.
    if (from.direction == PinDirection::Input && to.direction == PinDirection::Output) {
        std::swap(from, to);
    }
    if (from.direction != PinDirection::Output || to.direction != PinDirection::Input) {
        return "A link must connect an output to an input.";
    }
    const Node* src = findNode(from.node);
    const Node* dst = findNode(to.node);
    if (!src || !dst) {
        return "Unknown node.";
    }
    const auto& outputs = kindInfo(src->kind).outputs;
    const auto& inputs = kindInfo(dst->kind).inputs;
    if (from.slot < 0 || from.slot >= static_cast<int>(outputs.size()) || to.slot < 0 ||
        to.slot >= static_cast<int>(inputs.size())) {
        return "Unknown pin.";
    }
    if (outputs[from.slot].type != inputs[to.slot].type) {
        return std::format("Cannot connect {} to {}.", pinTypeName(outputs[from.slot].type),
                           pinTypeName(inputs[to.slot].type));
    }
    if (from.node == to.node || reaches(to.node, from.node)) {
        return "This link would create a cycle.";
    }
    return std::nullopt;
}

bool Graph::removeLink(int id) {
    const auto removed = std::erase_if(links_, [id](const Link& l) { return l.id == id; });
    if (removed > 0) {
        touch();
    }
    return removed > 0;
}

Node* Graph::findNode(int id) {
    auto it = std::find_if(nodes_.begin(), nodes_.end(), [id](const Node& n) { return n.id == id; });
    return it == nodes_.end() ? nullptr : &*it;
}

const Node* Graph::findNode(int id) const {
    return const_cast<Graph*>(this)->findNode(id);
}

const Link* Graph::findLink(int id) const {
    auto it = std::find_if(links_.begin(), links_.end(), [id](const Link& l) { return l.id == id; });
    return it == links_.end() ? nullptr : &*it;
}

const Link* Graph::inputLink(int node, int slot) const {
    auto it = std::find_if(links_.begin(), links_.end(),
                           [&](const Link& l) { return l.toNode == node && l.toSlot == slot; });
    return it == links_.end() ? nullptr : &*it;
}

const Node* Graph::inputNode(int node, int slot) const {
    const Link* link = inputLink(node, slot);
    return link ? findNode(link->fromNode) : nullptr;
}

bool Graph::reaches(int fromNode, int toNode) const {
    // Follows links downstream from `fromNode`.
    std::vector<int> stack{fromNode};
    std::set<int> visited;
    while (!stack.empty()) {
        const int current = stack.back();
        stack.pop_back();
        if (current == toNode) {
            return true;
        }
        if (!visited.insert(current).second) {
            continue;
        }
        for (const auto& link : links_) {
            if (link.fromNode == current) {
                stack.push_back(link.toNode);
            }
        }
    }
    return false;
}

std::vector<Diagnostic> Graph::validate() const {
    std::vector<Diagnostic> diagnostics;
    auto report = [&](Severity severity, int node, std::string message) {
        diagnostics.push_back(Diagnostic{severity, node, std::move(message)});
    };

    std::vector<const Node*> presents;
    for (const auto& node : nodes_) {
        if (node.kind == NodeKind::Present) {
            presents.push_back(&node);
        }
    }
    if (presents.empty()) {
        report(Severity::Error, -1, "The graph has no Present node, so nothing is rendered.");
    } else if (presents.size() > 1) {
        for (const Node* present : presents) {
            report(Severity::Error, present->id, "Only one Present node is supported.");
        }
    }

    for (const auto& node : nodes_) {
        const auto& inputs = kindInfo(node.kind).inputs;
        for (int slot = 0; slot < static_cast<int>(inputs.size()); ++slot) {
            if (!inputLink(node.id, slot)) {
                report(Severity::Error, node.id, std::format("Input '{}' is not connected.", inputs[slot].name));
            }
        }

        switch (node.kind) {
        case NodeKind::Scene:
            if (node.as<SceneParams>().path.empty()) {
                report(Severity::Error, node.id, "No scene file is set.");
            }
            break;
        case NodeKind::GaussianSplatting: {
            const Node* target = inputNode(node.id, 2);
            if (target && target->kind != NodeKind::SwapchainTarget) {
                report(Severity::Error, node.id,
                       "The target must be a Swapchain Target: klartraum's splatting backends render into "
                       "swapchain images and do not composite onto another node's output.");
            }
            const auto& params = node.as<SplattingParams>();
            if (params.backend == SplattingBackend::Compute && (params.splatTileX == 0 || params.splatTileY == 0)) {
                report(Severity::Error, node.id, "Splat tile size must not be zero.");
            }
            break;
        }
        case NodeKind::Present: {
            const Node* source = inputNode(node.id, 0);
            if (source && source->kind != NodeKind::GaussianSplatting) {
                report(Severity::Error, node.id, "Present needs a rendered image, e.g. from Gaussian Splatting.");
            }
            break;
        }
        default:
            break;
        }
    }

    // Nodes that do not feed the presented image are not compiled.
    if (presents.size() == 1) {
        for (const auto& node : nodes_) {
            if (node.id != presents.front()->id && !reaches(node.id, presents.front()->id)) {
                report(Severity::Info, node.id, "Not connected to Present; it is not compiled.");
            }
        }
    }
    return diagnostics;
}

bool Graph::hasErrors() const {
    const auto diagnostics = validate();
    return std::any_of(diagnostics.begin(), diagnostics.end(),
                       [](const Diagnostic& d) { return d.severity == Severity::Error; });
}

void Graph::clear() {
    nodes_.clear();
    links_.clear();
    nextNodeId_ = 1;
    nextLinkId_ = 1;
    touch();
}

Graph makeGaussianSplattingGraph(const std::string& scenePath, SplattingBackend backend) {
    Graph graph;
    const int scene = graph.addNode(NodeKind::Scene, {40.0f, 20.0f});
    const int camera = graph.addNode(NodeKind::Camera, {40.0f, 170.0f});
    const int target = graph.addNode(NodeKind::SwapchainTarget, {40.0f, 300.0f});
    const int splatting = graph.addNode(NodeKind::GaussianSplatting, {340.0f, 130.0f});
    const int present = graph.addNode(NodeKind::Present, {640.0f, 170.0f});

    graph.findNode(scene)->as<SceneParams>().path = scenePath;
    graph.findNode(splatting)->as<SplattingParams>().backend = backend;

    auto out = [](int node) { return PinRef{node, PinDirection::Output, 0}; };
    auto in = [](int node, int slot) { return PinRef{node, PinDirection::Input, slot}; };
    graph.connect(out(scene), in(splatting, 0));
    graph.connect(out(camera), in(splatting, 1));
    graph.connect(out(target), in(splatting, 2));
    graph.connect(out(splatting), in(present, 0));
    return graph;
}

} // namespace kstudio
