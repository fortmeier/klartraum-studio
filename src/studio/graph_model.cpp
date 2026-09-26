#include "studio/graph_model.hpp"

#include <algorithm>
#include <format>
#include <set>
#include <stdexcept>

namespace kstudio {

namespace {

constexpr PinDesc kSceneOutputs[] = {{"Gaussians", PinType::Gaussians}};
constexpr PinDesc kCameraOutputs[] = {{"Camera", PinType::Camera}};
constexpr PinDesc kImageOutputs[] = {{"Image", PinType::Image}};
constexpr PinDesc kSplattingInputs[] = {
    {"Gaussians", PinType::Gaussians},
    {"Camera", PinType::Camera},
    {"Target", PinType::Image},
};
constexpr PinDesc kImageInputs[] = {{"Image", PinType::Image}};
constexpr PinDesc kTensorInputs[] = {{"Tensor", PinType::Tensor}};
constexpr PinDesc kTensorOutputs[] = {{"Tensor", PinType::Tensor}};
constexpr PinDesc kSinkInputs[] = {{"Tensor", PinType::Tensor, PinType::Image}};

const NodeKindInfo kKinds[] = {
    {NodeKind::Scene, "scene", "Scene", "Sources", "Loads a 3D Gaussian model from an .spz file.", {}, kSceneOutputs},
    {NodeKind::ImageFile, "image_file", "Image File", "Sources",
     "Loads an image file (PNG, JPEG, ...) as a 1x3xHxW tensor with values in [0, 1].", {}, kTensorOutputs},
    {NodeKind::Camera, "camera", "Orbit Camera", "Rendering", "Camera uniform buffer driven by an orbit camera.", {},
     kCameraOutputs},
    {NodeKind::SwapchainTarget, "swapchain_target", "Swapchain Target", "Rendering",
     "The window's swapchain images, rendered into directly.", {}, kImageOutputs},
    {NodeKind::OffscreenTarget, "offscreen_target", "Offscreen Target", "Rendering",
     "An image of fixed size to render into for further processing.", {}, kImageOutputs},
    {NodeKind::GaussianSplatting, "gaussian_splatting", "Gaussian Splatting", "Rendering",
     "Renders the Gaussians into the target image (klartraum::createGaussianSplatting).", kSplattingInputs,
     kImageOutputs},
    {NodeKind::ImageToTensor, "image_to_tensor", "Image to Tensor", "Compute",
     "Converts a rendered offscreen image into a 1x3xHxW tensor.", kImageInputs, kTensorOutputs},
    {NodeKind::TensorToImage, "tensor_to_image", "Tensor to Image", "Compute",
     "Converts a 1x3xHxW tensor (values in [0, 1]) into an HxW image.", kTensorInputs, kImageOutputs},
    {NodeKind::Resample, "resample", "Resample", "Compute",
     "Resamples an image to a fixed size (klartraum::ImageResample).", kImageInputs, kImageOutputs},
    {NodeKind::OnnxModel, "onnx_model", "ONNX Model", "Compute",
     "Runs an ONNX network (klartraum::OnnxNetwork) on a tensor.", kTensorInputs, kTensorOutputs},
    {NodeKind::Present, "present", "Present", "Outputs",
     "Shows the image in the window every frame, stretched to the window's size.", kImageInputs, {}},
    {NodeKind::Preview, "preview", "Preview", "Outputs",
     "Shows a 1- or 3-channel image tensor, or an offscreen image, when the graph is run.", kSinkInputs, {}},
    {NodeKind::ImageFileWriter, "image_file_writer", "Image File Writer", "Outputs",
     "Writes a 1- or 3-channel image tensor, or an offscreen image, to a PNG file when the graph is run.",
     kSinkInputs, {}},
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
    case PinType::Tensor: return "Tensor";
    }
    return "?";
}

std::string_view backendName(SplattingBackend backend) {
    return backend == SplattingBackend::Raster ? "raster" : "compute";
}

std::string_view filterName(ResampleFilter filter) {
    return filter == ResampleFilter::Nearest ? "nearest" : "bilinear";
}

NodeParams defaultParams(NodeKind kind) {
    switch (kind) {
    case NodeKind::Scene: return SceneParams{};
    case NodeKind::Camera: return CameraParams{};
    case NodeKind::SwapchainTarget: return SwapchainTargetParams{};
    case NodeKind::GaussianSplatting: return SplattingParams{};
    case NodeKind::Present: return PresentParams{};
    case NodeKind::OffscreenTarget: return OffscreenTargetParams{};
    case NodeKind::ImageFile: return ImageFileParams{};
    case NodeKind::ImageToTensor: return ImageToTensorParams{};
    case NodeKind::TensorToImage: return TensorToImageParams{};
    case NodeKind::Resample: return ResampleParams{};
    case NodeKind::OnnxModel: return OnnxModelParams{};
    case NodeKind::Preview: return PreviewParams{};
    case NodeKind::ImageFileWriter: return ImageFileWriterParams{};
    }
    throw std::logic_error("unknown node kind");
}

bool isSink(NodeKind kind) {
    return kind == NodeKind::Preview || kind == NodeKind::ImageFileWriter;
}

bool producesImage(NodeKind kind) {
    return kind == NodeKind::GaussianSplatting || kind == NodeKind::TensorToImage || kind == NodeKind::Resample;
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
    if (!inputs[to.slot].accepts(outputs[from.slot].type)) {
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

std::optional<PinType> Graph::inputType(int node, int slot) const {
    const Link* link = inputLink(node, slot);
    const Node* source = link ? findNode(link->fromNode) : nullptr;
    if (!source) {
        return std::nullopt;
    }
    return kindInfo(source->kind).outputs[link->fromSlot].type;
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

std::vector<int> Graph::upstreamOf(int node) const {
    std::vector<int> result;
    std::set<int> visited;
    std::vector<int> stack{node};
    while (!stack.empty()) {
        const int current = stack.back();
        stack.pop_back();
        if (!visited.insert(current).second) {
            continue;
        }
        result.push_back(current);
        for (const auto& link : links_) {
            if (link.toNode == current) {
                stack.push_back(link.fromNode);
            }
        }
    }
    return result;
}

std::vector<int> Graph::topologicalOrder() const {
    // connect() keeps the graph acyclic, so repeatedly taking the first node
    // whose producers are all placed terminates.
    std::vector<int> order;
    std::set<int> placed;
    while (order.size() < nodes_.size()) {
        for (const auto& node : nodes_) {
            if (placed.contains(node.id)) {
                continue;
            }
            const bool ready = std::all_of(links_.begin(), links_.end(), [&](const Link& l) {
                return l.toNode != node.id || placed.contains(l.fromNode);
            });
            if (ready) {
                order.push_back(node.id);
                placed.insert(node.id);
                break;
            }
        }
    }
    return order;
}

std::vector<Diagnostic> Graph::validate() const {
    std::vector<Diagnostic> diagnostics;
    auto report = [&](Severity severity, int node, std::string message) {
        diagnostics.push_back(Diagnostic{severity, node, std::move(message)});
    };

    std::vector<const Node*> presents;
    std::vector<const Node*> sinks;
    for (const auto& node : nodes_) {
        if (node.kind == NodeKind::Present) {
            presents.push_back(&node);
        } else if (isSink(node.kind)) {
            sinks.push_back(&node);
        }
    }
    if (presents.empty() && sinks.empty()) {
        report(Severity::Error, -1,
               "Nothing to do: add a Present node to render every frame, or a Preview or Image File Writer to run "
               "the graph.");
    }
    if (presents.size() > 1) {
        for (const Node* present : presents) {
            report(Severity::Error, present->id, "Only one Present node is supported.");
        }
    }

    // Targets are empty images; what reads an image needs one with a result.
    auto checkRenderedImage = [&](const Node& node) {
        const Node* source = inputNode(node.id, 0);
        if (inputType(node.id, 0) == PinType::Image && !producesImage(source->kind)) {
            report(Severity::Error, node.id,
                   "Needs a rendered image, e.g. from Gaussian Splatting, Tensor to Image or Resample.");
        }
    };

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
            if (target && target->kind != NodeKind::SwapchainTarget && target->kind != NodeKind::OffscreenTarget) {
                report(Severity::Error, node.id,
                       "The target must be a Swapchain Target or an Offscreen Target: klartraum's splatting "
                       "backends render into an image and do not composite onto another node's output.");
            }
            const auto& params = node.as<SplattingParams>();
            if (params.backend == SplattingBackend::Compute && (params.splatTileX == 0 || params.splatTileY == 0)) {
                report(Severity::Error, node.id, "Splat tile size must not be zero.");
            }
            break;
        }
        case NodeKind::Present:
        case NodeKind::ImageToTensor:
        case NodeKind::Resample:
            checkRenderedImage(node);
            if (node.kind == NodeKind::Resample) {
                const auto& p = node.as<ResampleParams>();
                if (p.width == 0 || p.height == 0) {
                    report(Severity::Error, node.id, "Width and height must not be zero.");
                }
            }
            break;
        case NodeKind::Preview:
            checkRenderedImage(node);
            break;
        case NodeKind::OffscreenTarget: {
            const auto& p = node.as<OffscreenTargetParams>();
            if (p.width == 0 || p.height == 0) {
                report(Severity::Error, node.id, "Width and height must not be zero.");
            }
            break;
        }
        case NodeKind::ImageFile: {
            const auto& p = node.as<ImageFileParams>();
            if (p.path.empty()) {
                report(Severity::Error, node.id, "No image file is set.");
            }
            if (p.width == 0 || p.height == 0) {
                report(Severity::Error, node.id, "Width and height must not be zero.");
            }
            break;
        }
        case NodeKind::OnnxModel:
            if (node.as<OnnxModelParams>().path.empty()) {
                report(Severity::Error, node.id, "No ONNX model file is set.");
            }
            break;
        case NodeKind::ImageFileWriter:
            if (node.as<ImageFileWriterParams>().path.empty()) {
                report(Severity::Error, node.id, "No output file is set.");
            }
            checkRenderedImage(node);
            break;
        default:
            break;
        }
    }

    // Run executes once, outside the window's frame loop, which owns the
    // swapchain images.
    for (const Node* sink : sinks) {
        const auto upstream = upstreamOf(sink->id);
        if (std::any_of(upstream.begin(), upstream.end(),
                        [&](int id) { return findNode(id)->kind == NodeKind::SwapchainTarget; })) {
            report(Severity::Error, sink->id,
                   "Run cannot use the window's swapchain images; render into an Offscreen Target for it.");
        }
    }

    // Nodes that feed neither Present nor a sink are not compiled.
    std::set<int> used;
    for (const Node* output : presents) {
        for (int id : upstreamOf(output->id)) {
            used.insert(id);
        }
    }
    for (const Node* sink : sinks) {
        for (int id : upstreamOf(sink->id)) {
            used.insert(id);
        }
    }
    if (!presents.empty() || !sinks.empty()) {
        for (const auto& node : nodes_) {
            if (!used.contains(node.id)) {
                report(Severity::Info, node.id, "Not connected to Present or a sink; it is not compiled.");
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

namespace {

PinRef out(int node, int slot = 0) { return {node, PinDirection::Output, slot}; }
PinRef in(int node, int slot = 0) { return {node, PinDirection::Input, slot}; }

} // namespace

Graph makeAutoencoderGraph(const std::string& imagePath, const std::string& encoderPath,
                           const std::string& decoderPath, const std::string& outputPath) {
    Graph graph;
    const int image = graph.addNode(NodeKind::ImageFile, {0.0f, 60.0f});
    const int encoder = graph.addNode(NodeKind::OnnxModel, {310.0f, 60.0f});
    const int decoder = graph.addNode(NodeKind::OnnxModel, {620.0f, 60.0f});
    const int preview = graph.addNode(NodeKind::Preview, {930.0f, 0.0f});
    const int writer = graph.addNode(NodeKind::ImageFileWriter, {930.0f, 260.0f});

    graph.findNode(image)->as<ImageFileParams>().path = imagePath;
    graph.findNode(encoder)->as<OnnxModelParams>().path = encoderPath;
    graph.findNode(encoder)->title = "Encoder";
    graph.findNode(decoder)->as<OnnxModelParams>().path = decoderPath;
    graph.findNode(decoder)->title = "Decoder";
    graph.findNode(writer)->as<ImageFileWriterParams>().path = outputPath;

    graph.connect(out(image), in(encoder));
    graph.connect(out(encoder), in(decoder));
    graph.connect(out(decoder), in(preview));
    graph.connect(out(decoder), in(writer));
    return graph;
}

Graph makeSplatAutoencoderGraph(const std::string& scenePath, const std::string& encoderPath,
                                const std::string& decoderPath) {
    Graph graph;
    const int scene = graph.addNode(NodeKind::Scene, {0.0f, 0.0f});
    const int camera = graph.addNode(NodeKind::Camera, {0.0f, 150.0f});
    const int target = graph.addNode(NodeKind::OffscreenTarget, {0.0f, 300.0f});
    const int splatting = graph.addNode(NodeKind::GaussianSplatting, {310.0f, 120.0f});
    const int toTensor = graph.addNode(NodeKind::ImageToTensor, {620.0f, 150.0f});
    const int encoder = graph.addNode(NodeKind::OnnxModel, {930.0f, 150.0f});
    const int decoder = graph.addNode(NodeKind::OnnxModel, {1240.0f, 150.0f});
    const int preview = graph.addNode(NodeKind::Preview, {1550.0f, 100.0f});

    graph.findNode(scene)->as<SceneParams>().path = scenePath;
    graph.findNode(encoder)->as<OnnxModelParams>().path = encoderPath;
    graph.findNode(encoder)->title = "Encoder";
    graph.findNode(decoder)->as<OnnxModelParams>().path = decoderPath;
    graph.findNode(decoder)->title = "Decoder";

    graph.connect(out(scene), in(splatting, 0));
    graph.connect(out(camera), in(splatting, 1));
    graph.connect(out(target), in(splatting, 2));
    graph.connect(out(splatting), in(toTensor));
    graph.connect(out(toTensor), in(encoder));
    graph.connect(out(encoder), in(decoder));
    graph.connect(out(decoder), in(preview));
    return graph;
}

} // namespace kstudio
