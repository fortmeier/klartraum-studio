#include "studio/graph_model.hpp"

#include <algorithm>
#include <format>
#include <set>
#include <stdexcept>

namespace kstudio {

namespace {

constexpr PinDesc kCpuGaussiansOutputs[] = {{"Gaussians", PinType::GaussiansCpu}};
constexpr PinDesc kTransformInputs[] = {{"In", PinType::GaussiansCpu}};
constexpr PinDesc kTransformOutputs[] = {{"Out", PinType::GaussiansCpu}};
constexpr PinDesc kUploadInputs[] = {{"CPU", PinType::GaussiansCpu}};
constexpr PinDesc kMergeInputs[] = {{"A", PinType::GaussiansCpu}, {"B", PinType::GaussiansCpu}};
constexpr PinDesc kUploadOutputs[] = {{"GPU", PinType::GaussiansGpu}};
constexpr PinDesc kGpuMergeInputs[] = {{"A", PinType::GaussiansGpu}, {"B", PinType::GaussiansGpu}};
constexpr PinDesc kGpuTransformInputs[] = {{"Gaussians", PinType::GaussiansGpu}, {"Transform", PinType::TransformGpu}};
constexpr PinDesc kGpuGaussiansOutputs[] = {{"Gaussians", PinType::GaussiansGpu}};
constexpr PinDesc kNumberOutputs[] = {{"Value", PinType::NumberCpu}};
constexpr PinDesc kSineInputs[] = {{"X", PinType::NumberCpu}};
constexpr PinDesc kUploadNumberInputs[] = {{"CPU", PinType::NumberCpu}};
constexpr PinDesc kUploadNumberOutputs[] = {{"GPU", PinType::NumberGpu}};
constexpr PinDesc kMakeTransformInputs[] = {
    {"X", PinType::NumberGpu, std::nullopt, true},     {"Y", PinType::NumberGpu, std::nullopt, true},
    {"Z", PinType::NumberGpu, std::nullopt, true},     {"Pitch", PinType::NumberGpu, std::nullopt, true},
    {"Yaw", PinType::NumberGpu, std::nullopt, true},   {"Roll", PinType::NumberGpu, std::nullopt, true},
    {"Scale", PinType::NumberGpu, std::nullopt, true},
};
constexpr PinDesc kMakeTransformOutputs[] = {{"Transform", PinType::TransformGpu}};
constexpr PinDesc kCameraOutputs[] = {{"Camera", PinType::Camera}};
constexpr PinDesc kImageOutputs[] = {{"Image", PinType::Image}};
constexpr PinDesc kSplattingInputs[] = {
    {"Gaussians", PinType::GaussiansGpu},
    {"Camera", PinType::Camera},
    {"Target", PinType::Image},
};
constexpr PinDesc kImageInputs[] = {{"Image", PinType::Image}};
constexpr PinDesc kTensorInputs[] = {{"Tensor", PinType::Tensor}};
constexpr PinDesc kTensorOutputs[] = {{"Tensor", PinType::Tensor}};
constexpr PinDesc kSinkInputs[] = {{"Tensor", PinType::Tensor, PinType::Image}};

using enum ExecutionSite;
using enum Implementation;

constexpr std::string_view kOnBuild = "once, when the graph is built";
constexpr std::string_view kEveryExecution = "every frame (live) or every run";

const NodeKindInfo kKinds[] = {
    {NodeKind::Scene, "scene", "Scene", "Sources", "Loads a 3D Gaussian model from an .spz file into CPU memory.", {},
     kCpuGaussiansOutputs, Cpu, KlartraumFunction, "klartraum::loadGaussiansSpz", kOnBuild},
    {NodeKind::ImageFile, "image_file", "Image File", "Sources",
     "Loads an image file (PNG, JPEG, ...) as a 1x3xHxW tensor with values in [0, 1].", {}, kTensorOutputs, Upload,
     ComputeGraph, "decoded by the studio (stb_image), uploaded into a klartraum::TensorElement", kOnBuild},
    {NodeKind::Number, "number", "Number", "Numbers (CPU)", "A constant number.", {}, kNumberOutputs, Cpu, Studio,
     "evaluated by the studio (kstudio::evaluateNumber)", "every frame (live), once per run", true},
    {NodeKind::Time, "time", "Time", "Numbers (CPU)", "Seconds since the studio started, times a speed.", {},
     kNumberOutputs, Cpu, Studio, "evaluated by the studio (kstudio::evaluateNumber)",
     "every frame (live), once per run", true},
    {NodeKind::Sine, "sine", "Sine", "Numbers (CPU)",
     "amplitude * sin(frequency * 2 pi * x + phase) + offset, e.g. to swing something back and forth over time.",
     kSineInputs, kNumberOutputs, Cpu, Studio, "evaluated by the studio (kstudio::evaluateNumber)",
     "every frame (live), once per run", true},
    {NodeKind::UploadNumber, "upload_number", "Upload Number", "Numbers (CPU)",
     "Copies a CPU number into a GPU buffer before every frame, without rebuilding the graph.", kUploadNumberInputs,
     kUploadNumberOutputs, Upload, ComputeGraph, "klartraum::HostValues", "every frame (live), once per run"},
    {NodeKind::TransformGaussians, "transform_gaussians", "Transform (CPU)", "Gaussians (CPU)",
     "Scales, rotates and moves Gaussians in CPU memory.", kTransformInputs, kTransformOutputs, Cpu,
     KlartraumFunction, "klartraum::transformGaussians", kOnBuild},
    {NodeKind::MergeGaussians, "merge_gaussians", "Merge (CPU)", "Gaussians (CPU)",
     "Combines two sets of Gaussians in CPU memory into one.", kMergeInputs, kCpuGaussiansOutputs, Cpu, Studio,
     "kstudio::assembleGaussians", kOnBuild},
    {NodeKind::UploadGaussians, "upload_gaussians", "Upload Gaussians", "Gaussians (CPU)",
     "Uploads Gaussians into the GPU buffers a Gaussian Splatting reads.", kUploadInputs, kUploadOutputs,
     Upload, ComputeGraph, "klartraum::GaussianDataStandard (7 buffer elements)", kOnBuild},
    {NodeKind::MakeTransform, "make_transform", "Make Transform", "Gaussians (GPU)",
     "A GPU transform from translation, rotation (degrees about X, then Y, then Z) and scale. Inputs that are "
     "not connected take the node's values, which apply without rebuilding.",
     kMakeTransformInputs, kMakeTransformOutputs, Gpu, ComputeGraph, "klartraum::TransformBufferPass",
     kEveryExecution, true},
    {NodeKind::TransformGaussiansGpu, "transform_gaussians_gpu", "Transform (GPU)", "Gaussians (GPU)",
     "Moves Gaussians on the GPU by a transform, every frame.", kGpuTransformInputs, kGpuGaussiansOutputs, Gpu,
     ComputeGraph, "klartraum::GaussianTransformPass", kEveryExecution},
    {NodeKind::MergeGaussiansGpu, "merge_gaussians_gpu", "Merge (GPU)", "Gaussians (GPU)",
     "Combines two sets of Gaussians on the GPU, every frame.", kGpuMergeInputs, kGpuGaussiansOutputs, Gpu,
     ComputeGraph, "klartraum::GaussianMergePass", kEveryExecution},
    {NodeKind::Camera, "camera", "Orbit Camera", "Rendering", "Camera uniform buffer driven by an orbit camera.", {},
     kCameraOutputs, Upload, ComputeGraph, "klartraum::InterfaceCameraOrbit writing a klartraum::CameraUboType",
     "every frame (live), once per run", true},
    {NodeKind::SwapchainTarget, "swapchain_target", "Swapchain Target", "Rendering",
     "The window's swapchain images, rendered into directly.", {}, kImageOutputs, Gpu, ComputeGraph,
     "klartraum::ImageViewSrc over the swapchain", "every frame (live only)"},
    {NodeKind::OffscreenTarget, "offscreen_target", "Offscreen Target", "Rendering",
     "An image of fixed size to render into for further processing.", {}, kImageOutputs, Gpu, ComputeGraph,
     "klartraum::OffscreenTarget", kEveryExecution},
    {NodeKind::GaussianSplatting, "gaussian_splatting", "Gaussian Splatting", "Rendering",
     "Renders the Gaussians into the target image.", kSplattingInputs, kImageOutputs, Gpu, ComputeGraph,
     "klartraum::createGaussianSplatting", kEveryExecution},
    {NodeKind::ImageToTensor, "image_to_tensor", "Image to Tensor", "Compute",
     "Converts a rendered image into a 1x3xHxW tensor.", kImageInputs, kTensorOutputs, Gpu, ComputeGraph,
     "klartraum::GeneralComputation (image_to_tensor.comp)", kEveryExecution},
    {NodeKind::TensorToImage, "tensor_to_image", "Tensor to Image", "Compute",
     "Converts a 1x3xHxW tensor (values in [0, 1]) into an HxW image.", kTensorInputs, kImageOutputs, Gpu,
     ComputeGraph, "klartraum::GeneralComputation (tensor_to_image.comp)", kEveryExecution},
    {NodeKind::Resample, "resample", "Resample", "Compute", "Resamples an image to a fixed size.", kImageInputs,
     kImageOutputs, Gpu, ComputeGraph, "klartraum::ImageResample", kEveryExecution},
    {NodeKind::OnnxModel, "onnx_model", "ONNX Model", "Compute", "Runs an ONNX network on a tensor.", kTensorInputs,
     kTensorOutputs, Gpu, ComputeGraph, "klartraum::OnnxNetwork", kEveryExecution},
    {NodeKind::Present, "present", "Present", "Outputs",
     "Shows the image in the window every frame, stretched to the window's size.", kImageInputs, {}, Gpu,
     ComputeGraph, "klartraum::ImageResample and ImageViewSrcTransition into the swapchain, added by the studio",
     "every frame (live only)"},
    {NodeKind::Preview, "preview", "Preview", "Outputs",
     "Shows a 1- or 3-channel image tensor, or an image, when the graph is run.", kSinkInputs, {}, Readback, Studio,
     "read back by the studio, shown as an ImGui texture", "every run"},
    {NodeKind::ImageFileWriter, "image_file_writer", "Image File Writer", "Outputs",
     "Writes a 1- or 3-channel image tensor, or an image, to a PNG file when the graph is run.", kSinkInputs, {},
     Readback, Studio, "read back by the studio, written with stb_image_write", "every run"},
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
    case PinType::GaussiansCpu: return "Gaussians (CPU)";
    case PinType::GaussiansGpu: return "Gaussians (GPU)";
    case PinType::NumberCpu: return "Number (CPU)";
    case PinType::NumberGpu: return "Number (GPU)";
    case PinType::TransformGpu: return "Transform (GPU)";
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
    case NodeKind::TransformGaussians: return TransformGaussiansParams{};
    case NodeKind::MergeGaussians: return MergeGaussiansParams{};
    case NodeKind::UploadGaussians: return UploadGaussiansParams{};
    case NodeKind::Number: return NumberParams{};
    case NodeKind::Time: return TimeParams{};
    case NodeKind::Sine: return SineParams{};
    case NodeKind::UploadNumber: return UploadNumberParams{};
    case NodeKind::MakeTransform: return MakeTransformParams{};
    case NodeKind::TransformGaussiansGpu: return TransformGaussiansGpuParams{};
    case NodeKind::MergeGaussiansGpu: return MergeGaussiansGpuParams{};
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

std::string_view siteName(ExecutionSite site) {
    switch (site) {
    case ExecutionSite::Gpu: return "GPU";
    case ExecutionSite::Cpu: return "CPU";
    case ExecutionSite::Upload: return "CPU->GPU";
    case ExecutionSite::Readback: return "GPU->CPU";
    }
    return "?";
}

std::string_view implementationName(Implementation implementation) {
    switch (implementation) {
    case Implementation::ComputeGraph: return "klartraum graph";
    case Implementation::KlartraumFunction: return "klartraum function";
    case Implementation::Studio: return "studio";
    }
    return "?";
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
        std::string message = std::format("Cannot connect {} to {}.", pinTypeName(outputs[from.slot].type),
                                          pinTypeName(inputs[to.slot].type));
        if (outputs[from.slot].type == PinType::GaussiansCpu && inputs[to.slot].type == PinType::GaussiansGpu) {
            message += " Put an Upload Gaussians node in between.";
        } else if (outputs[from.slot].type == PinType::NumberCpu && inputs[to.slot].type == PinType::NumberGpu) {
            message += " Put an Upload Number node in between.";
        }
        return message;
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
            if (!inputs[slot].optional && !inputLink(node.id, slot)) {
                report(Severity::Error, node.id, std::format("Input '{}' is not connected.", inputs[slot].name));
            }
        }

        switch (node.kind) {
        case NodeKind::Scene:
            if (node.as<SceneParams>().path.empty()) {
                report(Severity::Error, node.id, "No scene file is set.");
            }
            break;
        case NodeKind::TransformGaussians:
            if (!(node.as<TransformGaussiansParams>().scale > 0.0f)) {
                report(Severity::Error, node.id, "Scale must be greater than zero.");
            }
            break;
        case NodeKind::MakeTransform:
            if (!inputLink(node.id, 6) && !(node.as<MakeTransformParams>().scale > 0.0f)) {
                report(Severity::Error, node.id, "Scale must be greater than zero.");
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
    const int scene = graph.addNode(NodeKind::Scene, {0.0f, 0.0f});
    const int upload = graph.addNode(NodeKind::UploadGaussians, {300.0f, 0.0f});
    const int camera = graph.addNode(NodeKind::Camera, {300.0f, 170.0f});
    const int target = graph.addNode(NodeKind::SwapchainTarget, {300.0f, 320.0f});
    const int splatting = graph.addNode(NodeKind::GaussianSplatting, {600.0f, 130.0f});
    const int present = graph.addNode(NodeKind::Present, {900.0f, 170.0f});

    graph.findNode(scene)->as<SceneParams>().path = scenePath;
    graph.findNode(splatting)->as<SplattingParams>().backend = backend;

    auto out = [](int node) { return PinRef{node, PinDirection::Output, 0}; };
    auto in = [](int node, int slot) { return PinRef{node, PinDirection::Input, slot}; };
    graph.connect(out(scene), in(upload, 0));
    graph.connect(out(upload), in(splatting, 0));
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

Graph makeCombinedScenesGraph(const std::string& scenePath, const SceneParams& movedScene,
                              const TransformGaussiansParams& transform, const CameraParams& cameraParams) {
    Graph graph;
    const int scene = graph.addNode(NodeKind::Scene, {0.0f, 0.0f});
    const int moved = graph.addNode(NodeKind::Scene, {0.0f, 160.0f});
    const int move = graph.addNode(NodeKind::TransformGaussians, {300.0f, 160.0f});
    const int merge = graph.addNode(NodeKind::MergeGaussians, {600.0f, 60.0f});
    const int upload = graph.addNode(NodeKind::UploadGaussians, {900.0f, 60.0f});
    const int camera = graph.addNode(NodeKind::Camera, {900.0f, 230.0f});
    const int target = graph.addNode(NodeKind::SwapchainTarget, {900.0f, 380.0f});
    const int splatting = graph.addNode(NodeKind::GaussianSplatting, {1200.0f, 170.0f});
    const int present = graph.addNode(NodeKind::Present, {1500.0f, 210.0f});

    graph.findNode(scene)->as<SceneParams>().path = scenePath;
    graph.findNode(moved)->as<SceneParams>() = movedScene;
    graph.findNode(move)->as<TransformGaussiansParams>() = transform;
    graph.findNode(camera)->as<CameraParams>() = cameraParams;

    graph.connect(out(scene), in(merge, 0));
    graph.connect(out(moved), in(move));
    graph.connect(out(move), in(merge, 1));
    graph.connect(out(merge), in(upload));
    graph.connect(out(upload), in(splatting, 0));
    graph.connect(out(camera), in(splatting, 1));
    graph.connect(out(target), in(splatting, 2));
    graph.connect(out(splatting), in(present));
    return graph;
}

Graph makeAnimatedScenesGraph(const std::string& scenePath, const SceneParams& movedScene,
                              const MakeTransformParams& placement, const SineParams& swing,
                              const CameraParams& cameraParams) {
    Graph graph;
    const int scene = graph.addNode(NodeKind::Scene, {0.0f, 0.0f});
    const int moved = graph.addNode(NodeKind::Scene, {0.0f, 200.0f});
    const int time = graph.addNode(NodeKind::Time, {0.0f, 420.0f});
    const int upload = graph.addNode(NodeKind::UploadGaussians, {300.0f, 0.0f});
    const int uploadMoved = graph.addNode(NodeKind::UploadGaussians, {300.0f, 200.0f});
    const int sine = graph.addNode(NodeKind::Sine, {300.0f, 420.0f});
    const int uploadYaw = graph.addNode(NodeKind::UploadNumber, {600.0f, 420.0f});
    const int transform = graph.addNode(NodeKind::MakeTransform, {900.0f, 300.0f});
    const int move = graph.addNode(NodeKind::TransformGaussiansGpu, {1200.0f, 200.0f});
    const int merge = graph.addNode(NodeKind::MergeGaussiansGpu, {1500.0f, 60.0f});
    const int camera = graph.addNode(NodeKind::Camera, {1500.0f, 260.0f});
    const int target = graph.addNode(NodeKind::SwapchainTarget, {1500.0f, 440.0f});
    const int splatting = graph.addNode(NodeKind::GaussianSplatting, {1800.0f, 200.0f});
    const int present = graph.addNode(NodeKind::Present, {2100.0f, 240.0f});

    graph.findNode(scene)->as<SceneParams>().path = scenePath;
    graph.findNode(moved)->as<SceneParams>() = movedScene;
    graph.findNode(sine)->as<SineParams>() = swing;
    graph.findNode(sine)->title = "Swing";
    graph.findNode(transform)->as<MakeTransformParams>() = placement;
    graph.findNode(camera)->as<CameraParams>() = cameraParams;

    graph.connect(out(scene), in(upload));
    graph.connect(out(moved), in(uploadMoved));
    graph.connect(out(time), in(sine));
    graph.connect(out(sine), in(uploadYaw));
    graph.connect(out(uploadYaw), in(transform, 4));  // yaw
    graph.connect(out(uploadMoved), in(move, 0));
    graph.connect(out(transform), in(move, 1));
    graph.connect(out(upload), in(merge, 0));
    graph.connect(out(move), in(merge, 1));
    graph.connect(out(merge), in(splatting, 0));
    graph.connect(out(camera), in(splatting, 1));
    graph.connect(out(target), in(splatting, 2));
    graph.connect(out(splatting), in(present));
    return graph;
}

Graph makeSplatAutoencoderGraph(const std::string& scenePath, const std::string& encoderPath,
                                const std::string& decoderPath) {
    Graph graph;
    const int scene = graph.addNode(NodeKind::Scene, {0.0f, 0.0f});
    const int upload = graph.addNode(NodeKind::UploadGaussians, {310.0f, 0.0f});
    const int camera = graph.addNode(NodeKind::Camera, {310.0f, 150.0f});
    const int target = graph.addNode(NodeKind::OffscreenTarget, {310.0f, 300.0f});
    const int splatting = graph.addNode(NodeKind::GaussianSplatting, {620.0f, 120.0f});
    const int toTensor = graph.addNode(NodeKind::ImageToTensor, {930.0f, 150.0f});
    const int encoder = graph.addNode(NodeKind::OnnxModel, {1240.0f, 150.0f});
    const int decoder = graph.addNode(NodeKind::OnnxModel, {1550.0f, 150.0f});
    const int preview = graph.addNode(NodeKind::Preview, {1860.0f, 100.0f});

    graph.findNode(scene)->as<SceneParams>().path = scenePath;
    graph.findNode(encoder)->as<OnnxModelParams>().path = encoderPath;
    graph.findNode(encoder)->title = "Encoder";
    graph.findNode(decoder)->as<OnnxModelParams>().path = decoderPath;
    graph.findNode(decoder)->title = "Decoder";

    graph.connect(out(scene), in(upload));
    graph.connect(out(upload), in(splatting, 0));
    graph.connect(out(camera), in(splatting, 1));
    graph.connect(out(target), in(splatting, 2));
    graph.connect(out(splatting), in(toTensor));
    graph.connect(out(toTensor), in(encoder));
    graph.connect(out(encoder), in(decoder));
    graph.connect(out(decoder), in(preview));
    return graph;
}

} // namespace kstudio
