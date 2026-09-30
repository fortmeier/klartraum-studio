#include "studio/graph_model.hpp"

#include <nlohmann/json.hpp>

#include "studio/meta_nodes.hpp"

#include <algorithm>
#include <filesystem>
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
constexpr PinDesc kBinaryLayerInputs[] = {{"A", PinType::Tensor}, {"B", PinType::Tensor, std::nullopt, true}};
constexpr PinDesc kTokensOutputs[] = {{"Ids", PinType::Tensor}, {"Mask", PinType::Tensor}};
constexpr PinDesc kLatentsOutputs[] = {{"Latents", PinType::Tensor}};
constexpr PinDesc kSamplerInputs[] = {{"Latents", PinType::Tensor}, {"Embeddings", PinType::Tensor}};

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
     kUploadNumberOutputs, Upload, ComputeGraph, "klartraum::HostFloat", "every frame (live), once per run"},
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
     kMakeTransformInputs, kMakeTransformOutputs, Gpu, ComputeGraph, "klartraum::TransformBuffer",
     kEveryExecution, true},
    {NodeKind::TransformGaussiansGpu, "transform_gaussians_gpu", "Transform (GPU)", "Gaussians (GPU)",
     "Moves Gaussians on the GPU by a transform, every frame.", kGpuTransformInputs, kGpuGaussiansOutputs, Gpu,
     ComputeGraph, "klartraum::GaussianTransform", kEveryExecution},
    {NodeKind::MergeGaussiansGpu, "merge_gaussians_gpu", "Merge (GPU)", "Gaussians (GPU)",
     "Combines two sets of Gaussians on the GPU, every frame.", kGpuMergeInputs, kGpuGaussiansOutputs, Gpu,
     ComputeGraph, "klartraum::GaussianMerge", kEveryExecution},
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
    {NodeKind::OnnxModel, "onnx_model", "ONNX Model", "Compute",
     "Runs an ONNX network; it has one tensor pin per model input and output.", kTensorInputs, kTensorOutputs, Gpu,
     ComputeGraph, "klartraum::OnnxNetwork", kEveryExecution},
    {NodeKind::Add, "add", "Add", "Layers", "A + B, elementwise; the shapes broadcast. Without B, adds the node's b.",
     kBinaryLayerInputs, kTensorOutputs, Gpu, ComputeGraph, "klartraum::layers::binary (Add)", kEveryExecution},
    {NodeKind::Subtract, "subtract", "Subtract", "Layers",
     "A - B, elementwise; the shapes broadcast. Without B, subtracts the node's b.", kBinaryLayerInputs,
     kTensorOutputs, Gpu, ComputeGraph, "klartraum::layers::binary (Sub)", kEveryExecution},
    {NodeKind::Multiply, "multiply", "Multiply", "Layers",
     "A * B, elementwise; the shapes broadcast. Without B, multiplies by the node's b.", kBinaryLayerInputs,
     kTensorOutputs, Gpu, ComputeGraph, "klartraum::layers::binary (Mul)", kEveryExecution},
    {NodeKind::Divide, "divide", "Divide", "Layers",
     "A / B, elementwise; the shapes broadcast. Without B, divides by the node's b.", kBinaryLayerInputs,
     kTensorOutputs, Gpu, ComputeGraph, "klartraum::layers::binary (Div)", kEveryExecution},
    {NodeKind::Relu, "relu", "ReLU", "Layers", "max(x, 0), elementwise.", kTensorInputs, kTensorOutputs, Gpu,
     ComputeGraph, "klartraum::layers::relu", kEveryExecution},
    {NodeKind::Sigmoid, "sigmoid", "Sigmoid", "Layers", "1 / (1 + exp(-x)), elementwise.", kTensorInputs,
     kTensorOutputs, Gpu, ComputeGraph, "klartraum::layers::unary (Sigmoid)", kEveryExecution},
    {NodeKind::Sqrt, "sqrt", "Sqrt", "Layers", "The square root, elementwise.", kTensorInputs, kTensorOutputs, Gpu,
     ComputeGraph, "klartraum::layers::unary (Sqrt)", kEveryExecution},
    {NodeKind::Softmax, "softmax", "Softmax", "Layers", "Softmax over the last axis.", kTensorInputs, kTensorOutputs,
     Gpu, ComputeGraph, "klartraum::layers::softmax", kEveryExecution},
    {NodeKind::Prompt, "sd_prompt", "Prompt", "Stable Diffusion",
     "A prompt and a negative prompt, tokenized with CLIP's byte-pair encoding into 2x77 int64 tensors: token "
     "ids (negative prompt first) and their attention mask.",
     {}, kTokensOutputs, Upload, ComputeGraph,
     "klartraum::ClipTokenizer on the CPU, uploaded into two klartraum::TensorElement<int64_t>", kOnBuild},
    {NodeKind::LatentNoise, "sd_latent_noise", "Latent Noise", "Stable Diffusion",
     "Gaussian noise of the latent size for a WxH image (1x4xH/8xW/8), from a seed or read from a file.", {},
     kLatentsOutputs, Cpu, Studio, "generated by the studio (kstudio::latentNoise)", "every run"},
    {NodeKind::DdimSampler, "sd_ddim_sampler", "DDIM Sampler", "Stable Diffusion",
     "Denoises latents step by step with the UNet, guided by the embeddings (classifier-free guidance). Runs "
     "between submissions of the run graph.",
     kSamplerInputs, kLatentsOutputs, Gpu, ComputeGraph,
     "klartraum::OnnxNetwork (UNet), submitted once per step; guidance and the DDIM update run on the CPU",
     "every run, once per denoising step"},
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
    {NodeKind::Meta, "meta", "Meta Node", "", "An instance of a meta node definition: a subgraph shown as one node.",
     {}, {}, Gpu, ComputeGraph, "its inner nodes (open it to see them)", "as its inner nodes"},
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
    case NodeKind::Add:
    case NodeKind::Subtract:
    case NodeKind::Multiply:
    case NodeKind::Divide: return BinaryLayerParams{};
    case NodeKind::Relu:
    case NodeKind::Sigmoid:
    case NodeKind::Sqrt:
    case NodeKind::Softmax: return UnaryLayerParams{};
    case NodeKind::Prompt: return PromptParams{};
    case NodeKind::LatentNoise: return LatentNoiseParams{};
    case NodeKind::DdimSampler: return DdimSamplerParams{};
    case NodeKind::Preview: return PreviewParams{};
    case NodeKind::ImageFileWriter: return ImageFileWriterParams{};
    case NodeKind::Meta: return MetaParams{};
    }
    throw std::logic_error("unknown node kind");
}

bool isSink(NodeKind kind) {
    return kind == NodeKind::Preview || kind == NodeKind::ImageFileWriter;
}

bool isBinaryLayer(NodeKind kind) {
    return kind == NodeKind::Add || kind == NodeKind::Subtract || kind == NodeKind::Multiply ||
           kind == NodeKind::Divide;
}

bool isUnaryLayer(NodeKind kind) {
    return kind == NodeKind::Relu || kind == NodeKind::Sigmoid || kind == NodeKind::Sqrt || kind == NodeKind::Softmax;
}

bool isStaged(NodeKind kind) {
    return kind == NodeKind::DdimSampler;
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
    const auto outputs = outputPins(*src);
    const auto inputs = inputPins(*dst);
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
    const auto outputs = outputPins(*source);
    return link->fromSlot < static_cast<int>(outputs.size()) ? std::optional(outputs[link->fromSlot].type)
                                                             : std::nullopt;
}

std::vector<Pin> Graph::inputPins(const Node& node) const {
    return pins(node, PinDirection::Input, *this, 0);
}

std::vector<Pin> Graph::outputPins(const Node& node) const {
    return pins(node, PinDirection::Output, *this, 0);
}

std::vector<Pin> Graph::pins(const Node& node, PinDirection direction, const Graph& resolver, int depth) const {
    const bool input = direction == PinDirection::Input;
    if (node.kind == NodeKind::OnnxModel) {
        std::vector<Pin> pins;
        const auto& p = node.as<OnnxModelParams>();
        for (const auto& name : input ? p.inputs : p.outputs) {
            pins.emplace_back(name, PinType::Tensor);
        }
        return pins;
    }
    if (node.kind == NodeKind::Meta) {
        // Definitions nest; a cycle between them is reported by flatten().
        const MetaDefinition* definition = resolver.findDefinition(node.as<MetaParams>().definition);
        if (!definition || depth > 16) {
            return {};
        }
        std::vector<Pin> pins;
        for (const auto& exposed : input ? definition->inputs : definition->outputs) {
            Pin pin(exposed.name, PinType::Tensor);
            bool first = true;
            bool optional = true;
            for (const auto& target : exposed.targets) {
                const Node* inner = definition->graph.findNode(target.node);
                if (!inner) {
                    continue;
                }
                const auto innerPins = definition->graph.pins(*inner, direction, resolver, depth + 1);
                if (target.slot < 0 || target.slot >= static_cast<int>(innerPins.size())) {
                    continue;
                }
                const Pin& innerPin = innerPins[target.slot];
                if (first) {
                    pin.type = innerPin.type;
                    pin.alsoAccepts = innerPin.alsoAccepts;
                    first = false;
                }
                optional = optional && innerPin.optional;
            }
            pin.optional = input && optional && !exposed.targets.empty();
            pins.push_back(std::move(pin));
        }
        return pins;
    }
    const auto& desc = input ? kindInfo(node.kind).inputs : kindInfo(node.kind).outputs;
    return {desc.begin(), desc.end()};
}

const MetaDefinition* Graph::findDefinition(std::string_view name) const {
    if (auto it = definitions_.find(name); it != definitions_.end()) {
        return it->second.get();
    }
    const auto& builtins = builtinDefinitions();
    auto it = builtins.find(name);
    return it == builtins.end() ? nullptr : it->second.get();
}

void Graph::setDefinition(std::shared_ptr<const MetaDefinition> definition, bool changesGraph) {
    const std::string name = definition->name;
    definitions_[name] = std::move(definition);
    if (changesGraph) {
        touch();
    }
}

bool Graph::removeDefinition(std::string_view name) {
    auto it = definitions_.find(name);
    if (it == definitions_.end()) {
        return false;
    }
    definitions_.erase(it);
    touch();
    return true;
}

int Graph::addMetaNode(const std::string& name, Vec2 position) {
    const int id = addNode(NodeKind::Meta, position);
    Node& node = *findNode(id);
    node.as<MetaParams>().definition = name;
    if (const MetaDefinition* definition = findDefinition(name)) {
        node.title = definition->title;
    }
    return id;
}

int Graph::setOnnxPins(int nodeId, std::vector<std::string> inputs, std::vector<std::string> outputs) {
    Node* node = findNode(nodeId);
    if (!node || node->kind != NodeKind::OnnxModel) {
        throw std::logic_error("setOnnxPins: not an ONNX Model node");
    }
    inputs.resize(std::min<size_t>(inputs.size(), kMaxPinsPerDirection));
    outputs.resize(std::min<size_t>(outputs.size(), kMaxPinsPerDirection));
    auto& p = node->as<OnnxModelParams>();
    if (p.inputs == inputs && p.outputs == outputs) {
        return 0;
    }
    p.inputs = std::move(inputs);
    p.outputs = std::move(outputs);
    const int in = static_cast<int>(p.inputs.size());
    const int out = static_cast<int>(p.outputs.size());
    const auto removed = std::erase_if(links_, [&](const Link& l) {
        return (l.toNode == nodeId && l.toSlot >= in) || (l.fromNode == nodeId && l.fromSlot >= out);
    });
    touch();
    return static_cast<int>(removed);
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
        const auto inputs = inputPins(node);
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
        case NodeKind::Prompt:
            if (node.as<PromptParams>().vocabulary.empty()) {
                report(Severity::Error, node.id, "No vocabulary file (CLIP's vocab.json) is set.");
            }
            break;
        case NodeKind::LatentNoise: {
            const auto& p = node.as<LatentNoiseParams>();
            if (p.width == 0 || p.height == 0 || p.width % 8 != 0 || p.height % 8 != 0) {
                report(Severity::Error, node.id, "Width and height must be positive multiples of 8.");
            }
            break;
        }
        case NodeKind::DdimSampler: {
            const auto& p = node.as<DdimSamplerParams>();
            if (p.path.empty()) {
                report(Severity::Error, node.id, "No UNet model file is set.");
            }
            if (p.steps == 0 || p.steps > 1000) {
                report(Severity::Error, node.id, "Steps must be between 1 and 1000.");
            }
            break;
        }
        case NodeKind::Meta:
            if (!findDefinition(node.as<MetaParams>().definition)) {
                report(Severity::Error, node.id,
                       "Unknown meta node definition '" + node.as<MetaParams>().definition + "'.");
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

    // The live graph is submitted once per frame; staged nodes need several
    // submissions with CPU work in between.
    for (const Node* present : presents) {
        for (int id : upstreamOf(present->id)) {
            if (isStaged(findNode(id)->kind)) {
                report(Severity::Error, id,
                       "Only runs with Run: connect it to a Preview or Image File Writer instead of Present.");
            }
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
    definitions_.clear();
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

Graph makeStableDiffusionGraph(const std::string& modelDirectory, uint32_t size, const std::string& prompt,
                               const std::string& negativePrompt, const std::string& outputPath) {
    const std::filesystem::path directory(modelDirectory);
    auto file = [&](const char* name) { return (directory / name).generic_string(); };

    Graph graph;
    const int promptNode = graph.addNode(NodeKind::Prompt, {0.0f, 0.0f});
    const int encoder = graph.addMetaNode(kTextEncoderDefinition, {310.0f, 0.0f});
    const int noise = graph.addNode(NodeKind::LatentNoise, {310.0f, 200.0f});
    const int sampler = graph.addNode(NodeKind::DdimSampler, {620.0f, 100.0f});
    const int decoder = graph.addMetaNode(kVaeDecoderDefinition, {930.0f, 100.0f});
    const int preview = graph.addNode(NodeKind::Preview, {1240.0f, 0.0f});
    const int writer = graph.addNode(NodeKind::ImageFileWriter, {1240.0f, 260.0f});

    auto& p = graph.findNode(promptNode)->as<PromptParams>();
    p.prompt = prompt;
    p.negativePrompt = negativePrompt;
    p.vocabulary = file("vocab.json");
    graph.findNode(encoder)->as<MetaParams>().values["Model"] = nlohmann::json(file("sd15_text_encoder.onnx")).dump();
    auto& n = graph.findNode(noise)->as<LatentNoiseParams>();
    n.width = size;
    n.height = size;
    graph.findNode(sampler)->as<DdimSamplerParams>().path = file("sd15_unet.onnx");
    graph.findNode(decoder)->as<MetaParams>().values["Model"] = nlohmann::json(file("sd15_vae_decoder.onnx")).dump();
    graph.findNode(writer)->as<ImageFileWriterParams>().path = outputPath;

    graph.connect(out(promptNode, 0), in(encoder, 0));  // token ids
    graph.connect(out(promptNode, 1), in(encoder, 1));  // attention mask
    graph.connect(out(noise), in(sampler, 0));
    graph.connect(out(encoder), in(sampler, 1));
    graph.connect(out(sampler), in(decoder));
    graph.connect(out(decoder), in(preview));
    graph.connect(out(decoder), in(writer));
    return graph;
}

} // namespace kstudio
