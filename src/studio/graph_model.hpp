#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace kstudio {

// The authoring graph: what the user edits in the node editor. It describes
// work at the level of klartraum's public building blocks. Two parts of it
// are compiled separately (see graph_compiler.hpp):
//  - the live part, everything feeding Present, runs every frame;
//  - the run part, everything feeding a sink (Preview, Image File Writer),
//    is executed once each time the user presses Run.

// Gaussians and numbers are CPU data until an upload node puts them into GPU
// buffers; images, tensors, transforms and camera buffers live on the GPU.
// A tensor's shape and element type are not part of the pin type; shape
// inference checks them (see tensor_shapes.hpp).
enum class PinType { GaussiansCpu, GaussiansGpu, Camera, Image, Tensor, NumberCpu, NumberGpu, TransformGpu };

// Where a node's work happens: on the GPU, on the CPU, or moving data between
// them.
enum class ExecutionSite { Gpu, Cpu, Upload, Readback };

// What a node is: elements of the klartraum compute graph, a klartraum
// function the studio calls on the CPU, or the studio's own code.
enum class Implementation { ComputeGraph, KlartraumFunction, Studio };

enum class NodeKind {
    Scene,
    TransformGaussians,
    MergeGaussians,
    UploadGaussians,
    Number,
    Time,
    Sine,
    UploadNumber,
    MakeTransform,
    TransformGaussiansGpu,
    MergeGaussiansGpu,
    Camera,
    SwapchainTarget,
    GaussianSplatting,
    Present,
    OffscreenTarget,
    ImageFile,
    ImageToTensor,
    TensorToImage,
    Resample,
    OnnxModel,
    Add,
    Subtract,
    Multiply,
    Divide,
    Relu,
    Sigmoid,
    Sqrt,
    Softmax,
    Prompt,
    LatentNoise,
    DdimSampler,
    Preview,
    ImageFileWriter,
    Meta,
};

enum class SplattingBackend { Compute, Raster };

enum class UpAxis { Y, Z };

enum class ResampleFilter { Nearest, Bilinear };

struct SceneParams {
    std::string path;
    bool flipY = false;  // mirror across the Y axis, e.g. for Nerfstudio exports
    bool operator==(const SceneParams&) const = default;
};

// Moves Gaussians: scaled about the origin, rotated (degrees about X, then Y,
// then Z) and translated.
struct TransformGaussiansParams {
    std::array<float, 3> translation = {0.0f, 0.0f, 0.0f};
    std::array<float, 3> rotation = {0.0f, 0.0f, 0.0f};
    float scale = 1.0f;
    bool operator==(const TransformGaussiansParams&) const = default;
};

struct MergeGaussiansParams {
    bool operator==(const MergeGaussiansParams&) const = default;
};

struct UploadGaussiansParams {
    bool operator==(const UploadGaussiansParams&) const = default;
};

// CPU numbers, evaluated every frame.
struct NumberParams {
    float value = 0.0f;
    bool operator==(const NumberParams&) const = default;
};

// Seconds since the studio started, times `speed`.
struct TimeParams {
    float speed = 1.0f;
    bool operator==(const TimeParams&) const = default;
};

// amplitude * sin(frequency * 2 pi * x + phase) + offset, phase in degrees.
struct SineParams {
    float amplitude = 1.0f;
    float frequency = 1.0f;
    float phase = 0.0f;
    float offset = 0.0f;
    bool operator==(const SineParams&) const = default;
};

struct UploadNumberParams {
    bool operator==(const UploadNumberParams&) const = default;
};

// A GPU transform from x, y, z, pitch, yaw, roll (degrees, about X, then Y,
// then Z) and scale; inputs that are not connected take these values.
struct MakeTransformParams {
    std::array<float, 3> translation = {0.0f, 0.0f, 0.0f};
    std::array<float, 3> rotation = {0.0f, 0.0f, 0.0f};  // pitch, yaw, roll
    float scale = 1.0f;
    bool operator==(const MakeTransformParams&) const = default;
    // In input order: x, y, z, pitch, yaw, roll, scale.
    float component(int index) const { return index < 3 ? translation[index] : index < 6 ? rotation[index - 3] : scale; }
};

struct TransformGaussiansGpuParams {
    bool operator==(const TransformGaussiansGpuParams&) const = default;
};

struct MergeGaussiansGpuParams {
    bool operator==(const MergeGaussiansGpuParams&) const = default;
};

// Orbit camera. These values are applied live and never require a rebuild.
struct CameraParams {
    float azimuth = 0.9f;
    float elevation = -0.5f;
    float distance = 1.0f;
    std::array<float, 3> target = {-0.5f, 0.0f, 0.5f};
    UpAxis up = UpAxis::Y;

    bool operator==(const CameraParams&) const = default;
};

struct SwapchainTargetParams {
    bool operator==(const SwapchainTargetParams&) const = default;
};

// Mirrors klartraum::GsplatConfig; each backend reads only some fields.
struct SplattingParams {
    SplattingBackend backend = SplattingBackend::Raster;

    // Compute backend
    float spreadMultiplier = 2.5f;
    uint32_t maxMod = 2u;
    uint32_t numSortWGsCap = 320u;
    uint32_t splatTileX = 8u;
    uint32_t splatTileY = 8u;

    // Raster backend
    int shDegree = 3;
    float alphaCullThreshold = 1.0f / 255.0f;
    bool useMeshShader = false;

    bool operator==(const SplattingParams&) const = default;
};

struct PresentParams {
    bool operator==(const PresentParams&) const = default;
};

// An image of fixed size that rendering can target instead of the swapchain.
struct OffscreenTargetParams {
    uint32_t width = 128;
    uint32_t height = 128;
    bool operator==(const OffscreenTargetParams&) const = default;
};

// Loads an image file and resizes it to width x height; outputs a
// 1x3xHxW tensor.
struct ImageFileParams {
    std::string path;
    uint32_t width = 128;
    uint32_t height = 128;
    bool operator==(const ImageFileParams&) const = default;
};

struct ImageToTensorParams {
    bool operator==(const ImageToTensorParams&) const = default;
};

struct TensorToImageParams {
    bool operator==(const TensorToImageParams&) const = default;
};

// Resamples an image to width x height (klartraum::ImageResample).
struct ResampleParams {
    uint32_t width = 128;
    uint32_t height = 128;
    ResampleFilter filter = ResampleFilter::Bilinear;
    bool operator==(const ResampleParams&) const = default;
};

// An ONNX model with one tensor pin per model input and output. The pin
// names are the model's input and output names; the studio updates them when
// it reads the model (Graph::setOnnxPins), so links stay attached by slot.
struct OnnxModelParams {
    std::string path;
    std::vector<std::string> inputs{"input"};
    std::vector<std::string> outputs{"output"};
    bool operator==(const OnnxModelParams&) const = default;
};

// Layers (klartraum::layers) on float tensors.

// Add, Subtract, Multiply, Divide: A op B, broadcasting both. If B is not
// connected, it is the one-element tensor `b`.
struct BinaryLayerParams {
    float b = 1.0f;
    bool operator==(const BinaryLayerParams&) const = default;
};

// ReLU, Sigmoid, Sqrt, Softmax (over the last axis).
struct UnaryLayerParams {
    bool operator==(const UnaryLayerParams&) const = default;
};

// Stable Diffusion 1.5, as exported by klartraum's scripts/sd15_onnx.

// A prompt and a negative prompt, tokenized with CLIP's byte-pair encoding
// (klartraum::ClipTokenizer) into two 2x77 int64 tensors: token ids and
// attention mask. `vocabulary` is CLIP's vocab.json; merges.txt must be next
// to it.
struct PromptParams {
    std::string prompt;
    std::string negativePrompt;
    std::string vocabulary;
    bool operator==(const PromptParams&) const = default;
};

// The latents a sampler starts from, 1x4x(height/8)x(width/8) for a
// width x height image: Gaussian noise from `seed`, or, if `path` is set,
// raw float32 values read from that file (e.g. initial_latents_f32.bin
// written by export_denoiser.py, to reproduce its reference).
struct LatentNoiseParams {
    uint32_t width = 512;
    uint32_t height = 512;
    uint32_t seed = 0;
    std::string path;
    bool operator==(const LatentNoiseParams&) const = default;
};

// Denoises latents with the UNet in `path`: DDIM with SD 1.5's scheduler and
// classifier-free guidance.
struct DdimSamplerParams {
    std::string path;
    uint32_t steps = 20;
    float guidanceScale = 7.5f;
    bool operator==(const DdimSamplerParams&) const = default;
};

// A meta node: an instance of a definition (see MetaDefinition), with values
// for the parameters the definition exposes, as JSON text by parameter
// name. Exposed parameters without a value keep the definition's.
struct MetaParams {
    std::string definition;
    std::map<std::string, std::string> values;
    bool operator==(const MetaParams&) const = default;
};

struct PreviewParams {
    bool operator==(const PreviewParams&) const = default;
};

struct ImageFileWriterParams {
    std::string path = "output.png";
    bool operator==(const ImageFileWriterParams&) const = default;
};

using NodeParams = std::variant<SceneParams, TransformGaussiansParams, MergeGaussiansParams, UploadGaussiansParams,
                                NumberParams, TimeParams, SineParams, UploadNumberParams, MakeTransformParams,
                                TransformGaussiansGpuParams, MergeGaussiansGpuParams, CameraParams, SwapchainTargetParams, SplattingParams, PresentParams,
                                OffscreenTargetParams, ImageFileParams, ImageToTensorParams, TensorToImageParams,
                                ResampleParams, OnnxModelParams, BinaryLayerParams, UnaryLayerParams, PromptParams, LatentNoiseParams,
                                DdimSamplerParams, PreviewParams, ImageFileWriterParams, MetaParams>;

// A pin of a node kind, as listed in the kinds table.
struct PinDesc {
    std::string_view name;
    PinType type;
    // An input may accept a second type (sinks take tensors and images).
    std::optional<PinType> alsoAccepts = std::nullopt;
    // An optional input may stay unconnected; the node then uses a parameter.
    bool optional = false;

    bool accepts(PinType other) const { return other == type || other == alsoAccepts; }
};

// A pin of a node in a graph: its kind's pin, or one that depends on the
// node's parameters (an ONNX model's inputs and outputs). See
// Graph::inputPins.
struct Pin {
    std::string name;
    PinType type = PinType::Tensor;
    std::optional<PinType> alsoAccepts = std::nullopt;
    bool optional = false;

    Pin() = default;
    Pin(std::string name, PinType type, std::optional<PinType> alsoAccepts = std::nullopt, bool optional = false)
        : name(std::move(name)), type(type), alsoAccepts(alsoAccepts), optional(optional) {}
    explicit Pin(const PinDesc& desc)
        : Pin(std::string(desc.name), desc.type, desc.alsoAccepts, desc.optional) {}

    bool accepts(PinType other) const { return other == type || other == alsoAccepts; }
};

struct NodeKindInfo {
    NodeKind kind;
    std::string_view name;        // stable identifier used in files
    std::string_view title;       // default display title
    std::string_view group;       // for the add-node menu
    std::string_view description;
    std::span<const PinDesc> inputs;
    std::span<const PinDesc> outputs;
    ExecutionSite site;
    Implementation implementation;
    std::string_view implementedBy;  // the klartraum class or function, or the studio code
    std::string_view timing;         // when the work happens
    // Parameters that apply while the live graph runs, without rebuilding it.
    bool liveParams = false;
};

const NodeKindInfo& kindInfo(NodeKind kind);
std::span<const NodeKindInfo> allKinds();
std::optional<NodeKind> kindFromName(std::string_view name);
std::string_view pinTypeName(PinType type);
std::string_view backendName(SplattingBackend backend);
std::string_view filterName(ResampleFilter filter);
std::string_view siteName(ExecutionSite site);
std::string_view implementationName(Implementation implementation);
NodeParams defaultParams(NodeKind kind);
// Preview and Image File Writer: executed by Run.
bool isSink(NodeKind kind);
// DDIM Sampler: Run executes it between submissions of its klartraum graph.
// It reads its input tensors back, runs the UNet once per denoising step with
// CPU work in between, and hands its result on as CPU data, which the nodes
// after it upload again. It cannot run live.
bool isStaged(NodeKind kind);
// Add, Subtract, Multiply, Divide.
bool isBinaryLayer(NodeKind kind);
// ReLU, Sigmoid, Sqrt, Softmax.
bool isUnaryLayer(NodeKind kind);
// Nodes whose Image output holds a result (Gaussian Splatting, Tensor to
// Image, Resample), as opposed to an empty target.
bool producesImage(NodeKind kind);

struct Vec2 {
    float x = 0.0f;
    float y = 0.0f;
    bool operator==(const Vec2&) const = default;
};

struct Node {
    int id = 0;
    NodeKind kind = NodeKind::Scene;
    std::string title;
    NodeParams params;
    Vec2 position;

    template <typename T> T& as() { return std::get<T>(params); }
    template <typename T> const T& as() const { return std::get<T>(params); }
};

// Pins are identified by (node, direction, slot). Their integer id, used by
// the node editor, packs these so that it is unique within a graph.
enum class PinDirection { Input, Output };

struct PinRef {
    int node = 0;
    PinDirection direction = PinDirection::Input;
    int slot = 0;

    bool operator==(const PinRef&) const = default;
};

constexpr int kMaxPinsPerDirection = 8;

// An output pin: (node id, output slot).
using OutputPin = std::pair<int, int>;
int pinId(const PinRef& pin);
PinRef pinFromId(int id);

struct Link {
    int id = 0;
    int fromNode = 0;   // output pin
    int fromSlot = 0;
    int toNode = 0;     // input pin
    int toSlot = 0;
};

enum class Severity { Info, Warning, Error };

struct Diagnostic {
    Severity severity = Severity::Error;
    int node = -1;  // -1: the graph as a whole
    std::string message;
};

struct MetaDefinition;

// Definitions by name.
using MetaDefinitions = std::map<std::string, std::shared_ptr<const MetaDefinition>, std::less<>>;

class Graph {
public:
    int addNode(NodeKind kind, Vec2 position = {});
    // Adds a node with the given id, e.g. when loading a file. Returns false
    // if the id is taken.
    bool addNodeWithId(Node node);
    bool removeNode(int id);

    // Connects an output pin to an input pin. An input accepts one link, so
    // an existing link into `to` is replaced. Returns an error message if the
    // pins are incompatible or the link would create a cycle.
    std::optional<std::string> connect(PinRef from, PinRef to);
    // The error connect() would report, without changing the graph.
    std::optional<std::string> checkConnection(PinRef from, PinRef to) const;
    bool removeLink(int id);

    Node* findNode(int id);
    const Node* findNode(int id) const;
    const Link* findLink(int id) const;
    // The link feeding an input pin, if any.
    const Link* inputLink(int node, int slot) const;
    const Node* inputNode(int node, int slot) const;
    // The type of the output feeding an input pin, if connected.
    std::optional<PinType> inputType(int node, int slot) const;

    // A node's pins: its kind's, or, for an ONNX model, one tensor pin per
    // model input and output.
    std::vector<Pin> inputPins(const Node& node) const;
    std::vector<Pin> outputPins(const Node& node) const;

    // Meta node definitions stored with this graph (and saved with it).
    // Meta nodes find their definition here first, then among the built-in
    // ones (builtinDefinitions()).
    const MetaDefinitions& definitions() const { return definitions_; }
    const MetaDefinition* findDefinition(std::string_view name) const;
    // Adds or replaces a definition; definitions are immutable, so editing
    // one replaces it for all its instances. Without `changesGraph` (e.g.
    // when only inner nodes moved) the revision stays.
    void setDefinition(std::shared_ptr<const MetaDefinition> definition, bool changesGraph = true);
    bool removeDefinition(std::string_view name);
    // Adds a meta node for definition `name`, titled like the definition.
    int addMetaNode(const std::string& name, Vec2 position = {});

    // Sets an ONNX Model node's pins to the model's input and output names
    // (at most kMaxPinsPerDirection each). Links to slots that no longer
    // exist are removed; returns how many.
    int setOnnxPins(int node, std::vector<std::string> inputs, std::vector<std::string> outputs);

    const std::vector<Node>& nodes() const { return nodes_; }
    std::vector<Node>& nodes() { return nodes_; }
    const std::vector<Link>& links() const { return links_; }

    // Checks the graph's structure; see GraphCompiler for what the klartraum
    // backends can express. Tensor shapes and files are checked by the
    // compiler (planGraph).
    std::vector<Diagnostic> validate() const;

    // The nodes `node` depends on, including itself.
    std::vector<int> upstreamOf(int node) const;
    // Nodes in dependency order (producers first); ties keep insertion order.
    std::vector<int> topologicalOrder() const;
    bool hasErrors() const;

    // Incremented on every change that may affect the compiled graph.
    uint64_t revision() const { return revision_; }
    void touch() { ++revision_; }

    void clear();

private:
    bool reaches(int fromNode, int toNode) const;
    // A node's pins; meta nodes resolve definitions in `resolver`, also for
    // the meta nodes nested in them.
    std::vector<Pin> pins(const Node& node, PinDirection direction, const Graph& resolver, int depth) const;

    MetaDefinitions definitions_;
    std::vector<Node> nodes_;
    std::vector<Link> links_;
    int nextNodeId_ = 1;
    int nextLinkId_ = 1;
    uint64_t revision_ = 0;
};

// A reusable subgraph: its inner nodes appear as one meta node. The pins and
// parameters it exposes stand for pins and parameters of inner nodes.
struct MetaPin {
    std::string name;
    // An exposed input feeds these inner input pins; an exposed output is
    // exactly one inner output pin.
    std::vector<PinRef> targets;
    bool operator==(const MetaPin&) const = default;
};

struct MetaParam {
    std::string name;
    int node = 0;
    std::string key;  // the inner node's parameter, as named in graph files
    bool operator==(const MetaParam&) const = default;
};

struct MetaDefinition {
    std::string name;  // identifier, unique among a graph's and the built-in definitions
    std::string title;
    std::string group;  // add-node menu group
    std::string description;
    Graph graph;
    std::vector<MetaPin> inputs;
    std::vector<MetaPin> outputs;
    std::vector<MetaParam> params;
    bool builtin = false;
};

// The definitions that come with the studio (see meta_nodes.hpp).
const MetaDefinitions& builtinDefinitions();

// The Gaussian-splatting pipeline: Scene, Camera and Swapchain Target feed a
// Gaussian Splatting node, whose image is presented.
Graph makeGaussianSplattingGraph(const std::string& scenePath, SplattingBackend backend = SplattingBackend::Raster);

// An image file is encoded and decoded by two ONNX models; the result is
// previewed and written to `outputPath` on Run.
Graph makeAutoencoderGraph(const std::string& imagePath, const std::string& encoderPath,
                           const std::string& decoderPath, const std::string& outputPath);

// Two scenes in one Gaussian Splatting: `movedScene` is transformed by
// `transform` and merged with `scene`, seen through `camera`.
Graph makeCombinedScenesGraph(const std::string& scenePath, const SceneParams& movedScene,
                              const TransformGaussiansParams& transform, const CameraParams& camera = {});

// Like makeCombinedScenesGraph, but moved and merged on the GPU every frame:
// `movedScene` is placed by a Make Transform whose yaw swings by `swing` over
// time (Time -> Sine -> Upload Number).
Graph makeAnimatedScenesGraph(const std::string& scenePath, const SceneParams& movedScene,
                              const MakeTransformParams& placement, const SineParams& swing,
                              const CameraParams& camera = {});

// A Gaussian-splatting rendering into an offscreen image, fed through the
// encoder and decoder and previewed on Run.
Graph makeSplatAutoencoderGraph(const std::string& scenePath, const std::string& encoderPath,
                                const std::string& decoderPath);

// Stable Diffusion 1.5 from the models export_denoiser.py writes into
// `modelDirectory` for `size` x `size` images: Prompt -> Text Encoder, Latent
// Noise -> DDIM Sampler -> VAE Decoder (the encoder and decoder are built-in
// meta nodes), previewed and written to `outputPath` on Run.
Graph makeStableDiffusionGraph(const std::string& modelDirectory, uint32_t size, const std::string& prompt,
                               const std::string& negativePrompt, const std::string& outputPath);

} // namespace kstudio
