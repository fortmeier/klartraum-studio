#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <span>
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
// Tokens are a Stable Diffusion prompt's CLIP token ids and attention mask.
enum class PinType { GaussiansCpu, GaussiansGpu, Camera, Image, Tensor, NumberCpu, NumberGpu, TransformGpu, Tokens };

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
    Prompt,
    TextEncoder,
    LatentNoise,
    DdimSampler,
    VaeDecoder,
    Preview,
    ImageFileWriter,
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

struct OnnxModelParams {
    std::string path;
    bool operator==(const OnnxModelParams&) const = default;
};

// Stable Diffusion 1.5, as exported by klartraum's scripts/sd15_onnx.

// A prompt and a negative prompt, tokenized with CLIP's byte-pair encoding
// (klartraum::ClipTokenizer). `vocabulary` is CLIP's vocab.json; merges.txt
// must be next to it.
struct PromptParams {
    std::string prompt;
    std::string negativePrompt;
    std::string vocabulary;
    bool operator==(const PromptParams&) const = default;
};

// The CLIP text encoder: 2x77 tokens -> 2x77x768 embeddings.
struct TextEncoderParams {
    std::string path;
    bool operator==(const TextEncoderParams&) const = default;
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

// Decodes latents into a 1x3xHxW image tensor with values in [0, 1].
struct VaeDecoderParams {
    std::string path;
    bool operator==(const VaeDecoderParams&) const = default;
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
                                ResampleParams, OnnxModelParams, PromptParams, TextEncoderParams, LatentNoiseParams,
                                DdimSamplerParams, VaeDecoderParams, PreviewParams, ImageFileWriterParams>;

struct PinDesc {
    std::string_view name;
    PinType type;
    // An input may accept a second type (sinks take tensors and images).
    std::optional<PinType> alsoAccepts = std::nullopt;
    // An optional input may stay unconnected; the node then uses a parameter.
    bool optional = false;

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
// DDIM Sampler and VAE Decoder: Run executes them between submissions of its
// klartraum graph. They read their input tensors back, run klartraum graphs
// of their own (the sampler one per denoising step, with CPU work in between)
// and hand their result on as CPU data, which the nodes after them upload
// again. They cannot run live.
bool isStaged(NodeKind kind);
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

    std::vector<Node> nodes_;
    std::vector<Link> links_;
    int nextNodeId_ = 1;
    int nextLinkId_ = 1;
    uint64_t revision_ = 0;
};

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
// Noise -> DDIM Sampler -> VAE Decoder, previewed and written to
// `outputPath` on Run.
Graph makeStableDiffusionGraph(const std::string& modelDirectory, uint32_t size, const std::string& prompt,
                               const std::string& negativePrompt, const std::string& outputPath);

} // namespace kstudio
