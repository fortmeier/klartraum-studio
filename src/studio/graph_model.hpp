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
//  - the live part, everything feeding Present, renders every frame;
//  - the run part, everything feeding a sink (Preview, Image File Writer),
//    is executed once each time the user presses Run.

enum class PinType { Gaussians, Camera, Image, Tensor };

enum class NodeKind {
    Scene,
    Camera,
    SwapchainTarget,
    GaussianSplatting,
    Present,
    OffscreenTarget,
    ImageFile,
    ImageToTensor,
    OnnxModel,
    Preview,
    ImageFileWriter,
};

enum class SplattingBackend { Compute, Raster };

enum class UpAxis { Y, Z };

struct SceneParams {
    std::string path;
    bool operator==(const SceneParams&) const = default;
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

struct OnnxModelParams {
    std::string path;
    bool operator==(const OnnxModelParams&) const = default;
};

struct PreviewParams {
    bool operator==(const PreviewParams&) const = default;
};

struct ImageFileWriterParams {
    std::string path = "output.png";
    bool operator==(const ImageFileWriterParams&) const = default;
};

using NodeParams = std::variant<SceneParams, CameraParams, SwapchainTargetParams, SplattingParams, PresentParams,
                                OffscreenTargetParams, ImageFileParams, ImageToTensorParams, OnnxModelParams,
                                PreviewParams, ImageFileWriterParams>;

struct PinDesc {
    std::string_view name;
    PinType type;
};

struct NodeKindInfo {
    NodeKind kind;
    std::string_view name;        // stable identifier used in files
    std::string_view title;       // default display title
    std::string_view group;       // for the add-node menu
    std::string_view description;
    std::span<const PinDesc> inputs;
    std::span<const PinDesc> outputs;
};

const NodeKindInfo& kindInfo(NodeKind kind);
std::span<const NodeKindInfo> allKinds();
std::optional<NodeKind> kindFromName(std::string_view name);
std::string_view pinTypeName(PinType type);
std::string_view backendName(SplattingBackend backend);
NodeParams defaultParams(NodeKind kind);
// Preview and Image File Writer: executed by Run.
bool isSink(NodeKind kind);

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

// A Gaussian-splatting rendering into an offscreen image, fed through the
// encoder and decoder and previewed on Run.
Graph makeSplatAutoencoderGraph(const std::string& scenePath, const std::string& encoderPath,
                                const std::string& decoderPath);

} // namespace kstudio
