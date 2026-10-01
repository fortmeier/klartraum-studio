#include "studio/studio_app.hpp"

#include <algorithm>
#include <chrono>
#include <format>
#include <set>

#include <imgui.h>
#include <imgui_node_editor.h>

#include "klartraum/gaussian_data_standard.hpp"
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "klartraum/klartraum_core.hpp"
#include "klartraum/interface_camera_orbit.hpp"

#include "studio/cpu_numbers.hpp"
#include "studio/graph_layout.hpp"
#include "studio/graph_serialization.hpp"
#include "studio/meta_nodes.hpp"
#include "studio/retained_results.hpp"
#include "studio/stable_diffusion.hpp"

#include <nlohmann/json.hpp>

namespace kstudio {

namespace ed = ax::NodeEditor;

namespace {

constexpr double kProfileInterval = 0.5;  // seconds between profiling readbacks

constexpr float kNodeWidth = 170.0f;

// Editor ids must be non-zero. Pins and links get their own ranges so an id
// never names two kinds of object.
constexpr uintptr_t kPinIdBase = uintptr_t{1} << 28;
constexpr uintptr_t kLinkIdBase = uintptr_t{1} << 29;

// Authoring graph: node ids are the graph's (they start at 1).
ed::NodeId authoringNodeId(int node) { return ed::NodeId(static_cast<uintptr_t>(node)); }
ed::PinId authoringPinId(const PinRef& pin) { return ed::PinId(kPinIdBase + static_cast<uintptr_t>(pinId(pin))); }
PinRef pinFromEditor(ed::PinId pin) { return pinFromId(static_cast<int>(pin.Get() - kPinIdBase)); }
ed::LinkId authoringLinkId(int link) { return ed::LinkId(kLinkIdBase + static_cast<uintptr_t>(link)); }
int linkFromEditor(ed::LinkId link) { return static_cast<int>(link.Get() - kLinkIdBase); }

// Compiled graph: element ids start at 0; per element, input slots count up
// from its pin base and the single output uses the last pin.
constexpr uintptr_t kElementPinStride = 512;
ed::NodeId compiledNodeId(int element) { return ed::NodeId(static_cast<uintptr_t>(element) + 1); }
int elementFromEditor(ed::NodeId node) { return static_cast<int>(node.Get()) - 1; }
ed::PinId compiledInputPinId(int element, int slot) {
    return ed::PinId(kPinIdBase + static_cast<uintptr_t>(element) * kElementPinStride + static_cast<uintptr_t>(slot));
}
ed::PinId compiledOutputPinId(int element) {
    return ed::PinId(kPinIdBase + static_cast<uintptr_t>(element) * kElementPinStride + kElementPinStride - 1);
}
ed::LinkId compiledLinkId(int edge) { return ed::LinkId(kLinkIdBase + static_cast<uintptr_t>(edge) + 1); }

ImU32 rgb(int r, int g, int b, int a = 255) { return IM_COL32(r, g, b, a); }

ImU32 pinColor(PinType type) {
    switch (type) {
    case PinType::GaussiansCpu: return rgb(176, 164, 140);
    case PinType::GaussiansGpu: return rgb(236, 178, 72);
    case PinType::NumberCpu: return rgb(176, 176, 132);
    case PinType::NumberGpu: return rgb(238, 218, 96);
    case PinType::TransformGpu: return rgb(110, 214, 214);
    case PinType::Camera: return rgb(110, 196, 140);
    case PinType::Image: return rgb(96, 160, 240);
    case PinType::Tensor: return rgb(206, 120, 226);
    }
    return rgb(200, 200, 200);
}

// Where a node runs, as shown in badges.
ImU32 siteColor(ExecutionSite site) {
    switch (site) {
    case ExecutionSite::Gpu: return rgb(110, 190, 255);
    case ExecutionSite::Cpu: return rgb(240, 185, 100);
    case ExecutionSite::Upload: return rgb(195, 155, 255);
    case ExecutionSite::Readback: return rgb(120, 215, 170);
    }
    return rgb(200, 200, 200);
}

// Rings around nodes that become elements of a klartraum compute graph, and
// the outline of elements the studio added on its own.
const ImU32 kLiveGraphColor = IM_COL32(90, 170, 255, 255);
const ImU32 kRunGraphColor = IM_COL32(120, 210, 130, 255);
const ImU32 kStudioAddedColor = IM_COL32(240, 185, 100, 255);

// GPU memory of uploaded Gaussians: position, rotation, scale, colour and
// opacity, and 3 x 15 SH coefficients, as floats.
double gaussianBytes(uint32_t count) {
    return count * (3.0 + 4.0 + 3.0 + 4.0 + 45.0) * sizeof(float);
}

// "GPU  klartraum graph": where a node runs and what it is made of.
void executionBadge(const NodeKindInfo& info) {
    ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(siteColor(info.site)), "%s", std::string(siteName(info.site)).c_str());
    ImGui::SameLine(0.0f, 6.0f);
    ImGui::TextDisabled("%s", std::string(implementationName(info.implementation)).c_str());
}

ImU32 kindColor(NodeKind kind) {
    switch (kind) {
    case NodeKind::Scene: return rgb(150, 104, 36);
    case NodeKind::Camera: return rgb(46, 120, 76);
    case NodeKind::SwapchainTarget: return rgb(44, 88, 150);
    case NodeKind::GaussianSplatting: return rgb(122, 60, 150);
    case NodeKind::Present: return rgb(70, 70, 82);
    case NodeKind::OffscreenTarget: return rgb(44, 110, 140);
    case NodeKind::ClearImage: return rgb(60, 90, 120);
    case NodeKind::Composite: return rgb(60, 105, 130);
    case NodeKind::DrawBasics: return rgb(70, 120, 110);
    case NodeKind::ImageFile: return rgb(150, 84, 50);
    case NodeKind::ImageToTensor: return rgb(110, 70, 140);
    case NodeKind::TensorToImage: return rgb(70, 90, 150);
    case NodeKind::Resample: return rgb(50, 110, 130);
    case NodeKind::UploadGaussians: return rgb(110, 80, 150);
    case NodeKind::Number:
    case NodeKind::Time:
    case NodeKind::Sine: return rgb(110, 110, 70);
    case NodeKind::UploadNumber: return rgb(120, 100, 40);
    case NodeKind::MakeTransform: return rgb(40, 120, 120);
    case NodeKind::TransformGaussiansGpu:
    case NodeKind::MergeGaussiansGpu: return rgb(150, 100, 40);
    case NodeKind::OnnxModel: return rgb(150, 60, 110);
    case NodeKind::Add:
    case NodeKind::Subtract:
    case NodeKind::Multiply:
    case NodeKind::Divide:
    case NodeKind::Relu:
    case NodeKind::Sigmoid:
    case NodeKind::Sqrt:
    case NodeKind::Softmax: return rgb(120, 70, 130);
    case NodeKind::Prompt: return rgb(130, 70, 150);
    case NodeKind::LatentNoise: return rgb(120, 110, 60);
    case NodeKind::DdimSampler: return rgb(160, 70, 90);
    case NodeKind::Meta: return rgb(70, 90, 120);
    case NodeKind::Preview: return rgb(60, 110, 100);
    case NodeKind::ImageFileWriter: return rgb(60, 100, 70);
    }
    return rgb(80, 80, 80);
}

ImU32 categoryColor(ElementCategory category) {
    switch (category) {
    case ElementCategory::Buffer: return rgb(120, 92, 44);
    case ElementCategory::Uniform: return rgb(46, 120, 76);
    case ElementCategory::Image: return rgb(44, 88, 150);
    case ElementCategory::Compute: return rgb(150, 60, 110);
    case ElementCategory::Graphics: return rgb(170, 84, 40);
    case ElementCategory::Sync: return rgb(90, 90, 100);
    case ElementCategory::Group: return rgb(122, 60, 150);
    case ElementCategory::Other: return rgb(80, 80, 80);
    }
    return rgb(80, 80, 80);
}

ImU32 brighten(ImU32 color, float amount) {
    ImVec4 c = ImGui::ColorConvertU32ToFloat4(color);
    c.x = std::min(1.0f, c.x + amount);
    c.y = std::min(1.0f, c.y + amount);
    c.z = std::min(1.0f, c.z + amount);
    return ImGui::ColorConvertFloat4ToU32(c);
}

ImVec4 severityColor(Severity severity) {
    switch (severity) {
    case Severity::Error: return ImVec4(1.0f, 0.42f, 0.38f, 1.0f);
    case Severity::Warning: return ImVec4(1.0f, 0.8f, 0.3f, 1.0f);
    case Severity::Info: return ImVec4(0.6f, 0.7f, 0.8f, 1.0f);
    }
    return ImVec4(1, 1, 1, 1);
}

const char* severityLabel(Severity severity) {
    switch (severity) {
    case Severity::Error: return "error";
    case Severity::Warning: return "warning";
    case Severity::Info: return "info";
    }
    return "";
}

// Draws a pin's icon as an item of `size`: squares for Gaussians, triangles
// for cameras, circles for images, diamonds for tensors; filled when
// connected.
void pinIcon(PinType type, bool connected, float size) {
    const ImVec2 min = ImGui::GetCursorScreenPos();
    ImGui::Dummy(ImVec2(size, size));
    ImDrawList* drawList = ImGui::GetWindowDrawList();
    const ImVec2 c(min.x + size * 0.5f, min.y + size * 0.5f);
    const float r = size * 0.3f;
    const ImU32 color = pinColor(type);
    switch (type) {
    case PinType::GaussiansCpu:
    case PinType::GaussiansGpu:
        if (connected) {
            drawList->AddRectFilled(ImVec2(c.x - r, c.y - r), ImVec2(c.x + r, c.y + r), color);
        } else {
            drawList->AddRect(ImVec2(c.x - r, c.y - r), ImVec2(c.x + r, c.y + r), color, 0.0f, 0, 1.5f);
        }
        break;
    case PinType::Camera: {
        const ImVec2 a(c.x - r, c.y - r), b(c.x + r, c.y), d(c.x - r, c.y + r);
        if (connected) {
            drawList->AddTriangleFilled(a, b, d, color);
        } else {
            drawList->AddTriangle(a, b, d, color, 1.5f);
        }
        break;
    }
    case PinType::Image:
        if (connected) {
            drawList->AddCircleFilled(c, r, color);
        } else {
            drawList->AddCircle(c, r, color, 0, 1.5f);
        }
        break;
    case PinType::NumberCpu:
    case PinType::NumberGpu:
        if (connected) {
            drawList->AddCircleFilled(c, r * 0.7f, color);
        } else {
            drawList->AddCircle(c, r * 0.7f, color, 0, 1.5f);
        }
        break;
    case PinType::TransformGpu:
        if (connected) {
            drawList->AddRectFilled(ImVec2(c.x - r, c.y - r), ImVec2(c.x + r, c.y + r), color, r * 0.5f);
        } else {
            drawList->AddRect(ImVec2(c.x - r, c.y - r), ImVec2(c.x + r, c.y + r), color, r * 0.5f, 0, 1.5f);
        }
        break;
    case PinType::Tensor: {
        const float d = r * 1.25f;
        const ImVec2 top(c.x, c.y - d), right(c.x + d, c.y), bottom(c.x, c.y + d), left(c.x - d, c.y);
        if (connected) {
            drawList->AddQuadFilled(top, right, bottom, left, color);
        } else {
            drawList->AddQuad(top, right, bottom, left, color, 1.5f);
        }
        break;
    }
    }
}

void elementPinIcon(bool connected, float size) {
    const ImVec2 min = ImGui::GetCursorScreenPos();
    ImGui::Dummy(ImVec2(size, size));
    const ImVec2 c(min.x + size * 0.5f, min.y + size * 0.5f);
    const ImU32 color = rgb(170, 170, 190);
    if (connected) {
        ImGui::GetWindowDrawList()->AddCircleFilled(c, size * 0.25f, color);
    } else {
        ImGui::GetWindowDrawList()->AddCircle(c, size * 0.25f, color, 0, 1.2f);
    }
}

bool inputText(const char* label, std::string& value) {
    char buffer[1024];
    const size_t n = std::min(value.size(), sizeof(buffer) - 1);
    value.copy(buffer, n);
    buffer[n] = '\0';
    if (ImGui::InputText(label, buffer, sizeof(buffer))) {
        value = buffer;
        return true;
    }
    return false;
}

bool inputTextMultiline(const char* label, std::string& value) {
    char buffer[2048];
    const size_t n = std::min(value.size(), sizeof(buffer) - 1);
    value.copy(buffer, n);
    buffer[n] = '\0';
    const ImVec2 size(-1.0f, ImGui::GetTextLineHeight() * 4.0f);
    if (ImGui::InputTextMultiline(label, buffer, sizeof(buffer), size)) {
        value = buffer;
        return true;
    }
    return false;
}

bool sliderUint(const char* label, uint32_t& value, uint32_t min, uint32_t max) {
    int v = static_cast<int>(value);
    if (ImGui::SliderInt(label, &v, static_cast<int>(min), static_cast<int>(max))) {
        value = static_cast<uint32_t>(v);
        return true;
    }
    return false;
}

bool comboUint(const char* label, uint32_t& value, std::initializer_list<uint32_t> choices) {
    bool changed = false;
    const std::string preview = std::to_string(value);
    if (ImGui::BeginCombo(label, preview.c_str())) {
        for (uint32_t choice : choices) {
            const bool selected = choice == value;
            if (ImGui::Selectable(std::to_string(choice).c_str(), selected)) {
                changed = choice != value;
                value = choice;
            }
        }
        ImGui::EndCombo();
    }
    return changed;
}

// Sample files, relative to the klartraum sources (see resolveInputPath).
constexpr const char* kSampleScene = "3rdparty/spz/samples/racoonfamily.spz";
constexpr const char* kSampleLantern = "data/lantern.spz";
// The combined-scenes example puts the lantern on the lawn in front of the
// raccoon stump and looks at both.
const TransformGaussiansParams kLanternPlacement{{0.68f, -1.0f, -0.68f}, {0.0f, 20.0f, 0.0f}, 1.6f};
const CameraParams kLanternView{1.57f, -0.35f, 1.9f, {-0.45f, 0.35f, -1.0f}, UpAxis::Y};
constexpr const char* kSampleImage = "data/lantern.jpg";
constexpr const char* kSampleEncoder = "data/onnx/simple_encoder.onnx";
constexpr const char* kSampleDecoder = "data/onnx/simple_decoder.onnx";
// Stable Diffusion 1.5 exported by klartraum's scripts/sd15_onnx/export_denoiser.py
// (not part of the klartraum repository; see the node documentation).
constexpr const char* kSampleSdModels = "data/onnx/sd15_denoiser_256";
constexpr uint32_t kSampleSdSize = 256;
constexpr const char* kSamplePrompt =
    "a realistic photograph of a traditional Japanese stone lantern in a green garden, single gray granite garden "
    "lantern, centered, moss, natural daylight";
constexpr const char* kSampleNegativePrompt = "blurry, distorted, oversaturated, text";
constexpr const char* kSampleSdBackgroundPrompt =
    "a realistic photograph of a quiet japanese garden with moss and stones, soft daylight";

std::string sdSample(const char* file) {
    return (std::filesystem::path(kSampleSdModels) / file).generic_string();
}

std::string defaultScenePath() {
    return kSampleScene;
}

double elapsedMs(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

std::string fileName(const std::string& path) {
    return path.empty() ? std::string("(no file)") : std::filesystem::path(path).filename().string();
}

// Text field for a uint32 parameter; values below `min` are raised to it.
bool inputUint(const char* label, uint32_t& value, uint32_t min = 1) {
    int v = static_cast<int>(value);
    if (ImGui::InputInt(label, &v, 1, 16) || ImGui::IsItemDeactivatedAfterEdit()) {
        const uint32_t clamped = static_cast<uint32_t>(std::max(v, static_cast<int>(min)));
        if (clamped != value) {
            value = clamped;
            return true;
        }
    }
    return false;
}

} // namespace

std::optional<Example> exampleFromName(std::string_view name) {
    if (name == "gaussian-splatting") return Example::GaussianSplatting;
    if (name == "autoencoder") return Example::Autoencoder;
    if (name == "splat-autoencoder") return Example::SplatAutoencoder;
    if (name == "combined-scenes") return Example::CombinedScenes;
    if (name == "animated-scenes") return Example::AnimatedScenes;
    if (name == "stable-diffusion") return Example::StableDiffusion;
    if (name == "stable-diffusion-background") return Example::StableDiffusionBackground;
    return std::nullopt;
}

StudioApp::StudioApp(klartraum::KlartraumEngine& engine, StudioOptions options, GLFWwindow* window)
    : engine_(engine), window_(window), options_(std::move(options)),
      retained_(std::make_unique<RetainedResults>(engine.getVulkanContext())) {
    profiling_ = options_.profiling;

    // Editor layouts are stored in the graph files, not in NodeEditor.json.
    ed::Config config;
    config.SettingsFile = nullptr;
    config.EnableSmoothZoom = true;
    authoringEditor_ = ed::CreateEditor(&config);
    compiledEditor_ = ed::CreateEditor(&config);
    setupStyle();

    camera_ = std::make_shared<klartraum::InterfaceCameraOrbit>(klartraum::InterfaceCameraOrbit::UpDirection::Y);
    engine_.setInterfaceCamera(camera_);
    pushCameraParams(CameraParams{});

    if (!options_.graphFile.empty()) {
        if (!openGraph(options_.graphFile)) {
            loadExample(options_.example);
        }
    } else {
        loadExample(options_.example);
    }
    // Compile right away so the first frame shows the scene.
    updatePlan();
    if (!appliedPlan_) {
        installBuilder(std::nullopt, {});
    }
}

StudioApp::~StudioApp() {
    vkDeviceWaitIdle(engine_.getVulkanContext().getDevice());
    // The builder refers to this object.
    engine_.setGraphBuilder(nullptr);
    ed::DestroyEditor(compiledEditor_);
    ed::DestroyEditor(authoringEditor_);
}

// ---------------------------------------------------------------------------
// Graph documents

void StudioApp::newDefaultGraph() {
    const std::string scene = options_.scenePath.empty() ? defaultScenePath() : options_.scenePath;
    setGraph(makeGaussianSplattingGraph(scene, options_.backend), {});
}

void StudioApp::loadExample(Example example) {
    switch (example) {
    case Example::GaussianSplatting:
        newDefaultGraph();
        break;
    case Example::Autoencoder:
        setGraph(makeAutoencoderGraph(kSampleImage, kSampleEncoder, kSampleDecoder, "autoencoded.png"), {});
        break;
    case Example::CombinedScenes: {
        const std::string scene = options_.scenePath.empty() ? defaultScenePath() : options_.scenePath;
        setGraph(makeCombinedScenesGraph(scene, SceneParams{kSampleLantern, true}, kLanternPlacement, kLanternView), {});
        break;
    }
    case Example::AnimatedScenes: {
        const std::string scene = options_.scenePath.empty() ? defaultScenePath() : options_.scenePath;
        MakeTransformParams placement;
        placement.translation = kLanternPlacement.translation;
        placement.scale = kLanternPlacement.scale;
        // Swings 30 degrees either way around the combined example's 20, once every 4 seconds.
        const SineParams swing{30.0f, 0.25f, 0.0f, kLanternPlacement.rotation[1]};
        setGraph(makeAnimatedScenesGraph(scene, SceneParams{kSampleLantern, true}, placement, swing, kLanternView), {});
        break;
    }
    case Example::SplatAutoencoder: {
        const std::string scene = options_.scenePath.empty() ? defaultScenePath() : options_.scenePath;
        setGraph(makeSplatAutoencoderGraph(scene, kSampleEncoder, kSampleDecoder), {});
        break;
    }
    case Example::StableDiffusion:
        setGraph(makeStableDiffusionGraph(kSampleSdModels, kSampleSdSize, kSamplePrompt, kSampleNegativePrompt,
                                          "stable_diffusion.png"),
                 {});
        break;
    case Example::StableDiffusionBackground:
        setGraph(makeStableDiffusionBackgroundGraph(kSampleSdModels, kSampleSdSize, kSampleSdBackgroundPrompt,
                                                    kSampleNegativePrompt, SceneParams{kSampleLantern, true},
                                                    kLanternPlacement, kLanternView),
                 {});
        break;
    }
    // The examples come with a layout of their own; start the view on it.
    fitRequested_ = true;
}

void StudioApp::setGraph(Graph graph, std::filesystem::path file) {
    // Results kept for the previous graph's live graph do not belong to this one.
    if (appliedPlan_ && !appliedPlan_->retained.empty()) {
        vkDeviceWaitIdle(engine_.getVulkanContext().getDevice());
        appliedPlan_.reset();
        installBuilder(std::nullopt, {});
    }
    retained_->clear();
    editPath_.clear();
    editFrom_.clear();
    graph_ = std::move(graph);
    graph_.touch();
    file_ = std::move(file);
    modified_ = false;
    selectedNode_ = -1;
    nodesToPlace_.clear();
    for (const auto& node : graph_.nodes()) {
        nodesToPlace_.push_back(node.id);
    }
    ed::SetCurrentEditor(authoringEditor_);
    ed::ClearSelection();
    ed::SetCurrentEditor(nullptr);
    fitRequested_ = true;
    // A new camera node's view replaces the current one.
    if (appliedPlan_) {
        appliedPlan_->cameraNode = -1;
    }
    // The plan and the revisions refer to the previous graph; revisions are
    // counted per graph, so the new one's may coincide with the old one's.
    plan_ = {};
    plannedRevision_ = ~uint64_t{0};
    lastRunRevision_ = ~uint64_t{0};
    // Run results belong to the previous graph.
    lastRun_.reset();
    runError_.clear();
    previews_.clear();
    if (compiledSource_ == 1) {
        compiledLayoutDirty_ = true;
    }
    updateWindowTitle();
}

bool StudioApp::openGraph(const std::filesystem::path& path) {
    try {
        setGraph(loadGraph(path), path);
        setStatus("Opened " + path.string());
        return true;
    } catch (const std::exception& e) {
        setStatus(std::string("Open failed: ") + e.what(), true);
        return false;
    }
}

bool StudioApp::saveGraphTo(const std::filesystem::path& path) {
    commitDefinition();
    try {
        saveGraph(graph_, path);
        file_ = path;
        modified_ = false;
        updateWindowTitle();
        setStatus("Saved " + path.string());
        return true;
    } catch (const std::exception& e) {
        setStatus(std::string("Save failed: ") + e.what(), true);
        return false;
    }
}

std::optional<std::filesystem::path> StudioApp::resolveInputPath(const std::string& path) const {
    namespace fs = std::filesystem;
    if (path.empty()) {
        return std::nullopt;
    }
    std::vector<fs::path> candidates{fs::path(path)};
    if (fs::path(path).is_relative()) {
        if (!file_.empty()) {
            candidates.push_back(file_.parent_path() / path);
        }
        candidates.push_back(fs::path(KLARTRAUM_SOURCE_DIR) / path);
    }
    for (const auto& candidate : candidates) {
        std::error_code ec;
        if (fs::is_regular_file(candidate, ec)) {
            return fs::absolute(candidate, ec);
        }
    }
    return std::nullopt;
}

std::filesystem::path StudioApp::resolveOutputPath(const std::string& path) const {
    const std::filesystem::path p(path);
    if (p.is_absolute() || file_.empty()) {
        return std::filesystem::absolute(p);
    }
    return file_.parent_path() / p;
}

std::shared_ptr<const OnnxModelInfo> StudioApp::onnxInfo(const std::string& path, std::string& error) {
    const auto resolved = resolveInputPath(path);
    if (!resolved) {
        error = "File not found: " + path;
        return nullptr;
    }
    return onnxInfo_.get(*resolved, &error);
}

// ---------------------------------------------------------------------------
// Compilation

void StudioApp::syncOnnxPins() {
    for (auto& node : graph_.nodes()) {
        if (node.kind != NodeKind::OnnxModel || node.as<OnnxModelParams>().path.empty()) {
            continue;
        }
        std::string error;
        const auto info = onnxInfo(node.as<OnnxModelParams>().path, error);
        if (!info) {
            continue;  // the plan reports it
        }
        std::vector<std::string> inputs, outputs;
        for (const auto& input : info->inputs) {
            inputs.push_back(input.name);
        }
        for (const auto& output : info->outputs) {
            outputs.push_back(output.name);
        }
        if (const int removed = graph_.setOnnxPins(node.id, std::move(inputs), std::move(outputs)); removed > 0) {
            setStatus(std::format("{}: the model's pins changed; {} link(s) removed", node.title, removed), true);
        }
    }
}

void StudioApp::updatePlan() {
    if (graph_.revision() != plannedRevision_) {
        // An ONNX Model node's pins follow its model file.
        syncOnnxPins();
        plan_ = planGraph(
            graph_, [this](const std::string& path, std::string& error) { return onnxInfo(path, error); },
            [this](const std::string& path) { return resolveInputPath(path).has_value(); });
        plannedRevision_ = graph_.revision();
    }

    const bool userIsEditing = ImGui::GetCurrentContext() && ImGui::IsAnyItemActive();
    if (plan_.run && autoRun_ && !userIsEditing && lastRunRevision_ != graph_.revision()) {
        runRequested_ = true;
    }

    if (!plan_.ok()) {
        // Without a Present node nothing should render live; with a broken
        // one, the last graph that compiled keeps running.
        const bool hasPresent = std::any_of(graph_.nodes().begin(), graph_.nodes().end(),
                                            [](const Node& n) { return n.kind == NodeKind::Present; });
        if (!hasPresent && appliedPlan_) {
            vkDeviceWaitIdle(engine_.getVulkanContext().getDevice());
            appliedPlan_.reset();
            failedPlan_.reset();
            installBuilder(std::nullopt, {});
        }
        return;
    }
    const LivePlan& pending = *plan_.live;

    // The live graph reads results of the last Run. Without them yet, Run
    // once on its own; after that only on request.
    if (!retained_->has(pending.retained)) {
        if (plan_.run && autoRunRevision_ != graph_.revision()) {
            autoRunRevision_ = graph_.revision();
            runRequested_ = true;
        }
        return;
    }

    // New storage for retained results needs a live graph built with it.
    const bool rebuild = !appliedPlan_ || pending.needsRebuildFrom(*appliedPlan_) ||
                         (!pending.retained.empty() && appliedGeneration_ != retained_->generation());
    if (rebuild) {
        const bool alreadyFailed = failedPlan_ && !pending.needsRebuildFrom(*failedPlan_);
        if (applyRequested_ || (autoApply_ && !alreadyFailed && !userIsEditing)) {
            applyRequested_ = false;
            apply(pending, plan_.flat.graph, plan_.flat.topNode);
        }
        return;
    }

    // Same pipelines; only which authoring nodes they stand for may differ.
    // Equal signatures list corresponding nodes at the same positions.
    if (appliedPlan_->cameraNode != pending.cameraNode) {
        if (const Node* camera = plan_.flat.graph.findNode(pending.cameraNode)) {
            pushCameraParams(camera->as<CameraParams>());
        }
    }
    if (appliedPlan_->nodes != pending.nodes) {
        std::map<int, int> renamed;
        for (size_t i = 0; i < pending.nodes.size(); ++i) {
            renamed[appliedPlan_->nodes[i]] = pending.nodes[i];
        }
        // The compiled view names document nodes; top-level nodes keep their
        // ids when flattened.
        for (auto& element : compiled_.nodes) {
            if (auto it = renamed.find(element.owner); it != renamed.end()) {
                element.owner = plan_.flat.top(it->second);
            }
        }
        for (auto& binding : liveBindings_) {
            if (auto it = renamed.find(binding.node); it != renamed.end()) {
                binding.node = it->second;
            }
        }
        // The running graph now stands for the current one.
        appliedGraph_ = plan_.flat.graph;
        appliedTopNode_ = plan_.flat.topNode;
    }
    appliedPlan_ = pending;
}

std::shared_ptr<const std::vector<klartraum::Gaussian3D>> StudioApp::loadScene(const std::string& path, bool flipY) {
    const auto resolved = resolveInputPath(path);
    if (!resolved) {
        throw std::runtime_error("scene file not found: " + path);
    }
    const auto key = std::make_pair(resolved->string(), flipY);
    if (auto it = scenes_.find(key); it != scenes_.end()) {
        return it->second;
    }
    const auto start = std::chrono::steady_clock::now();
    auto scene = std::make_shared<const std::vector<klartraum::Gaussian3D>>(
        klartraum::loadGaussiansSpz(resolved->string(), flipY));
    if (scene->empty()) {
        throw std::runtime_error("scene has no Gaussians: " + path);
    }
    logHostStep(std::format("decoded {} ({} Gaussians)", fileName(path), scene->size()), elapsedMs(start));
    scenes_[key] = scene;
    return scene;
}

std::shared_ptr<klartraum::GaussianDataStandard> StudioApp::loadGaussians(const std::vector<GaussianPart>& parts) {
    const std::string key = partsKey(parts);
    if (auto it = models_.find(key); it != models_.end()) {
        logHostStep(std::format("reused uploaded Gaussians ({})", it->second->count()), 0.0);
        return it->second;
    }
    auto gaussians =
        assembleGaussians(parts, [this](const std::string& path, bool flipY) { return loadScene(path, flipY); });
    if (parts.size() > 1 || !(parts[0].transform == Similarity{})) {
        // Placing and concatenating is timed together with the upload below.
        logHostStep(std::format("assembled {} Gaussians from {} part(s)", gaussians.size(), parts.size()), 0.0);
    }
    const auto start = std::chrono::steady_clock::now();
    auto model = std::make_shared<klartraum::GaussianDataStandard>(engine_.getVulkanContext(), std::move(gaussians));
    logHostStep(std::format("uploaded {:.0f} MB", gaussianBytes(model->count()) / 1e6), elapsedMs(start));
    models_[key] = model;
    return model;
}

void StudioApp::logHostStep(const std::string& step, double milliseconds) {
    if (hostLog_) {
        hostLog_->push_back(milliseconds > 0.0 ? std::format("{} ({:.0f} ms)", step, milliseconds) : step);
    }
}

std::shared_ptr<klartraum::GaussianDataStandard> StudioApp::uploadedGaussians(const Graph& graph, int node) const {
    try {
        auto it = models_.find(partsKey(gaussianParts(graph, node)));
        return it == models_.end() ? nullptr : it->second;
    } catch (const std::exception&) {
        return nullptr;  // inputs not connected
    }
}

void StudioApp::drawExecutionInfo(const Node& node) {
    const auto& info = kindInfo(node.kind);
    ImGui::SeparatorText("Runs as");
    executionBadge(info);
    ImGui::TextWrapped("Implemented by %s", std::string(info.implementedBy).c_str());
    ImGui::TextWrapped("Runs %s.", std::string(info.timing).c_str());

    const bool graphNode = info.implementation == Implementation::ComputeGraph;
    auto describe = [&](const char* part, bool used, const ElementGraph* compiled) {
        if (!used) {
            return;
        }
        if (!graphNode) {
            ImGui::BulletText("Prepares data for the %s graph; not part of a klartraum graph.", part);
            return;
        }
        int own = 0, added = 0;
        if (compiled) {
            for (const auto& element : compiled->nodes) {
                if (element.owner == node.id) {
                    ++(element.inserted ? added : own);
                }
            }
        }
        if (!compiled || compiled->empty()) {
            ImGui::BulletText("In the %s klartraum graph (not built yet).", part);
        } else if (own + added == 0) {
            ImGui::BulletText("In the %s klartraum graph, without elements of its own here.", part);
        } else if (added > 0) {
            ImGui::BulletText("In the %s klartraum graph: %d elements, %d more added by the studio.", part, own, added);
        } else {
            ImGui::BulletText("In the %s klartraum graph: %d elements.", part, own);
        }
    };
    if (editingDefinition()) {
        ImGui::TextDisabled("Part of a meta node definition; see its instances for how they run.");
        return;
    }
    const bool live = plan_.live && plan_.flat.covers(plan_.live->nodes, node.id);
    const bool run = plan_.run && plan_.flat.covers(plan_.run->nodes, node.id);
    describe("live", live, appliedPlan_ && appliedPlan_->contains(node.id) ? &compiled_ : nullptr);
    describe("run", run, lastRun_ ? &lastRun_->compiled : nullptr);
    if (!live && !run) {
        ImGui::TextDisabled("Not used by the live or the run graph.");
    }
    if (node.kind == NodeKind::UploadGaussians) {
        if (auto model = uploadedGaussians(graph_, node.id)) {
            ImGui::Text("Uploaded: %u Gaussians, %.0f MB of GPU buffers", model->count(),
                        gaussianBytes(model->count()) / 1e6);
        }
    }
}

std::optional<size_t> StudioApp::sceneCount(const SceneParams& scene) {
    const auto resolved = resolveInputPath(scene.path);
    if (!resolved) {
        return std::nullopt;
    }
    auto it = scenes_.find(std::make_pair(resolved->string(), scene.flipY));
    return it == scenes_.end() ? std::nullopt : std::optional<size_t>(it->second->size());
}

RunContext StudioApp::runContext() {
    RunContext context;
    context.resolveInput = [this](const std::string& path) {
        const auto resolved = resolveInputPath(path);
        if (!resolved) {
            throw std::runtime_error("file not found: " + path);
        }
        return *resolved;
    };
    context.resolveOutput = [this](const std::string& path) { return resolveOutputPath(path); };
    context.loadGaussians = [this](const std::vector<GaussianPart>& parts) { return loadGaussians(parts); };
    context.hostStep = [this](const std::string& step, double milliseconds) { logHostStep(step, milliseconds); };
    context.time = secondsSinceStart();
    context.onnxInfo = [this](const std::string& path, std::string& error) { return onnxInfo(path, error); };
    context.retained = retained_.get();
    return context;
}

void StudioApp::installBuilder(const std::optional<LivePlan>& plan, const Graph& graph,
                               const std::map<int, int>& topNode) {
    if (!plan) {
        compiled_ = {};
        compiledSignature_.clear();
        liveBindings_.clear();
        // Without a live graph the window is only cleared; the builder keeps
        // it resizable.
        engine_.setGraphBuilder([](klartraum::KlartraumEngine& e) { e.add(e.createRenderPass()); });
        return;
    }
    // The engine runs the builder now and again after every swapchain
    // recreation, so it must not throw: errors are reported via builderError_.
    // It keeps a copy of the graph, which the user goes on editing.
    engine_.setGraphBuilder([this, plan = *plan, graph, topNode](klartraum::KlartraumEngine& e) {
        liveHostSteps_.clear();
        hostLog_ = &liveHostSteps_;
        try {
            BuiltGraph built = buildLiveGraph(e, graph, plan, runContext());
            compiled_ = introspect(built.root, built.owners, plan.presentNode, built.inserted);
            for (auto& element : compiled_.nodes) {
                if (auto it = topNode.find(element.owner); it != topNode.end()) {
                    element.owner = it->second;
                }
            }
            liveBindings_ = std::move(built.bindings);
            applyBindings(graph, liveBindings_, secondsSinceStart());
            builderError_.clear();
        } catch (const std::exception& ex) {
            liveBindings_.clear();
            e.clearComputeGraphs();
            e.setCameraUBO(nullptr);
            compiled_ = {};
            builderError_ = ex.what();
        }
        hostLog_ = nullptr;

        // Keep the user's arrangement across swapchain rebuilds; lay out anew
        // only when the graph's structure changed.
        std::string signature;
        for (const auto& node : compiled_.nodes) {
            signature += node.type + "/" + node.name + ";";
        }
        if (signature != compiledSignature_) {
            compiledSignature_ = std::move(signature);
            compiledLayoutDirty_ = true;
            selectedElement_ = -1;
        }
    });
}

bool StudioApp::apply(const LivePlan& plan, const Graph& graph, const std::map<int, int>& topNode) {
    // The GUI runs between frames; once the GPU is idle the running graphs
    // can be released.
    vkDeviceWaitIdle(engine_.getVulkanContext().getDevice());
    if (profiling_) {
        engine_.enableProfiling();
    } else {
        engine_.disableProfiling();
    }
    timings_.clear();

    const bool newCamera = !appliedPlan_ || appliedPlan_->cameraNode != plan.cameraNode;
    installBuilder(plan, graph, topNode);

    if (!builderError_.empty()) {
        applyError_ = builderError_;
        failedPlan_ = plan;
        setStatus("Compile failed: " + applyError_, true);
        // Fall back to the last graph that compiled.
        if (appliedPlan_) {
            installBuilder(appliedPlan_, appliedGraph_, appliedTopNode_);
        } else {
            installBuilder(std::nullopt, {});
        }
        return false;
    }

    appliedPlan_ = plan;
    appliedGraph_ = graph;
    appliedTopNode_ = topNode;
    appliedGeneration_ = retained_->generation();
    failedPlan_.reset();
    applyError_.clear();
    if (newCamera) {
        if (const Node* camera = graph.findNode(plan.cameraNode)) {
            pushCameraParams(camera->as<CameraParams>());
        }
    }
    // Drop Gaussians that the running graph no longer uses. The scene files
    // it uses stay loaded, so moving a scene does not read them again.
    std::set<std::string> keepModels;
    std::set<std::pair<std::string, bool>> keepScenes;
    for (int id : plan.nodes) {
        const Node* node = graph.findNode(id);
        if (node->kind == NodeKind::UploadGaussians) {
            const auto parts = gaussianParts(graph, id);
            keepModels.insert(partsKey(parts));
            for (const auto& part : parts) {
                if (auto resolved = resolveInputPath(part.path)) {
                    keepScenes.emplace(resolved->string(), part.flipY);
                }
            }
        }
    }
    std::erase_if(models_, [&](const auto& entry) { return !keepModels.contains(entry.first); });
    std::erase_if(scenes_, [&](const auto& entry) { return !keepScenes.contains(entry.first); });

    setStatus(std::format("Compiled {} elements", compiled_.nodes.size()));
    return true;
}

void StudioApp::pushCameraParams(const CameraParams& params) {
    camera_->setUpDirection(params.up == UpAxis::Y ? klartraum::InterfaceCameraOrbit::UpDirection::Y
                                                   : klartraum::InterfaceCameraOrbit::UpDirection::Z);
    camera_->setAzimuth(params.azimuth);
    camera_->setElevation(params.elevation);
    camera_->setDistance(params.distance);
    camera_->setPosition(glm::vec3(params.target[0], params.target[1], params.target[2]));
}

void StudioApp::syncCamera() {
    // The mouse moves the orbit camera; mirror its state into the active
    // camera node so the inspector shows it and saving keeps the view.
    if (!appliedPlan_) {
        return;
    }
    Node* node = graph_.findNode(appliedPlan_->cameraNode);
    if (!node || node->kind != NodeKind::Camera) {
        return;
    }
    auto& params = node->as<CameraParams>();
    params.azimuth = static_cast<float>(camera_->getAzimuth());
    params.elevation = static_cast<float>(camera_->getElevation());
    params.distance = static_cast<float>(camera_->getDistance());
    const glm::vec3& target = camera_->getPosition();
    params.target = {target.x, target.y, target.z};
}

void StudioApp::refreshProfiling() {
    if (!profiling_ || !appliedPlan_) {
        return;
    }
    const double now = ImGui::GetTime();
    if (now - lastProfileTime_ < kProfileInterval) {
        return;
    }
    lastProfileTime_ = now;
    timings_.clear();
    for (const auto& [label, ms] : engine_.getProfilingResults()) {
        timings_.emplace(label, ms);  // keeps the first of duplicate labels
    }
}

// ---------------------------------------------------------------------------
// UI

void StudioApp::setupStyle() {
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 6.0f;
    style.FrameRounding = 4.0f;
    style.GrabRounding = 4.0f;
    style.TabRounding = 4.0f;
    style.WindowBorderSize = 1.0f;
    style.Colors[ImGuiCol_WindowBg].w = 0.92f;

    for (auto* editor : {authoringEditor_, compiledEditor_}) {
        ed::SetCurrentEditor(editor);
        ed::Style& nodes = ed::GetStyle();
        nodes.NodePadding = ImVec4(8.0f, 6.0f, 8.0f, 8.0f);
        nodes.NodeRounding = 6.0f;
        nodes.NodeBorderWidth = 1.5f;
        nodes.PinRounding = 0.0f;
        nodes.LinkStrength = 80.0f;
        nodes.Colors[ed::StyleColor_Bg] = ImColor(24, 24, 30, 235);
        nodes.Colors[ed::StyleColor_Grid] = ImColor(90, 90, 110, 50);
        nodes.Colors[ed::StyleColor_NodeBg] = ImColor(40, 40, 48, 245);
        nodes.Colors[ed::StyleColor_NodeBorder] = ImColor(90, 90, 104, 255);
        nodes.Colors[ed::StyleColor_PinRect] = ImColor(120, 170, 255, 60);
        nodes.Colors[ed::StyleColor_PinRectBorder] = ImColor(120, 170, 255, 110);
    }
    ed::SetCurrentEditor(nullptr);
}

void StudioApp::setStatus(std::string message, bool error) {
    status_ = std::move(message);
    statusIsError_ = error;
    statusTime_ = ImGui::GetCurrentContext() ? ImGui::GetTime() : 0.0;
}

void StudioApp::updateWindowTitle() {
    std::string title = "Klartraum Studio - " + (file_.empty() ? std::string("untitled") : file_.filename().string());
    if (modified_) {
        title += " *";
    }
    if (title != windowTitle_) {
        windowTitle_ = title;
        if (window_) {
            glfwSetWindowTitle(window_, windowTitle_.c_str());
        }
    }
}

void StudioApp::run() {
    runRequested_ = false;
    // Run writes the results the live graph reads; frames still in flight
    // must be done with them first.
    vkDeviceWaitIdle(engine_.getVulkanContext().getDevice());
    lastRunRevision_ = graph_.revision();
    if (!plan_.run) {
        runError_ = "Nothing to run: add a Preview or Image File Writer, and fix the errors of the nodes feeding it.";
        setStatus(runError_, true);
        return;
    }

    runHostSteps_.clear();
    hostLog_ = &runHostSteps_;
    struct StopLogging {
        std::vector<std::string>*& log;
        ~StopLogging() { log = nullptr; }
    } stopLogging{hostLog_};
    try {
        RunResult result = runGraph(engine_.getVulkanContext(), plan_.flat.graph, *plan_.run, runContext());
        // The compiled view names document nodes.
        for (auto& element : result.compiled.nodes) {
            element.owner = element.owner < 0 ? element.owner : plan_.flat.top(element.owner);
        }
        previews_.clear();
        for (const auto& [node, image] : result.images) {
            previews_[node] = std::make_unique<PreviewTexture>(engine_.getVulkanContext(), image);
        }
        std::string message = std::format("Run finished in {:.0f} ms", result.milliseconds);
        for (const auto& file : result.written) {
            message += ", wrote " + file.filename().string();
        }
        lastRun_ = std::move(result);
        runError_.clear();
        // With new results, a live graph that failed for want of them is
        // tried again.
        failedPlan_.reset();
        setStatus(message);
        if (compiledSource_ == 1) {
            compiledLayoutDirty_ = true;
        }
    } catch (const std::exception& e) {
        runError_ = e.what();
        setStatus("Run failed: " + runError_, true);
    }
}

void StudioApp::updateLiveValues() {
    if (liveBindings_.empty() || !appliedPlan_) {
        return;
    }
    // Live parameters are edited in the current graph; while it still
    // compiles to the running pipelines, its values apply.
    const bool current = plan_.live && !plan_.live->needsRebuildFrom(*appliedPlan_);
    try {
        applyBindings(current ? liveValuesGraph() : appliedGraph_, liveBindings_, secondsSinceStart());
    } catch (const std::exception&) {
        // An input was disconnected; the plan reports it, and the values stay.
    }
}

Graph StudioApp::liveValuesGraph() const {
    // Values edited in the document apply live; inner nodes of meta nodes
    // take them through their instance's exposed parameters.
    const bool hasMeta = std::any_of(graph_.nodes().begin(), graph_.nodes().end(),
                                     [](const Node& n) { return n.kind == NodeKind::Meta; });
    return hasMeta ? flatten(graph_).graph : graph_;
}

double StudioApp::secondsSinceStart() const {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
}

void StudioApp::drawGui() {
    // Actions requested by the editor or inspector last frame, which change
    // which nodes exist or which graph is shown.
    if (pendingOpen_ >= 0) {
        openMetaNode(std::exchange(pendingOpen_, -1));
    }
    if (pendingUngroup_ >= 0) {
        ungroup(std::exchange(pendingUngroup_, -1));
    }
    if (pendingClose_ >= 0) {
        closeDefinitions(static_cast<size_t>(std::exchange(pendingClose_, -1)));
    }
    if (ImGui::Shortcut(ImGuiMod_Ctrl | ImGuiKey_G, ImGuiInputFlags_RouteGlobal)) {
        pendingGroup_ = true;
    }
    if (std::exchange(pendingGroup_, false)) {
        groupSelection();
    }
    commitDefinition();
    syncCamera();
    updateLiveValues();
    if (runRequested_) {
        run();
    }
    if (ImGui::Shortcut(ImGuiKey_F5, ImGuiInputFlags_RouteGlobal)) {
        requestRun();
    }

    if (ImGui::Shortcut(ImGuiMod_Ctrl | ImGuiKey_S, ImGuiInputFlags_RouteGlobal)) {
        if (file_.empty()) {
            fileDialog_ = FileDialog::SaveAs;
        } else {
            saveGraphTo(file_);
        }
    }
    if (ImGui::Shortcut(ImGuiMod_Ctrl | ImGuiKey_O, ImGuiInputFlags_RouteGlobal)) {
        fileDialog_ = FileDialog::Open;
    }

    drawMenuBar();
    drawOverview();
    drawGraphWindow();
    drawInspector();
    drawFileDialog();

    updatePlan();
    refreshProfiling();
    updateWindowTitle();
}

void StudioApp::drawMenuBar() {
    if (!ImGui::BeginMainMenuBar()) {
        return;
    }
    if (ImGui::BeginMenu("File")) {
        if (ImGui::BeginMenu("New from Example")) {
            if (ImGui::MenuItem("Gaussian splatting (live)")) {
                loadExample(Example::GaussianSplatting);
            }
            if (ImGui::MenuItem("Image autoencoder (run)")) {
                loadExample(Example::Autoencoder);
            }
            if (ImGui::MenuItem("Splatting autoencoder (offscreen, run)")) {
                loadExample(Example::SplatAutoencoder);
            }
            if (ImGui::MenuItem("Raccoons and lantern (live)")) {
                loadExample(Example::CombinedScenes);
            }
            if (ImGui::MenuItem("Raccoons and swinging lantern (live, GPU)")) {
                loadExample(Example::AnimatedScenes);
            }
            if (ImGui::MenuItem("Stable Diffusion 1.5 (run)")) {
                loadExample(Example::StableDiffusion);
            }
            if (ImGui::MenuItem("Lantern over a Stable Diffusion image (live)")) {
                loadExample(Example::StableDiffusionBackground);
            }
            ImGui::EndMenu();
        }
        if (ImGui::MenuItem("New Empty Graph")) {
            setGraph(Graph{}, {});
        }
        ImGui::Separator();
        if (ImGui::MenuItem("Open...", "Ctrl+O")) {
            fileDialog_ = FileDialog::Open;
        }
        if (ImGui::MenuItem("Save", "Ctrl+S")) {
            if (file_.empty()) {
                fileDialog_ = FileDialog::SaveAs;
            } else {
                saveGraphTo(file_);
            }
        }
        if (ImGui::MenuItem("Save As...")) {
            fileDialog_ = FileDialog::SaveAs;
        }
        ImGui::Separator();
        if (ImGui::MenuItem("Quit", nullptr, false, window_ != nullptr)) {
            glfwSetWindowShouldClose(window_, GLFW_TRUE);
        }
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Graph")) {
        if (ImGui::MenuItem("Run", "F5", false, plan_.run.has_value())) {
            requestRun();
        }
        ImGui::MenuItem("Run on every change", nullptr, &autoRun_);
        ImGui::Separator();
        ImGui::MenuItem("Auto-apply changes", nullptr, &autoApply_);
        if (ImGui::MenuItem("Apply now", nullptr, false, plan_.ok())) {
            applyRequested_ = true;
            failedPlan_.reset();
        }
        ImGui::Separator();
        if (ImGui::MenuItem("Group selected nodes", "Ctrl+G")) {
            pendingGroup_ = true;
        }
        if (ImGui::MenuItem("Back to the graph", nullptr, false, editingDefinition())) {
            closeDefinitions(0);
        }
        ImGui::Separator();
        if (ImGui::MenuItem("Arrange authoring graph")) {
            layoutAuthoringGraph();
        }
        if (ImGui::MenuItem("Arrange compiled graph")) {
            compiledLayoutDirty_ = true;
        }
        ImGui::MenuItem("Hide buffer elements", nullptr, &hideBuffers_);
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("View")) {
        if (ImGui::MenuItem("Authoring graph", nullptr, activeTab_ == 0)) {
            requestedTab_ = 0;
        }
        if (ImGui::MenuItem("Compiled graph", nullptr, activeTab_ == 1)) {
            requestedTab_ = 1;
        }
        ImGui::EndMenu();
    }

    if (!status_.empty() && ImGui::GetTime() - statusTime_ < 8.0) {
        ImGui::Separator();
        ImGui::TextColored(statusIsError_ ? severityColor(Severity::Error) : ImVec4(0.7f, 0.85f, 0.7f, 1.0f), "%s",
                           status_.c_str());
    }
    ImGui::EndMainMenuBar();
}

void StudioApp::drawOverview() {
    const ImGuiIO& io = ImGui::GetIO();
    const float top = ImGui::GetFrameHeight();
    ImGui::SetNextWindowPos(ImVec2(8.0f, top + 8.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(300.0f, 0.0f), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Overview")) {
        ImGui::End();
        return;
    }

    auto& vc = engine_.getVulkanContext();
    const VkExtent2D extent = vc.getSwapChainExtent();
    ImGui::Text("%.1f FPS (%.2f ms)", io.Framerate, 1000.0f / std::max(io.Framerate, 0.001f));
    ImGui::Text("Resolution: %u x %u", extent.width, extent.height);
    if (appliedPlan_) {
        for (int id : appliedPlan_->nodes) {
            const Node& node = *appliedGraph_.findNode(id);
            if (node.kind == NodeKind::GaussianSplatting) {
                ImGui::Text("Backend: %s", std::string(backendName(node.as<SplattingParams>().backend)).c_str());
            } else if (node.kind == NodeKind::UploadGaussians) {
                if (auto model = uploadedGaussians(appliedGraph_, id)) {
                    ImGui::Text("%s: %u Gaussians", node.title.c_str(), model->count());
                }
            }
        }
        ImGui::Text("Compiled elements: %zu", compiled_.nodes.size());
        if (!liveHostSteps_.empty() && ImGui::TreeNode("CPU steps of the last live build")) {
            for (const auto& step : liveHostSteps_) {
                ImGui::BulletText("%s", step.c_str());
            }
            ImGui::TreePop();
        }
    } else {
        ImGui::TextDisabled("No graph compiled");
    }

    ImGui::SeparatorText("Live");
    const bool pending = plan_.ok() && (!appliedPlan_ || plan_.live->needsRebuildFrom(*appliedPlan_));
    const bool hasPresent = std::any_of(graph_.nodes().begin(), graph_.nodes().end(),
                                        [](const Node& n) { return n.kind == NodeKind::Present; });
    if (!hasPresent) {
        ImGui::TextDisabled("No Present node: nothing renders live.");
    } else if (!plan_.ok()) {
        ImGui::TextColored(severityColor(Severity::Error), "Graph has errors");
        if (appliedPlan_) {
            ImGui::TextDisabled("Showing the last graph that compiled.");
        }
    } else if (!applyError_.empty() && failedPlan_ && !plan_.live->needsRebuildFrom(*failedPlan_)) {
        ImGui::TextColored(severityColor(Severity::Error), "Compile failed");
        ImGui::TextWrapped("%s", applyError_.c_str());
    } else if (pending) {
        ImGui::TextColored(severityColor(Severity::Warning), "Changes not applied");
    } else {
        ImGui::TextColored(ImVec4(0.55f, 0.85f, 0.55f, 1.0f), "Up to date");
    }
    if (hasPresent) {
        ImGui::Checkbox("Auto-apply", &autoApply_);
        ImGui::SameLine();
        ImGui::BeginDisabled(!plan_.ok());
        if (ImGui::Button("Apply")) {
            applyRequested_ = true;
            failedPlan_.reset();
        }
        ImGui::EndDisabled();
    }

    drawRunControls();

    ImGui::SeparatorText("GPU profiling");
    if (ImGui::Checkbox("Per-element timings", &profiling_)) {
        // Takes effect with freshly compiled graphs.
        if (appliedPlan_) {
            const LivePlan plan = *appliedPlan_;
            appliedPlan_.reset();
            apply(plan, appliedGraph_, appliedTopNode_);
        }
    }
    if (profiling_) {
        ImGui::TextDisabled("Frames are synchronous while profiling.");
        std::vector<std::pair<std::string, float>> sorted(timings_.begin(), timings_.end());
        std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
        float total = 0.0f;
        for (const auto& entry : sorted) {
            total += entry.second;
        }
        ImGui::Text("GPU total: %.3f ms", total);
        if (ImGui::BeginTable("timings", 2, ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingStretchProp,
                              ImVec2(0.0f, 220.0f))) {
            ImGui::TableSetupColumn("Element");
            ImGui::TableSetupColumn("ms");
            for (const auto& [label, ms] : sorted) {
                if (ms <= 0.0f) {
                    continue;
                }
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(label.c_str());
                ImGui::TableNextColumn();
                const float fraction = total > 0.0f ? ms / total : 0.0f;
                ImGui::ProgressBar(fraction, ImVec2(-FLT_MIN, 0.0f), std::format("{:.3f}", ms).c_str());
            }
            ImGui::EndTable();
        }
    }

    ImGui::Separator();
    ImGui::TextDisabled("Scene: left drag orbits, wheel zooms");
    ImGui::End();
}

void StudioApp::drawRunControls() {
    ImGui::SeparatorText("Run");
    const bool hasSinks =
        std::any_of(graph_.nodes().begin(), graph_.nodes().end(), [](const Node& n) { return isSink(n.kind); });
    if (!hasSinks) {
        ImGui::TextDisabled("Add a Preview or Image File Writer to run the graph.");
        return;
    }
    ImGui::BeginDisabled(!plan_.run.has_value());
    if (ImGui::Button("Run (F5)")) {
        requestRun();
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::Checkbox("On every change", &autoRun_);
    if (!plan_.run) {
        ImGui::TextColored(severityColor(Severity::Error), "The nodes feeding the outputs have errors.");
    }
    if (!runError_.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, severityColor(Severity::Error));
        ImGui::TextWrapped("Last run failed: %s", runError_.c_str());
        ImGui::PopStyleColor();
    } else if (lastRun_) {
        ImGui::Text("Last run: %.0f ms, %zu elements", lastRun_->milliseconds, lastRun_->compiled.nodes.size());
        if (runIsOutdated()) {
            ImGui::SameLine();
            ImGui::TextColored(severityColor(Severity::Warning), "(outdated)");
        }
        for (const auto& file : lastRun_->written) {
            ImGui::TextDisabled("wrote %s", file.string().c_str());
        }
        if (!runHostSteps_.empty() && ImGui::TreeNode("CPU steps of the last run")) {
            for (const auto& step : runHostSteps_) {
                ImGui::BulletText("%s", step.c_str());
            }
            ImGui::TreePop();
        }
    }
}

bool StudioApp::drawPreview(int node, float width) {
    auto it = previews_.find(node);
    if (it == previews_.end()) {
        return false;
    }
    const PreviewTexture& texture = *it->second;
    const float height = width * static_cast<float>(texture.height()) / static_cast<float>(texture.width());
    ImGui::Image(ImTextureRef(texture.id()), ImVec2(width, height));
    return true;
}

void StudioApp::drawGraphWindow() {
    const ImGuiIO& io = ImGui::GetIO();
    const float height = std::max(260.0f, io.DisplaySize.y * 0.44f);
    ImGui::SetNextWindowPos(ImVec2(0.0f, io.DisplaySize.y - height), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(io.DisplaySize.x - 360.0f, height), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Graph", nullptr, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse)) {
        graphWindowHovered_ = false;
        ImGui::End();
        return;
    }
    graphWindowHovered_ = ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows);
    if (ImGui::BeginTabBar("graph_tabs")) {
        const ImGuiTabItemFlags authoringFlags = requestedTab_ == 0 ? ImGuiTabItemFlags_SetSelected : 0;
        const ImGuiTabItemFlags compiledFlags = requestedTab_ == 1 ? ImGuiTabItemFlags_SetSelected : 0;
        requestedTab_ = -1;
        if (ImGui::BeginTabItem("Authoring graph", nullptr, authoringFlags)) {
            activeTab_ = 0;
            drawAuthoringEditor();
            ImGui::EndTabItem();
        }
        const std::string compiledLabel =
            std::format("Compiled graph ({})###compiled", shownCompiled().nodes.size());
        if (ImGui::BeginTabItem(compiledLabel.c_str(), nullptr, compiledFlags)) {
            activeTab_ = 1;
            drawCompiledEditor();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
    ImGui::End();
}

void StudioApp::drawEditorToolbar(bool compiled) {
    ImGui::BeginDisabled(!plan_.run.has_value());
    if (ImGui::SmallButton("Run")) {
        requestRun();
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::SmallButton("Fit")) {
        fitRequested_ = true;
    }
    ImGui::SameLine();
    if (ImGui::SmallButton("Arrange")) {
        if (compiled) {
            compiledLayoutDirty_ = true;
        } else {
            layoutAuthoringGraph();
            fitRequested_ = true;
        }
    }
    if (compiled) {
        ImGui::SameLine();
        if (ImGui::Checkbox("Hide buffers", &hideBuffers_)) {
            compiledLayoutDirty_ = true;
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Live", compiledSource_ == 0)) {
            showCompiled(0);
        }
        ImGui::SameLine();
        ImGui::BeginDisabled(!lastRun_);
        if (ImGui::RadioButton("Last run", compiledSource_ == 1)) {
            showCompiled(1);
        }
        ImGui::EndDisabled();
    }
    ImGui::SameLine();
    if (compiled) {
        ImGui::TextDisabled("Read-only: the elements klartraum compiled. Scroll: zoom  Right-drag: pan  F: fit");
    } else {
        ImGui::TextDisabled("Scroll: zoom  Right-drag: pan  F: fit  Right-click: add node  Drag pins: connect  "
                            "Del/Backspace: delete");
    }
}

void StudioApp::drawAuthoringEditor() {
    // Plans, diagnostics and results refer to the document graph; inside a
    // definition they are not shown.
    const bool document = !editingDefinition();
    std::set<int> errorNodes;
    std::set<int> unusedNodes;
    for (const auto& d : plan_.diagnostics) {
        if (d.node < 0 || !document) {
            continue;
        }
        if (d.severity == Severity::Error) {
            errorNodes.insert(d.node);
        } else if (d.severity == Severity::Info) {
            unusedNodes.insert(d.node);
        }
    }

    // Nodes that become elements of the klartraum compute graphs; a meta
    // node when any of its inner nodes does.
    std::set<int> liveGraphNodes;
    std::set<int> runGraphNodes;
    auto collect = [&](const std::vector<int>& flatNodes, std::set<int>& into) {
        for (int id : flatNodes) {
            const Node* node = plan_.flat.graph.findNode(id);
            if (node && kindInfo(node->kind).implementation == Implementation::ComputeGraph) {
                into.insert(plan_.flat.top(id));
            }
        }
    };
    if (plan_.live && document) {
        collect(plan_.live->nodes, liveGraphNodes);
    }
    if (plan_.run && document) {
        collect(plan_.run->nodes, runGraphNodes);
    }

    drawBreadcrumbs();
    drawEditorToolbar(false);
    drawLegend();
    ed::SetCurrentEditor(authoringEditor_);
    ed::Begin("authoring");

    // Backspace is the delete key on Mac keyboards; the editor only handles Delete.
    if (graphWindowHovered_ && !ImGui::GetIO().WantTextInput && ImGui::IsKeyPressed(ImGuiKey_Backspace)) {
        deleteSelection();
    }

    auto& vc = engine_.getVulkanContext();
    const ImVec4 bodyText(0.75f, 0.75f, 0.8f, 1.0f);
    for (auto& node : view().nodes()) {
        const ed::NodeId nodeId = authoringNodeId(node.id);
        if (auto it = std::find(nodesToPlace_.begin(), nodesToPlace_.end(), node.id); it != nodesToPlace_.end()) {
            ed::SetNodePosition(nodeId, ImVec2(node.position.x, node.position.y));
            nodesToPlace_.erase(it);
        }

        const bool hasError = errorNodes.contains(node.id);
        const bool unused = unusedNodes.contains(node.id);
        if (hasError) {
            ed::PushStyleColor(ed::StyleColor_NodeBorder, ImVec4(0.95f, 0.35f, 0.3f, 1.0f));
        }
        if (unused) {
            ImGui::PushStyleVar(ImGuiStyleVar_Alpha, 0.55f);
        }

        // Wide enough for the longest pair of pin names on one row.
        const auto inputs = view().inputPins(node);
        const auto outputs = view().outputPins(node);
        float nodeWidth = kNodeWidth;
        {
            const float iconSize = ImGui::GetTextLineHeight();
            const float spacing = ImGui::GetStyle().ItemSpacing.x;
            for (size_t row = 0; row < std::max(inputs.size(), outputs.size()); ++row) {
                float needed = 0.0f;
                if (row < inputs.size()) {
                    needed += iconSize + spacing + ImGui::CalcTextSize(inputs[row].name.c_str()).x;
                }
                if (row < outputs.size()) {
                    needed += 3.0f * spacing + iconSize + spacing + ImGui::CalcTextSize(outputs[row].name.c_str()).x;
                }
                nodeWidth = std::max(nodeWidth, needed);
            }
        }

        ed::BeginNode(nodeId);
        ImGui::PushID(node.id);
        const float left = ImGui::GetCursorScreenPos().x;

        ImGui::BeginGroup();
        ImGui::TextUnformatted(node.title.c_str());
        ImGui::Dummy(ImVec2(nodeWidth, 0.0f));
        ImGui::EndGroup();
        const ImVec2 headerMin = ImGui::GetItemRectMin();
        const ImVec2 headerMax = ImGui::GetItemRectMax();
        ImGui::Dummy(ImVec2(0.0f, 2.0f));
        executionBadge(kindInfo(node.kind));

        // A short summary; the parameters are edited in the inspector.
        ImGui::PushStyleColor(ImGuiCol_Text, bodyText);
        switch (node.kind) {
        case NodeKind::Scene: {
            const auto& path = node.as<SceneParams>().path;
            ImGui::TextUnformatted(path.empty() ? "(no file)" : std::filesystem::path(path).filename().string().c_str());
            if (auto count = sceneCount(node.as<SceneParams>())) {
                ImGui::Text("%zu Gaussians", *count);
            }
            break;
        }
        case NodeKind::TransformGaussians: {
            const auto& p = node.as<TransformGaussiansParams>();
            ImGui::Text("move %.2f %.2f %.2f", p.translation[0], p.translation[1], p.translation[2]);
            ImGui::Text("turn %.0f %.0f %.0f", p.rotation[0], p.rotation[1], p.rotation[2]);
            ImGui::Text("scale %.2f", p.scale);
            break;
        }
        case NodeKind::MergeGaussians:
        case NodeKind::TransformGaussiansGpu:
        case NodeKind::MergeGaussiansGpu:
            break;
        case NodeKind::Number:
        case NodeKind::Time:
        case NodeKind::Sine:
        case NodeKind::UploadNumber:
            try {
                const int source = node.kind == NodeKind::UploadNumber
                                       ? (view().inputLink(node.id, 0) ? view().inputLink(node.id, 0)->fromNode : -1)
                                       : node.id;
                if (source >= 0) {
                    ImGui::Text("= %.3f", evaluateNumber(view(), source, secondsSinceStart()));
                }
            } catch (const std::exception&) {
                ImGui::TextDisabled("(input missing)");
            }
            break;
        case NodeKind::MakeTransform: {
            const auto& p = node.as<MakeTransformParams>();
            ImGui::Text("move %.2f %.2f %.2f", p.translation[0], p.translation[1], p.translation[2]);
            ImGui::Text("turn %.0f %.0f %.0f", p.rotation[0], p.rotation[1], p.rotation[2]);
            ImGui::Text("scale %.2f", p.scale);
            break;
        }
        case NodeKind::UploadGaussians:
            if (auto model = uploadedGaussians(view(), node.id)) {
                ImGui::Text("%u Gaussians, %.0f MB", model->count(), gaussianBytes(model->count()) / 1e6);
            }
            break;
        case NodeKind::Camera: {
            const auto& p = node.as<CameraParams>();
            ImGui::Text("az %.2f  el %.2f", p.azimuth, p.elevation);
            ImGui::Text("dist %.2f  up %s", p.distance, p.up == UpAxis::Y ? "Y" : "Z");
            break;
        }
        case NodeKind::SwapchainTarget: {
            const VkExtent2D extent = vc.getSwapChainExtent();
            ImGui::Text("%u x %u, %u images", extent.width, extent.height, vc.getNumberOfSwapChainImages());
            if (!node.as<SwapchainTargetParams>().clear) {
                ImGui::TextDisabled("not cleared");
            }
            break;
        }
        case NodeKind::Composite: {
            const auto& p = node.as<CompositeParams>();
            ImGui::Text("%s, %s", std::string(compositeModeName(p.mode)).c_str(),
                        std::string(compositeFitName(p.fit)).c_str());
            break;
        }
        case NodeKind::DrawBasics:
            ImGui::TextUnformatted(std::string(drawBasicsShapeName(node.as<DrawBasicsParams>().shape)).c_str());
            break;
        case NodeKind::ClearImage: {
            const auto& c = node.as<ClearImageParams>().color;
            ImGui::ColorButton("##color", ImVec4(c[0], c[1], c[2], c[3]), ImGuiColorEditFlags_NoTooltip);
            ImGui::SameLine();
            ImGui::Text("%.2f %.2f %.2f %.2f", c[0], c[1], c[2], c[3]);
            break;
        }
        case NodeKind::GaussianSplatting: {
            const auto& p = node.as<SplattingParams>();
            ImGui::Text("backend: %s", std::string(backendName(p.backend)).c_str());
            if (appliedPlan_ && appliedPlan_->contains(node.id)) {
                ImGui::Text("%zu elements", compiled_.nodes.size());
            }
            break;
        }
        case NodeKind::Present:
            ImGui::Text("%.0f FPS", ImGui::GetIO().Framerate);
            break;
        case NodeKind::OffscreenTarget: {
            const auto& p = node.as<OffscreenTargetParams>();
            ImGui::Text("%u x %u%s", p.width, p.height, p.clear ? "" : ", not cleared");
            break;
        }
        case NodeKind::ImageFile: {
            const auto& p = node.as<ImageFileParams>();
            ImGui::TextUnformatted(fileName(p.path).c_str());
            ImGui::Text("resized to %u x %u", p.width, p.height);
            break;
        }
        case NodeKind::OnnxModel:
            ImGui::TextUnformatted(fileName(node.as<OnnxModelParams>().path).c_str());
            break;
        case NodeKind::Add:
        case NodeKind::Subtract:
        case NodeKind::Multiply:
        case NodeKind::Divide:
            if (!view().inputLink(node.id, 1)) {
                ImGui::Text("b = %g", node.as<BinaryLayerParams>().b);
            }
            break;
        case NodeKind::Relu:
        case NodeKind::Sigmoid:
        case NodeKind::Sqrt:
        case NodeKind::Softmax:
            break;
        case NodeKind::Prompt: {
            const auto& prompt = node.as<PromptParams>().prompt;
            ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + kNodeWidth);
            ImGui::TextUnformatted(prompt.empty() ? "(empty prompt)"
                                                  : (prompt.size() > 60 ? prompt.substr(0, 57) + "..." : prompt).c_str());
            ImGui::PopTextWrapPos();
            break;
        }
        case NodeKind::LatentNoise: {
            const auto& p = node.as<LatentNoiseParams>();
            ImGui::Text("for %u x %u", p.width, p.height);
            if (p.path.empty()) {
                ImGui::Text("seed %u", p.seed);
            } else {
                ImGui::TextUnformatted(fileName(p.path).c_str());
            }
            break;
        }
        case NodeKind::DdimSampler: {
            const auto& p = node.as<DdimSamplerParams>();
            ImGui::TextUnformatted(fileName(p.path).c_str());
            ImGui::Text("%u steps, guidance %.1f", p.steps, p.guidanceScale);
            break;
        }
        case NodeKind::Meta: {
            const auto& p = node.as<MetaParams>();
            const MetaDefinition* definition = view().findDefinition(p.definition);
            if (!definition) {
                ImGui::TextDisabled("(unknown definition)");
                break;
            }
            if (definition->title != node.title) {
                ImGui::TextDisabled("%s", definition->title.c_str());
            }
            // Exposed values set on this node; files by their name.
            for (const auto& [name, value] : p.values) {
                const auto parsed = nlohmann::json::parse(value, nullptr, false);
                const std::string text = parsed.is_string() ? fileName(parsed.get<std::string>()) : value;
                ImGui::Text("%s: %s", name.c_str(), text.c_str());
            }
            break;
        }
        case NodeKind::ImageToTensor:
        case NodeKind::TensorToImage:
            break;
        case NodeKind::Resample: {
            const auto& p = node.as<ResampleParams>();
            ImGui::Text("%u x %u, %s", p.width, p.height, std::string(filterName(p.filter)).c_str());
            break;
        }
        case NodeKind::Preview:
            if (!drawPreview(node.id, kNodeWidth)) {
                ImGui::TextDisabled(plan_.run ? "Press Run (F5)" : "No result");
            } else if (runIsOutdated()) {
                ImGui::TextColored(severityColor(Severity::Warning), "outdated");
            }
            break;
        case NodeKind::ImageFileWriter: {
            ImGui::TextUnformatted(fileName(node.as<ImageFileWriterParams>().path).c_str());
            if (lastRun_ && lastRun_->images.contains(node.id)) {
                ImGui::TextDisabled("written by the last run");
            }
            break;
        }
        }
        // Tensor outputs show their shape and, unless float32, element type.
        if (plan_.run && document) {
            for (int slot = 0; slot < static_cast<int>(outputs.size()); ++slot) {
                const OutputPin pin = plan_.flat.flatOutput({node.id, slot});
                if (auto it = plan_.run->types.find(pin); it != plan_.run->types.end()) {
                    const std::string type = tensorTypeToString(it->second);
                    if (outputs.size() > 1) {
                        ImGui::Text("%s -> %s", outputs[slot].name.c_str(), type.c_str());
                    } else {
                        ImGui::Text("-> %s", type.c_str());
                    }
                }
            }
        }
        ImGui::PopStyleColor();

        // Pin rows: inputs on the left, outputs right-aligned.
        const size_t rows = std::max(inputs.size(), outputs.size());
        const float iconSize = ImGui::GetTextLineHeight();
        for (size_t row = 0; row < rows; ++row) {
            const int slot = static_cast<int>(row);
            bool sameLine = false;
            if (row < inputs.size()) {
                const auto& pin = inputs[row];
                const PinRef ref{node.id, PinDirection::Input, slot};
                ed::BeginPin(authoringPinId(ref), ed::PinKind::Input);
                ed::PinPivotAlignment(ImVec2(0.0f, 0.5f));
                ed::PinPivotSize(ImVec2(0.0f, 0.0f));
                pinIcon(pin.type, view().inputLink(node.id, slot) != nullptr, iconSize);
                ImGui::SameLine();
                ImGui::TextUnformatted(pin.name.data(), pin.name.data() + pin.name.size());
                ed::EndPin();
                sameLine = true;
            }
            if (row < outputs.size()) {
                const auto& pin = outputs[row];
                const PinRef ref{node.id, PinDirection::Output, slot};
                const float width = ImGui::CalcTextSize(pin.name.data(), pin.name.data() + pin.name.size()).x +
                                    ImGui::GetStyle().ItemSpacing.x + iconSize;
                if (sameLine) {
                    ImGui::SameLine();
                }
                ImGui::SetCursorScreenPos(ImVec2(left + nodeWidth - width, ImGui::GetCursorScreenPos().y));
                ed::BeginPin(authoringPinId(ref), ed::PinKind::Output);
                ed::PinPivotAlignment(ImVec2(1.0f, 0.5f));
                ed::PinPivotSize(ImVec2(0.0f, 0.0f));
                ImGui::TextUnformatted(pin.name.data(), pin.name.data() + pin.name.size());
                ImGui::SameLine();
                const bool connected = std::any_of(view().links().begin(), view().links().end(), [&](const Link& l) {
                    return l.fromNode == node.id && l.fromSlot == slot;
                });
                pinIcon(pin.type, connected, iconSize);
                ed::EndPin();
            }
        }

        ImGui::PopID();
        ed::EndNode();
        drawNodeHeader(nodeId, headerMin, headerMax, kindColor(node.kind));
        drawGraphRings(nodeId, liveGraphNodes.contains(node.id), runGraphNodes.contains(node.id));

        if (unused) {
            ImGui::PopStyleVar();
        }
        if (hasError) {
            ed::PopStyleColor();
        }
    }

    for (const auto& link : view().links()) {
        const PinType type = view().inputType(link.toNode, link.toSlot).value_or(PinType::Tensor);
        ed::Link(authoringLinkId(link.id), authoringPinId({link.fromNode, PinDirection::Output, link.fromSlot}),
                 authoringPinId({link.toNode, PinDirection::Input, link.toSlot}),
                 ImGui::ColorConvertU32ToFloat4(pinColor(type)), 2.5f);
    }

    // Dragging from a pin: accept or reject the link while it is drawn.
    if (ed::BeginCreate(ImVec4(1.0f, 1.0f, 1.0f, 1.0f), 2.0f)) {
        ed::PinId start, end;
        if (ed::QueryNewLink(&start, &end) && start && end) {
            const PinRef a = pinFromEditor(start);
            const PinRef b = pinFromEditor(end);
            // A tensor into an image input goes through a Tensor to Image node.
            const bool convert = view().needsTensorToImage(a, b);
            if (auto error = convert ? std::optional<std::string>() : view().checkConnection(a, b)) {
                ed::RejectNewItem(ImVec4(1.0f, 0.35f, 0.3f, 1.0f), 2.0f);
                ed::Suspend();
                ImGui::SetTooltip("%s", error->c_str());
                ed::Resume();
            } else {
                if (convert) {
                    ed::Suspend();
                    ImGui::SetTooltip("Inserts a Tensor to Image node");
                    ed::Resume();
                }
                if (ed::AcceptNewItem(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), 3.0f)) {
                    int inserted = -1;
                    if (auto problem = view().connectConverting(a, b, &inserted)) {
                        setStatus(*problem, true);
                    } else if (inserted >= 0) {
                        nodesToPlace_.push_back(inserted);
                    }
                    modified_ = true;
                }
            }
        }
    }
    ed::EndCreate();

    if (ed::BeginDelete()) {
        ed::LinkId link;
        while (ed::QueryDeletedLink(&link)) {
            if (ed::AcceptDeletedItem()) {
                view().removeLink(linkFromEditor(link));
                modified_ = true;
            }
        }
        ed::NodeId node;
        while (ed::QueryDeletedNode(&node)) {
            if (ed::AcceptDeletedItem()) {
                view().removeNode(static_cast<int>(node.Get()));
                if (selectedNode_ == static_cast<int>(node.Get())) {
                    selectedNode_ = -1;
                }
                modified_ = true;
            }
        }
    }
    ed::EndDelete();

    // Context menus. Mouse positions are in canvas space while the editor is
    // active, so the position for a new node is taken before suspending.
    const ImVec2 canvasMouse = ImGui::GetMousePos();
    ed::Suspend();
    if (const ed::NodeId clicked = ed::GetDoubleClickedNode()) {
        pendingOpen_ = static_cast<int>(clicked.Get());
    }
    ed::NodeId contextNode;
    ed::LinkId contextLink;
    if (ed::ShowNodeContextMenu(&contextNode)) {
        contextNode_ = static_cast<int>(contextNode.Get());
        ImGui::OpenPopup("node_menu");
    } else if (ed::ShowLinkContextMenu(&contextLink)) {
        contextLink_ = linkFromEditor(contextLink);
        ImGui::OpenPopup("link_menu");
    } else if (ed::ShowBackgroundContextMenu()) {
        newNodePosition_ = Vec2{canvasMouse.x, canvasMouse.y};
        ImGui::OpenPopup("add_node");
    }
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8.0f, 8.0f));
    if (ImGui::BeginPopup("add_node")) {
        ImGui::TextDisabled("Add node");
        // Node kinds, and meta node definitions in their group (the graph's
        // own ones in "Graph definitions"), in the kinds' group order. Inside
        // a definition, it and the ones it is opened from are left out.
        struct Entry {
            const NodeKindInfo* kind = nullptr;
            const MetaDefinition* definition = nullptr;
        };
        std::vector<std::pair<std::string, std::vector<Entry>>> groups;
        auto entries = [&](const std::string& name) -> std::vector<Entry>& {
            for (auto& [group, list] : groups) {
                if (group == name) {
                    return list;
                }
            }
            return groups.emplace_back(name, std::vector<Entry>{}).second;
        };
        for (const auto& info : allKinds()) {
            if (info.kind != NodeKind::Meta) {
                entries(std::string(info.group)).push_back({&info, nullptr});
            }
        }
        auto addDefinitions = [&](const MetaDefinitions& definitions, const char* fallbackGroup) {
            for (const auto& [name, definition] : definitions) {
                if (std::find(editPath_.begin(), editPath_.end(), name) == editPath_.end()) {
                    entries(definition->group.empty() ? fallbackGroup : definition->group).push_back({nullptr, definition.get()});
                }
            }
        };
        addDefinitions(builtinDefinitions(), "Meta nodes");
        addDefinitions(graph_.definitions(), "Graph definitions");

        for (const auto& [group, list] : groups) {
            ImGui::SeparatorText(group.c_str());
            for (const auto& entry : list) {
                if (entry.definition) {
                    const MetaDefinition& d = *entry.definition;
                    if (ImGui::MenuItem(d.title.c_str(), "meta")) {
                        const int id = view().addMetaNode(d.name, newNodePosition_);
                        // The built-in Stable Diffusion nodes start with the sample models.
                        if (d.name == kTextEncoderDefinition) {
                            view().findNode(id)->as<MetaParams>().values["Model"] =
                                nlohmann::json(sdSample("sd15_text_encoder.onnx")).dump();
                        } else if (d.name == kVaeDecoderDefinition) {
                            view().findNode(id)->as<MetaParams>().values["Model"] =
                                nlohmann::json(sdSample("sd15_vae_decoder.onnx")).dump();
                        }
                        nodesToPlace_.push_back(id);
                        pendingSelection_ = id;
                        selectedNode_ = id;
                        modified_ = true;
                    }
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("%s\nMeta node%s: %zu inner nodes", d.description.c_str(),
                                          d.builtin ? " (built-in)" : "", d.graph.nodes().size());
                    }
                    continue;
                }
                const NodeKindInfo& info = *entry.kind;
                if (ImGui::MenuItem(std::string(info.title).c_str(), std::string(siteName(info.site)).c_str())) {
                    const int id = view().addNode(info.kind, newNodePosition_);
                    if (info.kind == NodeKind::Scene) {
                        view().findNode(id)->as<SceneParams>().path = defaultScenePath();
                    } else if (info.kind == NodeKind::ImageFile) {
                        view().findNode(id)->as<ImageFileParams>().path = kSampleImage;
                    } else if (info.kind == NodeKind::Prompt) {
                        auto& p = view().findNode(id)->as<PromptParams>();
                        p.prompt = kSamplePrompt;
                        p.negativePrompt = kSampleNegativePrompt;
                        p.vocabulary = sdSample("vocab.json");
                    } else if (info.kind == NodeKind::LatentNoise) {
                        auto& p = view().findNode(id)->as<LatentNoiseParams>();
                        p.width = kSampleSdSize;
                        p.height = kSampleSdSize;
                    } else if (info.kind == NodeKind::DdimSampler) {
                        view().findNode(id)->as<DdimSamplerParams>().path = sdSample("sd15_unet.onnx");
                    }
                    nodesToPlace_.push_back(id);
                    pendingSelection_ = id;
                    selectedNode_ = id;
                    modified_ = true;
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("%s\n%s, %s: %s", std::string(info.description).c_str(),
                                      std::string(siteName(info.site)).c_str(),
                                      std::string(implementationName(info.implementation)).c_str(),
                                      std::string(info.implementedBy).c_str());
                }
            }
        }
        ImGui::EndPopup();
    }
    if (ImGui::BeginPopup("node_menu")) {
        if (Node* node = view().findNode(contextNode_)) {
            ImGui::TextDisabled("%s", node->title.c_str());
            ImGui::Separator();
            if (node->kind == NodeKind::Meta) {
                if (ImGui::MenuItem("Open")) {
                    pendingOpen_ = node->id;
                }
                if (ImGui::MenuItem("Ungroup")) {
                    pendingUngroup_ = node->id;
                }
            }
            if (ImGui::MenuItem("Group selected nodes", "Ctrl+G")) {
                pendingGroup_ = true;
            }
            if (ImGui::MenuItem("Delete")) {
                ed::Resume();
                ed::DeleteNode(authoringNodeId(node->id));
                ed::Suspend();
            }
        }
        ImGui::EndPopup();
    }
    if (ImGui::BeginPopup("link_menu")) {
        if (ImGui::MenuItem("Disconnect")) {
            ed::Resume();
            ed::DeleteLink(authoringLinkId(contextLink_));
            ed::Suspend();
        }
        ImGui::EndPopup();
    }
    ImGui::PopStyleVar();
    ed::Resume();

    // Requested selections are applied once the editor knows the node,
    // i.e. after it has been drawn.
    if (pendingSelection_ >= 0 && std::find(nodesToPlace_.begin(), nodesToPlace_.end(), pendingSelection_) ==
                                      nodesToPlace_.end()) {
        ed::ClearSelection();
        if (view().findNode(pendingSelection_)) {
            ed::SelectNode(authoringNodeId(pendingSelection_));
        }
        pendingSelection_ = -1;
    }

    // Selection drives the inspector.
    std::vector<ed::NodeId> selected(ed::GetSelectedObjectCount());
    selected.resize(ed::GetSelectedNodes(selected.data(), static_cast<int>(selected.size())));
    if (!selected.empty()) {
        const bool keep = std::any_of(selected.begin(), selected.end(), [&](ed::NodeId id) {
            return static_cast<int>(id.Get()) == selectedNode_;
        });
        if (!keep) {
            selectedNode_ = static_cast<int>(selected.front().Get());
        }
    } else if (ed::IsBackgroundClicked()) {
        selectedNode_ = -1;
    }

    // Remember where the user put things, for saving.
    for (auto& node : view().nodes()) {
        if (std::find(nodesToPlace_.begin(), nodesToPlace_.end(), node.id) != nodesToPlace_.end()) {
            continue;
        }
        const ImVec2 pos = ed::GetNodePosition(authoringNodeId(node.id));
        if (pos.x == FLT_MAX) {
            continue;  // not placed yet
        }
        const Vec2 position{pos.x, pos.y};
        if (!(position == node.position)) {
            node.position = position;
            modified_ = true;
        }
    }

    handleFit(authoringFitFrames_);
    ed::End();
    ed::SetCurrentEditor(nullptr);
}

void StudioApp::handleFit(int& pendingFrames) {
    // Node sizes are only known once they have been drawn, so fitting waits
    // a frame after nodes were (re)placed.
    if (fitRequested_) {
        pendingFrames = 2;
        fitRequested_ = false;
    }
    if (pendingFrames > 0 && --pendingFrames == 0) {
        ed::NavigateToContent(0.0f);
    }
}

void StudioApp::drawGraphRings(ed::NodeId node, bool live, bool run) {
    if (!live && !run) {
        return;
    }
    const ImVec2 pos = ed::GetNodePosition(node);
    const ImVec2 size = ed::GetNodeSize(node);
    if (pos.x == FLT_MAX || size.x <= 0.0f) {
        return;
    }
    const float rounding = ed::GetStyle().NodeRounding;
    ImDrawList* drawList = ed::GetNodeBackgroundDrawList(node);
    float pad = 5.0f;
    for (const auto& [member, color] : {std::pair{live, kLiveGraphColor}, std::pair{run, kRunGraphColor}}) {
        if (member) {
            drawList->AddRect(ImVec2(pos.x - pad, pos.y - pad), ImVec2(pos.x + size.x + pad, pos.y + size.y + pad),
                              color, rounding + pad, 0, 2.5f);
            pad += 5.0f;
        }
    }
}

void StudioApp::drawLegend() {
    auto chip = [](ImU32 color, const char* text) {
        ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(color), "%s", text);
        ImGui::SameLine();
    };
    ImGui::TextDisabled("Runs on:");
    ImGui::SameLine();
    for (ExecutionSite site : {ExecutionSite::Gpu, ExecutionSite::Cpu, ExecutionSite::Upload, ExecutionSite::Readback}) {
        chip(siteColor(site), std::string(siteName(site)).c_str());
    }
    ImGui::TextDisabled(" Made of: klartraum graph elements | a klartraum function | studio code.  Ring:");
    ImGui::SameLine();
    chip(kLiveGraphColor, "live graph");
    chip(kRunGraphColor, "run graph");
    ImGui::NewLine();
}

void StudioApp::drawNodeHeader(ed::NodeId node, ImVec2 headerMin, ImVec2 headerMax, ImU32 color) {
    if (!ImGui::IsItemVisible()) {
        return;
    }
    // The header group spans the node's content; widen it to the node's
    // border using the node padding (left, top, right, bottom).
    const ed::Style& style = ed::GetStyle();
    const float border = style.NodeBorderWidth * 0.5f;
    const ImVec2 min(headerMin.x - style.NodePadding.x + border, headerMin.y - style.NodePadding.y + border);
    const ImVec2 max(headerMax.x + style.NodePadding.z - border, headerMax.y + 2.0f);
    ImDrawList* drawList = ed::GetNodeBackgroundDrawList(node);
    drawList->AddRectFilled(min, max, color, style.NodeRounding, ImDrawFlags_RoundCornersTop);
    drawList->AddLine(ImVec2(min.x, max.y), ImVec2(max.x, max.y), brighten(color, 0.2f), 1.0f);
}

void StudioApp::deleteSelection() {
    // Deleting goes through the editor so its BeginDelete handling applies;
    // call between ed::Begin and ed::End of the authoring editor.
    std::vector<ed::LinkId> links(ed::GetSelectedObjectCount());
    links.resize(ed::GetSelectedLinks(links.data(), static_cast<int>(links.size())));
    for (auto link : links) {
        ed::DeleteLink(link);
    }
    std::vector<ed::NodeId> nodes(ed::GetSelectedObjectCount());
    nodes.resize(ed::GetSelectedNodes(nodes.data(), static_cast<int>(nodes.size())));
    for (auto node : nodes) {
        ed::DeleteNode(node);
    }
}

void StudioApp::layoutAuthoringGraph() {
    std::map<int, int> index;
    for (const auto& node : view().nodes()) {
        index.emplace(node.id, static_cast<int>(index.size()));
    }
    std::vector<std::pair<int, int>> edges;
    for (const auto& link : view().links()) {
        edges.emplace_back(index.at(link.fromNode), index.at(link.toNode));
    }
    LayoutOptions options;
    options.columnSpacing = kNodeWidth + 140.0f;
    options.rowSpacing = 140.0f;
    const auto positions = layeredLayout(static_cast<int>(view().nodes().size()), edges, options);
    for (auto& node : view().nodes()) {
        node.position = positions[index.at(node.id)];
        nodesToPlace_.push_back(node.id);
    }
    modified_ = true;
}

void StudioApp::layoutCompiledGraph(bool measured) {
    std::vector<int> visible;
    std::map<int, int> index;
    for (const auto& node : shownCompiled().nodes) {
        if (hideBuffers_ && node.category == ElementCategory::Buffer) {
            continue;
        }
        index.emplace(node.id, static_cast<int>(visible.size()));
        visible.push_back(node.id);
    }
    std::vector<std::pair<int, int>> edges;
    for (const auto& edge : shownCompiled().edges) {
        if (index.contains(edge.from) && index.contains(edge.to)) {
            edges.emplace_back(index.at(edge.from), index.at(edge.to));
        }
    }
    LayoutOptions options;
    options.columnSpacing = kNodeWidth + 120.0f;
    options.rowSpacing = 120.0f;
    if (measured) {
        for (int id : visible) {
            options.heights.push_back(ed::GetNodeSize(compiledNodeId(id)).y);
        }
    }
    const auto positions = layeredLayout(static_cast<int>(visible.size()), edges, options);
    for (size_t i = 0; i < visible.size(); ++i) {
        ed::SetNodePosition(compiledNodeId(visible[i]), ImVec2(positions[i].x, positions[i].y));
    }
}

void StudioApp::drawCompiledEditor() {
    // Without a live graph, show the last run's.
    if (compiledSource_ == 0 && compiled_.empty() && lastRun_) {
        showCompiled(1);
    }
    drawEditorToolbar(true);
    if (shownCompiled().empty()) {
        ImGui::TextDisabled(compiledSource_ == 0 ? "No live graph is compiled." : "Nothing has been run yet.");
        return;
    }

    ed::SetCurrentEditor(compiledEditor_);
    ed::Begin("compiled");
    if (compiledLayoutDirty_) {
        layoutCompiledGraph(false);
        compiledLayoutDirty_ = false;
        compiledLayoutNeedsMeasure_ = true;
    }

    // Elements belong to document nodes; inside a definition, the selected
    // node is an inner one and highlights nothing.
    const int highlightOwner = editingDefinition() ? -1 : selectedNode_;
    float maxMs = 0.0f;
    for (const auto& [label, ms] : shownTimings()) {
        maxMs = std::max(maxMs, ms);
    }
    const float iconSize = ImGui::GetTextLineHeight();

    for (const auto& node : shownCompiled().nodes) {
        if (hideBuffers_ && node.category == ElementCategory::Buffer) {
            continue;
        }
        const ed::NodeId nodeId = compiledNodeId(node.id);
        const bool highlighted = highlightOwner >= 0 && node.owner == highlightOwner;
        if (highlighted) {
            ed::PushStyleColor(ed::StyleColor_NodeBorder, ImVec4(1.0f, 0.84f, 0.35f, 1.0f));
        } else if (node.inserted) {
            ed::PushStyleColor(ed::StyleColor_NodeBorder, ImGui::ColorConvertU32ToFloat4(kStudioAddedColor));
        }

        ed::BeginNode(nodeId);
        ImGui::PushID(node.id);
        const float left = ImGui::GetCursorScreenPos().x;
        ImGui::BeginGroup();
        ImGui::TextUnformatted(node.label().c_str());
        ImGui::Dummy(ImVec2(kNodeWidth, 0.0f));
        ImGui::EndGroup();
        const ImVec2 headerMin = ImGui::GetItemRectMin();
        const ImVec2 headerMax = ImGui::GetItemRectMax();
        ImGui::Dummy(ImVec2(0.0f, 2.0f));

        if (node.inserted) {
            ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(kStudioAddedColor), "added by studio");
        }
        ImGui::TextDisabled("%s", node.type.c_str());
        if (!node.outputs.empty()) {
            // The single output sits at the right of the type row.
            ImGui::SameLine();
            ImGui::SetCursorScreenPos(ImVec2(left + kNodeWidth - iconSize, ImGui::GetCursorScreenPos().y));
            ed::BeginPin(compiledOutputPinId(node.id), ed::PinKind::Output);
            ed::PinPivotAlignment(ImVec2(1.0f, 0.5f));
            ed::PinPivotSize(ImVec2(0.0f, 0.0f));
            elementPinIcon(true, iconSize);
            ed::EndPin();
        }
        if (auto it = shownTimings().find(node.label()); it != shownTimings().end() && it->second > 0.0f) {
            const float t = maxMs > 0.0f ? it->second / maxMs : 0.0f;
            ImGui::TextColored(ImVec4(0.6f + 0.4f * t, 0.9f - 0.5f * t, 0.5f - 0.3f * t, 1.0f), "%.3f ms", it->second);
        }

        // One pin per input slot; hidden producers are listed by name.
        for (const auto& edge : shownCompiled().edges) {
            if (edge.to != node.id) {
                continue;
            }
            const ElementNode* from = shownCompiled().find(edge.from);
            const bool hidden = hideBuffers_ && from && from->category == ElementCategory::Buffer;
            ed::BeginPin(compiledInputPinId(node.id, edge.slot), ed::PinKind::Input);
            ed::PinPivotAlignment(ImVec2(0.0f, 0.5f));
            ed::PinPivotSize(ImVec2(0.0f, 0.0f));
            elementPinIcon(!hidden, iconSize);
            ImGui::SameLine();
            if (hidden) {
                ImGui::TextDisabled("%d: %s", edge.slot, from->label().c_str());
            } else {
                ImGui::TextDisabled("%d", edge.slot);
            }
            ed::EndPin();
        }
        ImGui::PopID();
        ed::EndNode();
        drawNodeHeader(nodeId, headerMin, headerMax, categoryColor(node.category));
        if (highlighted || node.inserted) {
            ed::PopStyleColor();
        }
    }

    for (const auto& edge : shownCompiled().edges) {
        const ElementNode* from = shownCompiled().find(edge.from);
        const ElementNode* to = shownCompiled().find(edge.to);
        if (!from || !to) {
            continue;
        }
        if (hideBuffers_ && (from->category == ElementCategory::Buffer || to->category == ElementCategory::Buffer)) {
            continue;
        }
        const bool highlighted = highlightOwner >= 0 && (from->owner == highlightOwner || to->owner == highlightOwner);
        ed::Link(compiledLinkId(edge.id), compiledOutputPinId(edge.from), compiledInputPinId(edge.to, edge.slot),
                 highlighted ? ImVec4(1.0f, 0.84f, 0.35f, 0.9f) : ImVec4(0.6f, 0.6f, 0.68f, 0.75f),
                 highlighted ? 2.5f : 1.5f);
    }

    // Node sizes are known once the nodes have been drawn.
    if (compiledLayoutNeedsMeasure_ && !compiledLayoutDirty_) {
        layoutCompiledGraph(true);
        compiledLayoutNeedsMeasure_ = false;
        fitRequested_ = true;
    }

    std::vector<ed::NodeId> selected(ed::GetSelectedObjectCount());
    selected.resize(ed::GetSelectedNodes(selected.data(), static_cast<int>(selected.size())));
    selectedElement_ = selected.empty() ? -1 : elementFromEditor(selected.front());

    handleFit(compiledFitFrames_);
    ed::End();
    ed::SetCurrentEditor(nullptr);
}

void StudioApp::drawInspector() {
    const ImGuiIO& io = ImGui::GetIO();
    const float top = ImGui::GetFrameHeight();
    ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x - 352.0f, top + 8.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(344.0f, io.DisplaySize.y - top - 16.0f), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Inspector")) {
        ImGui::End();
        return;
    }

    if (activeTab_ == 1 && selectedElement_ >= 0) {
        if (const ElementNode* element = shownCompiled().find(selectedElement_)) {
            drawElementInspector(*element);
            ImGui::End();
            return;
        }
    }

    if (Node* node = view().findNode(selectedNode_)) {
        drawNodeInspector(*node);
    } else {
        ImGui::TextDisabled("Select a node to edit it.");
        ImGui::SeparatorText("Graph");
        ImGui::Text("%zu nodes, %zu links", view().nodes().size(), view().links().size());
        ImGui::TextWrapped("File: %s", file_.empty() ? "(unsaved)" : file_.string().c_str());
        drawDiagnostics(-2);
    }
    ImGui::End();
}

void StudioApp::drawNodeInspector(Node& node) {
    const auto& info = kindInfo(node.kind);
    ImGui::PushStyleColor(ImGuiCol_Text, ImGui::ColorConvertU32ToFloat4(brighten(kindColor(node.kind), 0.35f)));
    ImGui::TextUnformatted(info.title.data(), info.title.data() + info.title.size());
    ImGui::PopStyleColor();
    ImGui::TextWrapped("%s", std::string(info.description).c_str());
    ImGui::Spacing();
    if (inputText("Title", node.title)) {
        modified_ = true;
    }
    drawExecutionInfo(node);

    // Parameters that need new pipelines touch the graph's revision; the
    // compiler decides whether a rebuild is really needed.
    bool changed = false;
    auto& vc = engine_.getVulkanContext();

    // An ONNX model file with its inputs, outputs and operators. Samples
    // containing a directory are relative to the klartraum sources; the Stable
    // Diffusion ones are file names in the exported model directory.
    auto modelFile = [&](std::string& path, std::initializer_list<const char*> samples) {
        bool edited = false;
        ImGui::SeparatorText("Model");
        edited |= inputText("File", path);
        if (ImGui::BeginCombo("Samples", "choose...")) {
            for (const char* sample : samples) {
                const std::string file = std::string_view(sample).find('/') == std::string_view::npos
                                             ? sdSample(sample)
                                             : std::string(sample);
                if (ImGui::Selectable(fileName(file).c_str())) {
                    path = file;
                    edited = true;
                }
            }
            ImGui::EndCombo();
        }
        std::string error;
        if (const auto info = path.empty() ? nullptr : onnxInfo(path, error)) {
            for (const auto& input : info->inputs) {
                ImGui::BulletText("in  %s: %s", input.name.c_str(), shapeToString(input.shape).c_str());
            }
            for (const auto& output : info->outputs) {
                ImGui::BulletText("out %s: %s", output.name.c_str(), shapeToString(output.shape).c_str());
            }
            std::string ops;
            for (const auto& op : info->opTypes) {
                ops += (ops.empty() ? "" : ", ") + op;
            }
            ImGui::TextWrapped("Operators: %s", ops.c_str());
        } else if (!path.empty()) {
            ImGui::TextColored(severityColor(Severity::Error), "%s", error.c_str());
        }
        return edited;
    };

    switch (node.kind) {
    case NodeKind::Scene: {
        auto& p = node.as<SceneParams>();
        ImGui::SeparatorText("Scene");
        changed |= inputText("File", p.path);
        if (ImGui::BeginCombo("Samples", "choose...")) {
            for (const char* sample : {"3rdparty/spz/samples/racoonfamily.spz", "3rdparty/spz/samples/hornedlizard.spz",
                                       kSampleLantern}) {
                if (ImGui::Selectable(std::filesystem::path(sample).filename().string().c_str())) {
                    p.path = sample;
                    // The lantern is a Nerfstudio export, upside down otherwise.
                    p.flipY = std::string_view(sample) == kSampleLantern;
                    changed = true;
                }
            }
            ImGui::EndCombo();
        }
        changed |= ImGui::Checkbox("Flip Y", &p.flipY);
        ImGui::SameLine();
        ImGui::TextDisabled("(mirror across the Y axis, e.g. for Nerfstudio exports)");
        if (auto resolved = resolveInputPath(p.path)) {
            ImGui::TextDisabled("%s", resolved->string().c_str());
            if (auto count = sceneCount(p)) {
                ImGui::Text("%zu Gaussians loaded", *count);
            }
        } else if (!p.path.empty()) {
            ImGui::TextColored(severityColor(Severity::Error), "File not found");
        }
        break;
    }
    case NodeKind::TransformGaussians: {
        auto& p = node.as<TransformGaussiansParams>();
        ImGui::SeparatorText("Transform");
        changed |= ImGui::DragFloat3("Translation", p.translation.data(), 0.01f);
        changed |= ImGui::DragFloat3("Rotation (deg)", p.rotation.data(), 0.5f, -360.0f, 360.0f, "%.1f");
        changed |= ImGui::DragFloat("Scale", &p.scale, 0.005f, 0.001f, 1000.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
        if (ImGui::Button("Reset")) {
            p = TransformGaussiansParams{};
            changed = true;
        }
        ImGui::TextWrapped("Scales about the origin, rotates about X, then Y, then Z, then moves. Orientations and "
                           "view-dependent colours turn along. Changes rebuild the Gaussians once you let go.");
        break;
    }
    case NodeKind::Number: {
        auto& p = node.as<NumberParams>();
        ImGui::SeparatorText("Number");
        changed |= ImGui::DragFloat("Value", &p.value, 0.01f);
        break;
    }
    case NodeKind::Time: {
        auto& p = node.as<TimeParams>();
        ImGui::SeparatorText("Time");
        changed |= ImGui::DragFloat("Speed", &p.speed, 0.01f);
        ImGui::Text("t = %.2f s", secondsSinceStart() * p.speed);
        ImGui::TextDisabled("Seconds since the studio started, times the speed.");
        break;
    }
    case NodeKind::Sine: {
        auto& p = node.as<SineParams>();
        ImGui::SeparatorText("amplitude * sin(frequency * 2 pi * x + phase) + offset");
        changed |= ImGui::DragFloat("Amplitude", &p.amplitude, 0.1f);
        changed |= ImGui::DragFloat("Frequency", &p.frequency, 0.01f);
        changed |= ImGui::DragFloat("Phase (deg)", &p.phase, 1.0f);
        changed |= ImGui::DragFloat("Offset", &p.offset, 0.1f);
        break;
    }
    case NodeKind::UploadNumber:
        ImGui::SeparatorText("Upload");
        ImGui::TextWrapped("Copies the CPU number into a klartraum::HostFloat buffer before every frame, so "
                           "GPU nodes read the current value without the graph being rebuilt.");
        break;
    case NodeKind::MakeTransform: {
        auto& p = node.as<MakeTransformParams>();
        ImGui::SeparatorText("Transform (GPU)");
        changed |= ImGui::DragFloat3("Translation", p.translation.data(), 0.01f);
        changed |= ImGui::DragFloat3("Pitch Yaw Roll", p.rotation.data(), 0.5f, -360.0f, 360.0f, "%.1f");
        changed |= ImGui::DragFloat("Scale", &p.scale, 0.005f, 0.001f, 1000.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
        std::string connected;
        for (int i = 0; i < 7; ++i) {
            if (view().inputLink(node.id, i)) {
                connected += (connected.empty() ? "" : ", ") + std::string(kindInfo(node.kind).inputs[i].name);
            }
        }
        if (!connected.empty()) {
            ImGui::TextWrapped("Taken from the inputs instead: %s.", connected.c_str());
        }
        ImGui::TextWrapped("Values apply while the graph runs, without rebuilding it. Rotates about X (pitch), "
                           "then Y (yaw), then Z (roll); scales and rotates about the origin, then moves.");
        break;
    }
    case NodeKind::TransformGaussiansGpu:
        ImGui::SeparatorText("Transform (GPU)");
        ImGui::TextWrapped("Writes moved copies of the Gaussians every frame: positions, orientations, sizes and "
                           "view-dependent colour follow the transform.");
        break;
    case NodeKind::MergeGaussiansGpu:
        ImGui::SeparatorText("Merge (GPU)");
        ImGui::TextWrapped("Writes A's and B's Gaussians into one set every frame, so one Gaussian Splatting "
                           "renders and depth-sorts them together.");
        break;
    case NodeKind::MergeGaussians:
        ImGui::SeparatorText("Merge");
        ImGui::TextWrapped("Combines the Gaussians of A and B into one set that a single Gaussian Splatting "
                           "renders, so the two sort correctly against each other. Chain Merge nodes for more.");
        break;
    case NodeKind::Camera: {
        auto& p = node.as<CameraParams>();
        const bool active = appliedPlan_ && appliedPlan_->cameraNode == node.id;
        ImGui::SeparatorText("Orbit camera");
        if (!active) {
            ImGui::TextDisabled("Not the active camera; values apply once it is.");
        }
        bool live = false;
        live |= ImGui::SliderFloat("Azimuth", &p.azimuth, -3.1416f, 3.1416f);
        live |= ImGui::SliderFloat("Elevation", &p.elevation, -1.5f, 1.5f);
        live |= ImGui::SliderFloat("Distance", &p.distance, 0.05f, 10.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
        live |= ImGui::DragFloat3("Target", p.target.data(), 0.01f);
        int up = p.up == UpAxis::Y ? 0 : 1;
        if (ImGui::Combo("Up axis", &up, "Y\0Z\0")) {
            p.up = up == 0 ? UpAxis::Y : UpAxis::Z;
            live = true;
        }
        if (ImGui::Button("Reset view")) {
            const UpAxis keepUp = p.up;
            p = CameraParams{};
            p.up = keepUp;
            live = true;
        }
        if (live) {
            modified_ = true;
            if (active) {
                pushCameraParams(p);
            }
        }
        ImGui::TextDisabled("Applied live, no recompile.");
        break;
    }
    case NodeKind::SwapchainTarget: {
        const VkExtent2D extent = vc.getSwapChainExtent();
        ImGui::SeparatorText("Swapchain");
        ImGui::Text("Extent: %u x %u", extent.width, extent.height);
        ImGui::Text("Images: %u", vc.getNumberOfSwapChainImages());
        ImGui::TextDisabled("Follows the window size.");
        changed |= ImGui::Checkbox("Clear to black", &node.as<SwapchainTargetParams>().clear);
        ImGui::TextDisabled("Renderers draw over what the target holds. Without clearing, each frame draws over "
                            "an earlier one.");
        break;
    }
    case NodeKind::Composite: {
        auto& p = node.as<CompositeParams>();
        ImGui::SeparatorText("Drawing");
        int mode = p.mode == CompositeMode::Over ? 1 : 0;
        if (ImGui::Combo("Mode", &mode, "replace\0over (by the image's alpha)\0")) {
            p.mode = mode == 1 ? CompositeMode::Over : CompositeMode::Replace;
            changed = true;
        }
        int fit = static_cast<int>(p.fit);
        if (ImGui::Combo("Fit", &fit, "stretch\0fit (whole image, aspect kept)\0fill (covers the target, cropped)\0")) {
            p.fit = static_cast<CompositeFit>(fit);
            changed = true;
        }
        ImGui::TextWrapped("Draws the image onto the target, per frame, and passes the target on: renderers after "
                           "it draw over the image. The image is only read, so it may be a Run result.");
        break;
    }
    case NodeKind::DrawBasics: {
        auto& p = node.as<DrawBasicsParams>();
        ImGui::SeparatorText("Drawing");
        int shape = static_cast<int>(p.shape);
        if (ImGui::Combo("Shape", &shape, "triangle\0cube\0axes\0")) {
            p.shape = static_cast<DrawBasicsShape>(shape);
            changed = true;
        }
        ImGui::TextWrapped("Draws the shape with the camera over the image, per frame, in a render pass that keeps "
                           "the image's contents. Only on images of the window's size.");
        break;
    }
    case NodeKind::ClearImage: {
        auto& c = node.as<ClearImageParams>().color;
        ImGui::SeparatorText("Color");
        changed |= ImGui::ColorEdit4("Color", c.data(), ImGuiColorEditFlags_Float);
        ImGui::TextDisabled("Clears the target in place before it is rendered into (klartraum::ClearImage).");
        break;
    }
    case NodeKind::GaussianSplatting: {
        auto& p = node.as<SplattingParams>();
        ImGui::SeparatorText("Backend");
        int backend = p.backend == SplattingBackend::Compute ? 0 : 1;
        if (ImGui::Combo("Backend", &backend, "compute (tile-binned)\0raster (hardware)\0")) {
            p.backend = backend == 0 ? SplattingBackend::Compute : SplattingBackend::Raster;
            changed = true;
        }
        const bool compute = p.backend == SplattingBackend::Compute;

        ImGui::SeparatorText("Compute backend");
        ImGui::BeginDisabled(!compute);
        changed |= ImGui::SliderFloat("Spread", &p.spreadMultiplier, 1.0f, 4.0f, "%.2f sigma");
        changed |= sliderUint("Max mod", p.maxMod, 1, 8);
        changed |= sliderUint("Sort WG cap", p.numSortWGsCap, 16, 1024);
        changed |= comboUint("Tile X", p.splatTileX, {4u, 8u, 16u});
        changed |= comboUint("Tile Y", p.splatTileY, {4u, 8u, 16u});
        ImGui::EndDisabled();

        ImGui::SeparatorText("Raster backend");
        ImGui::BeginDisabled(compute);
        changed |= ImGui::SliderInt("SH degree", &p.shDegree, 0, 3);
        changed |= ImGui::SliderFloat("Alpha cull", &p.alphaCullThreshold, 0.0f, 0.2f, "%.4f",
                                      ImGuiSliderFlags_Logarithmic);
        ImGui::BeginDisabled(!vc.isMeshShaderSupported());
        changed |= ImGui::Checkbox("Mesh shader path", &p.useMeshShader);
        ImGui::EndDisabled();
        if (!vc.isMeshShaderSupported()) {
            ImGui::SameLine();
            ImGui::TextDisabled("(unsupported)");
        }
        ImGui::EndDisabled();

        if (ImGui::Button("Reset parameters")) {
            const SplattingBackend keep = p.backend;
            p = SplattingParams{};
            p.backend = keep;
            changed = true;
        }
        ImGui::TextDisabled("Changes rebuild the pipelines.");

        if (appliedPlan_ && appliedPlan_->contains(node.id) && !compiled_.empty()) {
            ImGui::SeparatorText("Compiled");
            std::map<ElementCategory, int> counts;
            for (const auto& element : compiled_.nodes) {
                if (element.owner == node.id) {
                    ++counts[element.category];
                }
            }
            for (const auto& [category, count] : counts) {
                ImGui::BulletText("%d %s", count, std::string(categoryName(category)).c_str());
            }
            if (ImGui::SmallButton("Show compiled graph")) {
                requestedTab_ = 1;
            }
        }
        break;
    }
    case NodeKind::Present:
        ImGui::SeparatorText("Present");
        ImGui::Text("%.1f FPS", ImGui::GetIO().Framerate);
        break;
    case NodeKind::OffscreenTarget: {
        auto& p = node.as<OffscreenTargetParams>();
        ImGui::SeparatorText("Image size");
        changed |= inputUint("Width", p.width);
        changed |= inputUint("Height", p.height);
        changed |= ImGui::Checkbox("Clear to black", &p.clear);
        ImGui::TextDisabled("Renderers draw over what the target holds.");
        break;
    }
    case NodeKind::ImageFile: {
        auto& p = node.as<ImageFileParams>();
        ImGui::SeparatorText("Image file");
        changed |= inputText("File", p.path);
        if (ImGui::BeginCombo("Samples", "choose...")) {
            if (ImGui::Selectable(fileName(kSampleImage).c_str())) {
                p.path = kSampleImage;
                changed = true;
            }
            ImGui::EndCombo();
        }
        if (auto resolved = resolveInputPath(p.path)) {
            ImGui::TextDisabled("%s", resolved->string().c_str());
        } else if (!p.path.empty()) {
            ImGui::TextColored(severityColor(Severity::Error), "File not found");
        }
        ImGui::SeparatorText("Resize to");
        changed |= inputUint("Width", p.width);
        changed |= inputUint("Height", p.height);
        ImGui::TextDisabled("Output: 1x3x%ux%u tensor, values in [0, 1]", p.height, p.width);
        break;
    }
    case NodeKind::ImageToTensor:
        ImGui::SeparatorText("Conversion");
        ImGui::TextWrapped("Converts the rendered image into a 1x3xHxW tensor (values in [0, 1]) of the image's "
                           "size with klartraum's image_to_tensor shader. Put a Resample in front to get a fixed "
                           "size from the window's swapchain.");
        break;
    case NodeKind::TensorToImage:
        ImGui::SeparatorText("Conversion");
        ImGui::TextWrapped("Converts a 1x3xHxW tensor into an HxW offscreen image with klartraum's tensor_to_image "
                           "shader; values are clamped to [0, 1]. Feed it to Present, a Preview, an Image File "
                           "Writer, a Resample or an Image to Tensor node.");
        break;
    case NodeKind::Resample: {
        auto& p = node.as<ResampleParams>();
        ImGui::SeparatorText("Output size");
        changed |= inputUint("Width", p.width);
        changed |= inputUint("Height", p.height);
        int filter = p.filter == ResampleFilter::Nearest ? 0 : 1;
        if (ImGui::Combo("Filter", &filter, "nearest\0bilinear\0")) {
            p.filter = filter == 0 ? ResampleFilter::Nearest : ResampleFilter::Bilinear;
            changed = true;
        }
        ImGui::TextDisabled("Stretches the image to %u x %u (klartraum::ImageResample).", p.width, p.height);
        break;
    }
    case NodeKind::OnnxModel:
        changed |= modelFile(node.as<OnnxModelParams>().path, {kSampleEncoder, kSampleDecoder});
        break;
    case NodeKind::Add:
    case NodeKind::Subtract:
    case NodeKind::Multiply:
    case NodeKind::Divide:
        ImGui::SeparatorText("Operand");
        if (view().inputLink(node.id, 1)) {
            ImGui::TextDisabled("B is connected; b is not used.");
        } else {
            changed |= ImGui::DragFloat("b", &node.as<BinaryLayerParams>().b, 0.01f, 0.0f, 0.0f, "%.6g");
            ImGui::TextDisabled("B is not connected: the node uses b as a one-element tensor.");
        }
        break;
    case NodeKind::Relu:
    case NodeKind::Sigmoid:
    case NodeKind::Sqrt:
    case NodeKind::Softmax:
        break;
    case NodeKind::Prompt: {
        auto& p = node.as<PromptParams>();
        ImGui::SeparatorText("Prompt");
        changed |= inputTextMultiline("##prompt", p.prompt);
        ImGui::SeparatorText("Negative prompt");
        changed |= inputTextMultiline("##negative", p.negativePrompt);
        ImGui::TextDisabled("Guidance steers away from the negative prompt. At most 75 tokens each.");
        ImGui::SeparatorText("Tokenizer");
        changed |= inputText("Vocabulary", p.vocabulary);
        if (auto resolved = resolveInputPath(p.vocabulary)) {
            ImGui::TextDisabled("%s", resolved->string().c_str());
            if (!std::filesystem::exists(resolved->parent_path() / "merges.txt")) {
                ImGui::TextColored(severityColor(Severity::Error), "merges.txt is missing next to it");
            }
        } else if (!p.vocabulary.empty()) {
            ImGui::TextColored(severityColor(Severity::Error), "File not found");
        }
        ImGui::TextDisabled("CLIP's vocab.json, with merges.txt next to it, as export_denoiser.py writes them.");
        break;
    }
    case NodeKind::LatentNoise: {
        auto& p = node.as<LatentNoiseParams>();
        ImGui::SeparatorText("Image size");
        changed |= inputUint("Width", p.width, 8);
        changed |= inputUint("Height", p.height, 8);
        ImGui::TextDisabled("Output: 1x4x%ux%u latents; must match the size the UNet was exported for.", p.height / 8,
                            p.width / 8);
        ImGui::SeparatorText("Noise");
        changed |= inputUint("Seed", p.seed, 0);
        changed |= inputText("File", p.path);
        ImGui::TextDisabled("Optional: raw float32 latents replacing the seed's noise, e.g. initial_latents_f32.bin.");
        if (!p.path.empty() && !resolveInputPath(p.path)) {
            ImGui::TextColored(severityColor(Severity::Error), "File not found");
        }
        break;
    }
    case NodeKind::DdimSampler: {
        auto& p = node.as<DdimSamplerParams>();
        changed |= modelFile(p.path, {"sd15_unet.onnx"});
        ImGui::SeparatorText("Sampling");
        changed |= inputUint("Steps", p.steps);
        changed |= ImGui::DragFloat("Guidance", &p.guidanceScale, 0.05f, 0.0f, 30.0f, "%.2f");
        ImGui::TextWrapped("The UNet runs once per step on the negative and the positive prompt; the studio blends "
                           "their noise estimates (guidance) and takes the DDIM step on the CPU. Run waits for all "
                           "steps.");
        break;
    }
    case NodeKind::Meta:
        drawMetaInspector(node, changed);
        break;
    case NodeKind::Preview:
        ImGui::SeparatorText("Result");
        if (drawPreview(node.id, ImGui::GetContentRegionAvail().x)) {
            const auto& image = lastRun_->images.at(node.id);
            ImGui::Text("%u x %u", image.width, image.height);
            if (runIsOutdated()) {
                ImGui::SameLine();
                ImGui::TextColored(severityColor(Severity::Warning), "(outdated, press Run)");
            }
        } else {
            ImGui::TextDisabled("Press Run (F5) to compute it.");
        }
        break;
    case NodeKind::ImageFileWriter: {
        auto& p = node.as<ImageFileWriterParams>();
        ImGui::SeparatorText("Output file");
        changed |= inputText("File", p.path);
        if (!p.path.empty()) {
            ImGui::TextDisabled("%s", resolveOutputPath(p.path).string().c_str());
        }
        ImGui::TextDisabled("Written as PNG on every run.");
        break;
    }
    }

    if (changed) {
        view().touch();
        modified_ = true;
    }

    ImGui::SeparatorText("Pins");
    const auto inputs = view().inputPins(node);
    for (int slot = 0; slot < static_cast<int>(inputs.size()); ++slot) {
        const Node* src = view().inputNode(node.id, slot);
        ImGui::BulletText("in  %s <- %s", inputs[slot].name.c_str(), src ? src->title.c_str() : "(not connected)");
    }
    for (const auto& pin : view().outputPins(node)) {
        ImGui::BulletText("out %s (%s)", pin.name.c_str(), std::string(pinTypeName(pin.type)).c_str());
    }

    if (editingDefinition()) {
        drawInterfaceEditor(node);
    } else {
        drawDiagnostics(node.id);
    }
    ImGui::Spacing();
    if (ImGui::Button("Delete node")) {
        view().removeNode(node.id);
        selectedNode_ = -1;
        modified_ = true;
    }
}

void StudioApp::drawElementInspector(const ElementNode& element) {
    ImGui::PushStyleColor(ImGuiCol_Text, ImGui::ColorConvertU32ToFloat4(brighten(categoryColor(element.category), 0.35f)));
    ImGui::TextUnformatted(element.label().c_str());
    ImGui::PopStyleColor();
    ImGui::TextDisabled("Compiled klartraum element (read-only)");
    if (element.inserted) {
        ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(kStudioAddedColor), "Added by the studio");
        ImGui::TextWrapped("No node asks for this element directly; the studio adds it to connect nodes, e.g. a "
                           "layout transition or Present's resample into the swapchain.");
    }
    ImGui::Separator();
    ImGui::Text("Type: %s", element.type.c_str());
    ImGui::Text("Name: %s", element.name.empty() ? "(unnamed)" : element.name.c_str());
    ImGui::Text("Category: %s", std::string(categoryName(element.category)).c_str());
    if (const Node* owner = graph_.findNode(element.owner)) {
        ImGui::Text("Built for: %s", owner->title.c_str());
    }
    if (auto it = shownTimings().find(element.label()); it != shownTimings().end()) {
        ImGui::Text("GPU time: %.3f ms", it->second);
    }

    ImGui::SeparatorText("Inputs");
    for (const auto& edge : shownCompiled().edges) {
        if (edge.to == element.id) {
            const ElementNode* from = shownCompiled().find(edge.from);
            ImGui::BulletText("%d: %s (%s)", edge.slot, from->label().c_str(), from->type.c_str());
        }
    }
    ImGui::SeparatorText("Consumers");
    for (int id : element.outputs) {
        const ElementNode* to = shownCompiled().find(id);
        ImGui::BulletText("%s (%s)", to->label().c_str(), to->type.c_str());
    }
    if (element.outputs.empty()) {
        ImGui::TextDisabled("none (graph root)");
    }
}

void StudioApp::drawDiagnostics(int node) {
    // node == -2: all diagnostics
    bool any = false;
    for (const auto& d : plan_.diagnostics) {
        if (node != -2 && d.node != node) {
            continue;
        }
        if (!any) {
            ImGui::SeparatorText("Diagnostics");
            any = true;
        }
        ImGui::PushStyleColor(ImGuiCol_Text, severityColor(d.severity));
        const Node* subject = graph_.findNode(d.node);
        if (node == -2 && subject) {
            ImGui::TextWrapped("[%s] %s: %s", severityLabel(d.severity), subject->title.c_str(), d.message.c_str());
        } else {
            ImGui::TextWrapped("[%s] %s", severityLabel(d.severity), d.message.c_str());
        }
        ImGui::PopStyleColor();
    }
    if (!applyError_.empty() && node == -2) {
        ImGui::PushStyleColor(ImGuiCol_Text, severityColor(Severity::Error));
        ImGui::TextWrapped("[compile] %s", applyError_.c_str());
        ImGui::PopStyleColor();
    }
}

// ---------------------------------------------------------------------------
// Meta nodes

const MetaDefinition* StudioApp::editedDefinition() const {
    return editPath_.empty() ? nullptr : graph_.findDefinition(editPath_.back());
}

void StudioApp::openMetaNode(int node) {
    const Node* meta = view().findNode(node);
    if (!meta || meta->kind != NodeKind::Meta) {
        return;
    }
    openDefinition(meta->as<MetaParams>().definition, node);
}

void StudioApp::openDefinition(const std::string& name, int from) {
    if (!graph_.findDefinition(name)) {
        setStatus("Unknown meta node definition '" + name + "'", true);
        return;
    }
    commitDefinition();
    editPath_.push_back(name);
    editFrom_.push_back(from);
    loadEditGraph();
}

void StudioApp::closeDefinitions(size_t depth) {
    if (depth >= editPath_.size()) {
        return;
    }
    commitDefinition();
    editPath_.resize(depth);
    editFrom_.resize(depth);
    if (editingDefinition()) {
        loadEditGraph();
        return;
    }
    selectedNode_ = -1;
    nodesToPlace_.clear();
    for (const auto& node : graph_.nodes()) {
        nodesToPlace_.push_back(node.id);
    }
    ed::SetCurrentEditor(authoringEditor_);
    ed::ClearSelection();
    ed::SetCurrentEditor(nullptr);
    fitRequested_ = true;
}

void StudioApp::loadEditGraph() {
    const MetaDefinition* definition = editedDefinition();
    if (!definition) {
        editPath_.clear();
        editFrom_.clear();
        return;
    }
    // The document's definitions resolve nested meta nodes while editing.
    editGraph_ = definition->graph;
    for (const auto& [name, local] : graph_.definitions()) {
        editGraph_.setDefinition(local);
    }
    editInputs_ = definition->inputs;
    editOutputs_ = definition->outputs;
    editParams_ = definition->params;
    editInterfaceChanged_ = false;
    editRevision_ = editGraph_.revision();
    selectedNode_ = -1;
    nodesToPlace_.clear();
    for (const auto& node : editGraph_.nodes()) {
        nodesToPlace_.push_back(node.id);
    }
    ed::SetCurrentEditor(authoringEditor_);
    ed::ClearSelection();
    ed::SetCurrentEditor(nullptr);
    fitRequested_ = true;
}

void StudioApp::commitDefinition() {
    const MetaDefinition* definition = editedDefinition();
    if (!definition || definition->builtin) {
        return;
    }
    const bool structural = editGraph_.revision() != editRevision_ || editInterfaceChanged_;
    bool moved = false;
    for (const auto& node : editGraph_.nodes()) {
        const Node* stored = definition->graph.findNode(node.id);
        moved = moved || !stored || !(stored->position == node.position) || stored->title != node.title;
    }
    if (!structural && !moved) {
        return;
    }
    const MetaDefinition before = *definition;
    auto updated = std::make_shared<MetaDefinition>(*definition);
    updated->graph = editGraph_;
    updated->inputs = editInputs_;
    updated->outputs = editOutputs_;
    updated->params = editParams_;
    // Pins of nested meta nodes resolve while the document's definitions are
    // still in the inner graph.
    pruneInterface(*updated);
    for (const auto& [name, local] : editGraph_.definitions()) {
        updated->graph.removeDefinition(name);
    }
    editInputs_ = updated->inputs;
    editOutputs_ = updated->outputs;
    editParams_ = updated->params;
    graph_.setDefinition(updated, structural);
    if (const int removed = updateMetaLinks(graph_, updated->name, before); removed > 0) {
        setStatus(std::format("{} link(s) to removed pins of '{}' were removed", removed, updated->title), true);
    }
    editRevision_ = editGraph_.revision();
    editInterfaceChanged_ = false;
    modified_ = true;
}

void StudioApp::duplicateEditedDefinition() {
    const MetaDefinition* definition = editedDefinition();
    if (!definition || editPath_.size() != 1) {
        return;
    }
    auto copy = std::make_shared<MetaDefinition>(*definition);
    copy->name = uniqueDefinitionName(graph_, definition->name + "_copy");
    copy->title = definition->title + " (copy)";
    copy->group.clear();
    copy->builtin = false;
    graph_.setDefinition(copy);
    if (Node* node = graph_.findNode(editFrom_[0]); node && node->kind == NodeKind::Meta) {
        node->as<MetaParams>().definition = copy->name;
        graph_.touch();
    }
    editPath_[0] = copy->name;
    editGraph_.setDefinition(copy);
    modified_ = true;
    setStatus("'" + copy->title + "' is a copy in this graph; the node uses it now");
}

std::vector<int> StudioApp::selectedNodes() {
    ed::SetCurrentEditor(authoringEditor_);
    std::vector<ed::NodeId> selected(static_cast<size_t>(ed::GetSelectedObjectCount()));
    selected.resize(ed::GetSelectedNodes(selected.data(), static_cast<int>(selected.size())));
    ed::SetCurrentEditor(nullptr);
    std::vector<int> nodes;
    for (const auto& id : selected) {
        if (view().findNode(static_cast<int>(id.Get()))) {
            nodes.push_back(static_cast<int>(id.Get()));
        }
    }
    return nodes;
}

void StudioApp::groupSelection() {
    const MetaDefinition* edited = editedDefinition();
    if (edited && edited->builtin) {
        setStatus("Built-in definitions cannot be changed; duplicate it first", true);
        return;
    }
    const std::vector<int> nodes = selectedNodes();
    if (nodes.empty()) {
        setStatus("Select the nodes to group first", true);
        return;
    }
    try {
        const std::string name = uniqueDefinitionName(graph_, "group");
        const int meta = groupNodes(view(), nodes, name, "Group");
        if (editingDefinition()) {
            // A group inside a definition is a definition of the document.
            graph_.setDefinition(view().definitions().at(name));
        }
        nodesToPlace_.push_back(meta);
        pendingSelection_ = meta;
        selectedNode_ = meta;
        modified_ = true;
        setStatus(std::format("Grouped {} nodes into a meta node; double-click it to open it", nodes.size()));
    } catch (const std::exception& e) {
        setStatus(std::string("Group failed: ") + e.what(), true);
    }
}

void StudioApp::ungroup(int node) {
    const MetaDefinition* edited = editedDefinition();
    if (edited && edited->builtin) {
        setStatus("Built-in definitions cannot be changed; duplicate it first", true);
        return;
    }
    try {
        const std::vector<int> added = ungroupNode(view(), node);
        nodesToPlace_.insert(nodesToPlace_.end(), added.begin(), added.end());
        if (selectedNode_ == node) {
            selectedNode_ = -1;
        }
        modified_ = true;
    } catch (const std::exception& e) {
        setStatus(std::string("Ungroup failed: ") + e.what(), true);
    }
}

void StudioApp::drawBreadcrumbs() {
    if (!editingDefinition()) {
        return;
    }
    if (ImGui::SmallButton("Graph")) {
        pendingClose_ = 0;
    }
    for (size_t i = 0; i < editPath_.size(); ++i) {
        ImGui::SameLine();
        ImGui::TextDisabled(">");
        ImGui::SameLine();
        const MetaDefinition* definition = graph_.findDefinition(editPath_[i]);
        const std::string title = definition ? definition->title : editPath_[i];
        if (i + 1 < editPath_.size()) {
            if (ImGui::SmallButton((title + "##crumb" + std::to_string(i)).c_str())) {
                pendingClose_ = static_cast<int>(i + 1);
            }
        } else {
            ImGui::TextUnformatted(title.c_str());
        }
    }
    const MetaDefinition* definition = editedDefinition();
    if (definition && definition->builtin) {
        ImGui::TextColored(severityColor(Severity::Warning), "Built-in definition: shown read-only, changes are not kept.");
        if (editPath_.size() == 1) {
            ImGui::SameLine();
            if (ImGui::SmallButton("Duplicate into this graph")) {
                duplicateEditedDefinition();
            }
        }
    } else {
        ImGui::TextDisabled("Changes apply to every node using this definition. Expose pins and parameters in the "
                           "inspector.");
    }
}

void StudioApp::drawInterfaceEditor(const Node& node) {
    const MetaDefinition* definition = editedDefinition();
    if (!definition) {
        return;
    }
    ImGui::SeparatorText("Meta node interface");
    const bool readOnly = definition->builtin;
    if (readOnly) {
        ImGui::BeginDisabled();
    }
    auto unique = [](const auto& list, const std::string& base) {
        std::string name = base;
        for (int i = 2; std::any_of(list.begin(), list.end(), [&](const auto& e) { return e.name == name; }); ++i) {
            name = base + " " + std::to_string(i);
        }
        return name;
    };
    auto pinToggles = [&](std::vector<MetaPin>& exposed, const std::vector<Pin>& pins, PinDirection direction,
                          const char* label) {
        for (int slot = 0; slot < static_cast<int>(pins.size()); ++slot) {
            const PinRef ref{node.id, direction, slot};
            auto owner = std::find_if(exposed.begin(), exposed.end(), [&](const MetaPin& pin) {
                return std::find(pin.targets.begin(), pin.targets.end(), ref) != pin.targets.end();
            });
            bool on = owner != exposed.end();
            const std::string text = std::format("{} {}{}##{}{}", label, pins[slot].name,
                                                 on ? " as '" + owner->name + "'" : std::string(), label, slot);
            if (ImGui::Checkbox(text.c_str(), &on)) {
                if (on) {
                    exposed.push_back({unique(exposed, pins[slot].name), {ref}});
                } else {
                    std::erase(owner->targets, ref);
                    if (owner->targets.empty()) {
                        exposed.erase(owner);
                    }
                }
                editInterfaceChanged_ = true;
            }
        }
    };
    pinToggles(editInputs_, view().inputPins(node), PinDirection::Input, "Expose input");
    pinToggles(editOutputs_, view().outputPins(node), PinDirection::Output, "Expose output");
    for (const auto& key : paramKeys(node.params)) {
        auto owner = std::find_if(editParams_.begin(), editParams_.end(),
                                  [&](const MetaParam& p) { return p.node == node.id && p.key == key; });
        bool on = owner != editParams_.end();
        const std::string text =
            std::format("Expose {}{}##param{}", key, on ? " as '" + owner->name + "'" : std::string(), key);
        if (ImGui::Checkbox(text.c_str(), &on)) {
            if (on) {
                std::string name = key;
                name[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(name[0])));
                editParams_.push_back({unique(editParams_, name), node.id, key});
            } else {
                editParams_.erase(owner);
            }
            editInterfaceChanged_ = true;
        }
    }
    if (readOnly) {
        ImGui::EndDisabled();
    }
}

void StudioApp::drawMetaInspector(Node& node, bool& changed) {
    auto& p = node.as<MetaParams>();
    const MetaDefinition* definition = view().findDefinition(p.definition);
    if (!definition) {
        ImGui::TextColored(severityColor(Severity::Error), "Unknown definition '%s'", p.definition.c_str());
        return;
    }
    ImGui::SeparatorText("Definition");
    ImGui::TextWrapped("%s%s", definition->title.c_str(), definition->builtin ? " (built-in)" : " (in this graph)");
    if (!definition->description.empty()) {
        ImGui::TextWrapped("%s", definition->description.c_str());
    }
    ImGui::TextDisabled("%zu inner nodes", definition->graph.nodes().size());
    if (ImGui::Button("Open")) {
        pendingOpen_ = node.id;
    }
    ImGui::SameLine();
    if (ImGui::Button("Ungroup")) {
        pendingUngroup_ = node.id;
    }

    if (definition->params.empty()) {
        return;
    }
    ImGui::SeparatorText("Parameters");
    for (const auto& param : definition->params) {
        ImGui::PushID(param.name.c_str());
        const Node* inner = definition->graph.findNode(param.node);
        const bool overridden = p.values.contains(param.name);
        const std::string text = overridden ? p.values.at(param.name)
                                            : (inner ? paramValue(inner->params, param.key) : std::nullopt).value_or("null");
        const auto value = nlohmann::json::parse(text, nullptr, false);
        std::optional<nlohmann::json> edited;
        if (value.is_string()) {
            std::string v = value.get<std::string>();
            if (inputText(param.name.c_str(), v)) {
                edited = v;
            }
            if (param.key == "path" && !v.empty()) {
                if (auto resolved = resolveInputPath(v)) {
                    ImGui::TextDisabled("%s", resolved->string().c_str());
                } else {
                    ImGui::TextColored(severityColor(Severity::Error), "File not found");
                }
            }
        } else if (value.is_number_integer()) {
            int v = value.get<int>();
            if (ImGui::InputInt(param.name.c_str(), &v)) {
                edited = v;
            }
        } else if (value.is_number()) {
            float v = value.get<float>();
            if (ImGui::DragFloat(param.name.c_str(), &v, 0.01f, 0.0f, 0.0f, "%.6g")) {
                edited = v;
            }
        } else if (value.is_boolean()) {
            bool v = value.get<bool>();
            if (ImGui::Checkbox(param.name.c_str(), &v)) {
                edited = v;
            }
        } else {
            ImGui::Text("%s: %s", param.name.c_str(), text.c_str());
        }
        if (edited) {
            p.values[param.name] = edited->dump();
            changed = true;
        }
        if (overridden) {
            ImGui::SameLine();
            if (ImGui::SmallButton("Reset")) {
                p.values.erase(param.name);
                changed = true;
            }
        }
        ImGui::PopID();
    }
}

void StudioApp::drawFileDialog() {
    const char* popup = fileDialog_ == FileDialog::Open ? "Open graph" : "Save graph as";
    if (fileDialog_ != FileDialog::None && !ImGui::IsPopupOpen(popup)) {
        fileDialogPath_ = file_.empty() ? std::string("gaussian_splatting.ktgraph.json") : file_.string();
        ImGui::OpenPopup(popup);
    }
    ImGui::SetNextWindowSize(ImVec2(520.0f, 0.0f), ImGuiCond_Appearing);
    if (ImGui::BeginPopupModal(popup, nullptr, ImGuiWindowFlags_NoResize)) {
        ImGui::TextUnformatted("Path (relative to the working directory):");
        ImGui::SetNextItemWidth(-FLT_MIN);
        if (ImGui::IsWindowAppearing()) {
            ImGui::SetKeyboardFocusHere();
        }
        inputText("##path", fileDialogPath_);
        const bool confirmed = ImGui::Button(fileDialog_ == FileDialog::Open ? "Open" : "Save") ||
                               ImGui::IsKeyPressed(ImGuiKey_Enter);
        ImGui::SameLine();
        const bool cancelled = ImGui::Button("Cancel") || ImGui::IsKeyPressed(ImGuiKey_Escape);
        if (confirmed && !fileDialogPath_.empty()) {
            const bool ok = fileDialog_ == FileDialog::Open ? openGraph(fileDialogPath_) : saveGraphTo(fileDialogPath_);
            if (ok) {
                fileDialog_ = FileDialog::None;
                ImGui::CloseCurrentPopup();
            }
        } else if (cancelled) {
            fileDialog_ = FileDialog::None;
            ImGui::CloseCurrentPopup();
        }
        if (!status_.empty() && statusIsError_) {
            ImGui::TextColored(severityColor(Severity::Error), "%s", status_.c_str());
        }
        ImGui::EndPopup();
    }
}

} // namespace kstudio
