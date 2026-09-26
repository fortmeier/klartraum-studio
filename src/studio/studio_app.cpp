#include "studio/studio_app.hpp"

#include <algorithm>
#include <format>
#include <set>

#include <imgui.h>
#include <imgui_node_editor.h>

#include "klartraum/gaussian_data_standard.hpp"
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "klartraum/klartraum_core.hpp"
#include "klartraum/interface_camera_orbit.hpp"

#include "studio/graph_layout.hpp"
#include "studio/graph_serialization.hpp"

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
    case PinType::Gaussians: return rgb(236, 178, 72);
    case PinType::Camera: return rgb(110, 196, 140);
    case PinType::Image: return rgb(96, 160, 240);
    case PinType::Tensor: return rgb(206, 120, 226);
    }
    return rgb(200, 200, 200);
}

ImU32 kindColor(NodeKind kind) {
    switch (kind) {
    case NodeKind::Scene: return rgb(150, 104, 36);
    case NodeKind::Camera: return rgb(46, 120, 76);
    case NodeKind::SwapchainTarget: return rgb(44, 88, 150);
    case NodeKind::GaussianSplatting: return rgb(122, 60, 150);
    case NodeKind::Present: return rgb(70, 70, 82);
    case NodeKind::OffscreenTarget: return rgb(44, 110, 140);
    case NodeKind::ImageFile: return rgb(150, 84, 50);
    case NodeKind::ImageToTensor: return rgb(110, 70, 140);
    case NodeKind::OnnxModel: return rgb(150, 60, 110);
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
    case PinType::Gaussians:
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
constexpr const char* kSampleImage = "data/lantern.jpg";
constexpr const char* kSampleEncoder = "data/onnx/simple_encoder.onnx";
constexpr const char* kSampleDecoder = "data/onnx/simple_decoder.onnx";

std::string defaultScenePath() {
    return kSampleScene;
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
    return std::nullopt;
}

StudioApp::StudioApp(klartraum::KlartraumEngine& engine, StudioOptions options, GLFWwindow* window)
    : engine_(engine), window_(window), options_(std::move(options)) {
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
        installBuilder(std::nullopt, nullptr);
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
    case Example::SplatAutoencoder: {
        const std::string scene = options_.scenePath.empty() ? defaultScenePath() : options_.scenePath;
        setGraph(makeSplatAutoencoderGraph(scene, kSampleEncoder, kSampleDecoder), {});
        break;
    }
    }
    // The examples come with a layout of their own; start the view on it.
    fitRequested_ = true;
}

void StudioApp::setGraph(Graph graph, std::filesystem::path file) {
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

void StudioApp::updatePlan() {
    if (graph_.revision() != plannedRevision_) {
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
            installBuilder(std::nullopt, nullptr);
        }
        return;
    }
    const SplattingPlan& pending = *plan_.splatting;

    const bool rebuild = !appliedPlan_ || pending.needsRebuildFrom(*appliedPlan_);
    if (rebuild) {
        const bool alreadyFailed = failedPlan_ && !pending.needsRebuildFrom(*failedPlan_);
        if (applyRequested_ || (autoApply_ && !alreadyFailed && !userIsEditing)) {
            applyRequested_ = false;
            apply(pending);
        }
        return;
    }

    // Same pipelines; only which authoring nodes they stand for may differ.
    if (appliedPlan_->cameraNode != pending.cameraNode) {
        if (const Node* camera = graph_.findNode(pending.cameraNode)) {
            pushCameraParams(camera->as<CameraParams>());
        }
    }
    const bool ownersChanged = appliedPlan_->sceneNode != pending.sceneNode ||
                               appliedPlan_->cameraNode != pending.cameraNode ||
                               appliedPlan_->targetNode != pending.targetNode ||
                               appliedPlan_->splattingNode != pending.splattingNode;
    if (ownersChanged) {
        for (auto& element : compiled_.nodes) {
            if (element.owner == appliedPlan_->sceneNode) element.owner = pending.sceneNode;
            else if (element.owner == appliedPlan_->cameraNode) element.owner = pending.cameraNode;
            else if (element.owner == appliedPlan_->targetNode) element.owner = pending.targetNode;
            else if (element.owner == appliedPlan_->splattingNode) element.owner = pending.splattingNode;
        }
    }
    appliedPlan_ = pending;
}

std::shared_ptr<klartraum::GaussianDataStandard> StudioApp::loadModel(const std::string& path) {
    const auto resolved = resolveInputPath(path);
    if (!resolved) {
        throw std::runtime_error("scene file not found: " + path);
    }
    const std::string key = resolved->string();
    if (auto it = models_.find(key); it != models_.end()) {
        return it->second;
    }
    auto& vc = engine_.getVulkanContext();
    auto model = std::make_shared<klartraum::GaussianDataStandard>(vc, key);
    if (model->count() == 0) {
        throw std::runtime_error("scene has no Gaussians: " + key);
    }
    models_[key] = model;
    return model;
}

void StudioApp::installBuilder(const std::optional<SplattingPlan>& plan,
                               const std::shared_ptr<klartraum::GaussianDataStandard>& model) {
    if (!plan) {
        compiled_ = {};
        compiledSignature_.clear();
        // Without a live graph the window is only cleared; the builder keeps
        // it resizable.
        engine_.setGraphBuilder([](klartraum::KlartraumEngine& e) { e.add(e.createRenderPass()); });
        return;
    }
    // The engine runs the builder now and again after every swapchain
    // recreation, so it must not throw: errors are reported via builderError_.
    engine_.setGraphBuilder([this, plan = *plan, model](klartraum::KlartraumEngine& e) {
        try {
            BuiltGraph built = buildGraph(e, plan, model);
            compiled_ = introspect(built.root, built.owners, plan.splattingNode);
            builderError_.clear();
        } catch (const std::exception& ex) {
            e.clearComputeGraphs();
            compiled_ = {};
            builderError_ = ex.what();
        }

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

bool StudioApp::apply(const SplattingPlan& plan) {

    std::shared_ptr<klartraum::GaussianDataStandard> model;
    try {
        model = loadModel(plan.scenePath);
    } catch (const std::exception& e) {
        applyError_ = e.what();
        failedPlan_ = plan;
        setStatus("Compile failed: " + applyError_, true);
        return false;
    }

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
    installBuilder(plan, model);

    if (!builderError_.empty()) {
        applyError_ = builderError_;
        failedPlan_ = plan;
        setStatus("Compile failed: " + applyError_, true);
        // Fall back to the last graph that compiled.
        if (appliedPlan_) {
            installBuilder(appliedPlan_, loadModel(appliedPlan_->scenePath));
        } else {
            installBuilder(std::nullopt, nullptr);
        }
        return false;
    }

    appliedPlan_ = plan;
    failedPlan_.reset();
    applyError_.clear();
    if (newCamera) {
        if (const Node* camera = graph_.findNode(plan.cameraNode)) {
            pushCameraParams(camera->as<CameraParams>());
        }
    }
    // Drop models that the running graph no longer uses.
    const auto keep = resolveInputPath(plan.scenePath);
    std::erase_if(models_, [&](const auto& entry) { return !keep || entry.first != keep->string(); });

    setStatus(std::format("Compiled {} elements ({} backend)", compiled_.nodes.size(),
                          backendName(plan.params.backend)));
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
    lastRunRevision_ = graph_.revision();
    if (!plan_.run) {
        runError_ = "Nothing to run: add a Preview or Image File Writer, and fix the errors of the nodes feeding it.";
        setStatus(runError_, true);
        return;
    }

    RunContext context;
    context.resolveInput = [this](const std::string& path) {
        const auto resolved = resolveInputPath(path);
        if (!resolved) {
            throw std::runtime_error("file not found: " + path);
        }
        return *resolved;
    };
    context.resolveOutput = [this](const std::string& path) { return resolveOutputPath(path); };
    context.loadScene = [this](const std::string& path) { return loadModel(path); };
    context.onnxInfo = [this](const std::string& path, std::string& error) { return onnxInfo(path, error); };

    try {
        RunResult result = runGraph(engine_.getVulkanContext(), graph_, *plan_.run, context);
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
        setStatus(message);
        if (compiledSource_ == 1) {
            compiledLayoutDirty_ = true;
        }
    } catch (const std::exception& e) {
        runError_ = e.what();
        setStatus("Run failed: " + runError_, true);
    }
}

void StudioApp::drawGui() {
    syncCamera();
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
        ImGui::Text("Backend: %s", std::string(backendName(appliedPlan_->params.backend)).c_str());
        if (auto resolved = resolveInputPath(appliedPlan_->scenePath)) {
            if (auto it = models_.find(resolved->string()); it != models_.end()) {
                ImGui::Text("Gaussians: %u", it->second->count());
            }
        }
        ImGui::Text("Compiled elements: %zu", compiled_.nodes.size());
    } else {
        ImGui::TextDisabled("No graph compiled");
    }

    ImGui::SeparatorText("Live");
    const bool pending = plan_.ok() && (!appliedPlan_ || plan_.splatting->needsRebuildFrom(*appliedPlan_));
    const bool hasPresent = std::any_of(graph_.nodes().begin(), graph_.nodes().end(),
                                        [](const Node& n) { return n.kind == NodeKind::Present; });
    if (!hasPresent) {
        ImGui::TextDisabled("No Present node: nothing renders live.");
    } else if (!plan_.ok()) {
        ImGui::TextColored(severityColor(Severity::Error), "Graph has errors");
        if (appliedPlan_) {
            ImGui::TextDisabled("Showing the last graph that compiled.");
        }
    } else if (!applyError_.empty() && failedPlan_ && !plan_.splatting->needsRebuildFrom(*failedPlan_)) {
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
            SplattingPlan plan = *appliedPlan_;
            appliedPlan_.reset();
            apply(plan);
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
    std::set<int> errorNodes;
    std::set<int> unusedNodes;
    for (const auto& d : plan_.diagnostics) {
        if (d.node < 0) {
            continue;
        }
        if (d.severity == Severity::Error) {
            errorNodes.insert(d.node);
        } else if (d.severity == Severity::Info) {
            unusedNodes.insert(d.node);
        }
    }

    drawEditorToolbar(false);
    ed::SetCurrentEditor(authoringEditor_);
    ed::Begin("authoring");

    // Backspace is the delete key on Mac keyboards; the editor only handles Delete.
    if (graphWindowHovered_ && !ImGui::GetIO().WantTextInput && ImGui::IsKeyPressed(ImGuiKey_Backspace)) {
        deleteSelection();
    }

    auto& vc = engine_.getVulkanContext();
    const ImVec4 bodyText(0.75f, 0.75f, 0.8f, 1.0f);
    for (auto& node : graph_.nodes()) {
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

        ed::BeginNode(nodeId);
        ImGui::PushID(node.id);
        const float left = ImGui::GetCursorScreenPos().x;

        ImGui::BeginGroup();
        ImGui::TextUnformatted(node.title.c_str());
        ImGui::Dummy(ImVec2(kNodeWidth, 0.0f));
        ImGui::EndGroup();
        const ImVec2 headerMin = ImGui::GetItemRectMin();
        const ImVec2 headerMax = ImGui::GetItemRectMax();
        ImGui::Dummy(ImVec2(0.0f, 2.0f));

        // A short summary; the parameters are edited in the inspector.
        ImGui::PushStyleColor(ImGuiCol_Text, bodyText);
        switch (node.kind) {
        case NodeKind::Scene: {
            const auto& path = node.as<SceneParams>().path;
            ImGui::TextUnformatted(path.empty() ? "(no file)" : std::filesystem::path(path).filename().string().c_str());
            if (auto resolved = resolveInputPath(path)) {
                if (auto it = models_.find(resolved->string()); it != models_.end()) {
                    ImGui::Text("%u Gaussians", it->second->count());
                }
            }
            break;
        }
        case NodeKind::Camera: {
            const auto& p = node.as<CameraParams>();
            ImGui::Text("az %.2f  el %.2f", p.azimuth, p.elevation);
            ImGui::Text("dist %.2f  up %s", p.distance, p.up == UpAxis::Y ? "Y" : "Z");
            break;
        }
        case NodeKind::SwapchainTarget: {
            const VkExtent2D extent = vc.getSwapChainExtent();
            ImGui::Text("%u x %u, %u images", extent.width, extent.height, vc.getNumberOfSwapChainImages());
            break;
        }
        case NodeKind::GaussianSplatting: {
            const auto& p = node.as<SplattingParams>();
            ImGui::Text("backend: %s", std::string(backendName(p.backend)).c_str());
            if (appliedPlan_ && appliedPlan_->splattingNode == node.id) {
                ImGui::Text("%zu elements", compiled_.nodes.size());
            }
            break;
        }
        case NodeKind::Present:
            ImGui::Text("%.0f FPS", ImGui::GetIO().Framerate);
            break;
        case NodeKind::OffscreenTarget: {
            const auto& p = node.as<OffscreenTargetParams>();
            ImGui::Text("%u x %u", p.width, p.height);
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
        case NodeKind::ImageToTensor:
            break;
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
        // Tensor outputs show their shape.
        if (plan_.run) {
            if (auto it = plan_.run->shapes.find(node.id); it != plan_.run->shapes.end()) {
                ImGui::Text("-> %s", shapeToString(it->second).c_str());
            }
        }
        ImGui::PopStyleColor();

        // Pin rows: inputs on the left, outputs right-aligned.
        const auto& info = kindInfo(node.kind);
        const size_t rows = std::max(info.inputs.size(), info.outputs.size());
        const float iconSize = ImGui::GetTextLineHeight();
        for (size_t row = 0; row < rows; ++row) {
            const int slot = static_cast<int>(row);
            bool sameLine = false;
            if (row < info.inputs.size()) {
                const auto& pin = info.inputs[row];
                const PinRef ref{node.id, PinDirection::Input, slot};
                ed::BeginPin(authoringPinId(ref), ed::PinKind::Input);
                ed::PinPivotAlignment(ImVec2(0.0f, 0.5f));
                ed::PinPivotSize(ImVec2(0.0f, 0.0f));
                pinIcon(pin.type, graph_.inputLink(node.id, slot) != nullptr, iconSize);
                ImGui::SameLine();
                ImGui::TextUnformatted(pin.name.data(), pin.name.data() + pin.name.size());
                ed::EndPin();
                sameLine = true;
            }
            if (row < info.outputs.size()) {
                const auto& pin = info.outputs[row];
                const PinRef ref{node.id, PinDirection::Output, slot};
                const float width = ImGui::CalcTextSize(pin.name.data(), pin.name.data() + pin.name.size()).x +
                                    ImGui::GetStyle().ItemSpacing.x + iconSize;
                if (sameLine) {
                    ImGui::SameLine();
                }
                ImGui::SetCursorScreenPos(ImVec2(left + kNodeWidth - width, ImGui::GetCursorScreenPos().y));
                ed::BeginPin(authoringPinId(ref), ed::PinKind::Output);
                ed::PinPivotAlignment(ImVec2(1.0f, 0.5f));
                ed::PinPivotSize(ImVec2(0.0f, 0.0f));
                ImGui::TextUnformatted(pin.name.data(), pin.name.data() + pin.name.size());
                ImGui::SameLine();
                const bool connected = std::any_of(graph_.links().begin(), graph_.links().end(), [&](const Link& l) {
                    return l.fromNode == node.id && l.fromSlot == slot;
                });
                pinIcon(pin.type, connected, iconSize);
                ed::EndPin();
            }
        }

        ImGui::PopID();
        ed::EndNode();
        drawNodeHeader(nodeId, headerMin, headerMax, kindColor(node.kind));

        if (unused) {
            ImGui::PopStyleVar();
        }
        if (hasError) {
            ed::PopStyleColor();
        }
    }

    for (const auto& link : graph_.links()) {
        const Node* from = graph_.findNode(link.fromNode);
        const PinType type = kindInfo(from->kind).outputs[link.fromSlot].type;
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
            if (auto error = graph_.checkConnection(a, b)) {
                ed::RejectNewItem(ImVec4(1.0f, 0.35f, 0.3f, 1.0f), 2.0f);
                ed::Suspend();
                ImGui::SetTooltip("%s", error->c_str());
                ed::Resume();
            } else if (ed::AcceptNewItem(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), 3.0f)) {
                graph_.connect(a, b);
                modified_ = true;
            }
        }
    }
    ed::EndCreate();

    if (ed::BeginDelete()) {
        ed::LinkId link;
        while (ed::QueryDeletedLink(&link)) {
            if (ed::AcceptDeletedItem()) {
                graph_.removeLink(linkFromEditor(link));
                modified_ = true;
            }
        }
        ed::NodeId node;
        while (ed::QueryDeletedNode(&node)) {
            if (ed::AcceptDeletedItem()) {
                graph_.removeNode(static_cast<int>(node.Get()));
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
        std::string_view group;
        for (const auto& info : allKinds()) {
            if (info.group != group) {
                group = info.group;
                ImGui::SeparatorText(std::string(group).c_str());
            }
            if (ImGui::MenuItem(std::string(info.title).c_str())) {
                const int id = graph_.addNode(info.kind, newNodePosition_);
                if (info.kind == NodeKind::Scene) {
                    graph_.findNode(id)->as<SceneParams>().path = defaultScenePath();
                } else if (info.kind == NodeKind::ImageFile) {
                    graph_.findNode(id)->as<ImageFileParams>().path = kSampleImage;
                }
                nodesToPlace_.push_back(id);
                pendingSelection_ = id;
                selectedNode_ = id;
                modified_ = true;
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("%s", std::string(info.description).c_str());
            }
        }
        ImGui::EndPopup();
    }
    if (ImGui::BeginPopup("node_menu")) {
        if (Node* node = graph_.findNode(contextNode_)) {
            ImGui::TextDisabled("%s", node->title.c_str());
            ImGui::Separator();
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
        if (graph_.findNode(pendingSelection_)) {
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
    for (auto& node : graph_.nodes()) {
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
    for (const auto& node : graph_.nodes()) {
        index.emplace(node.id, static_cast<int>(index.size()));
    }
    std::vector<std::pair<int, int>> edges;
    for (const auto& link : graph_.links()) {
        edges.emplace_back(index.at(link.fromNode), index.at(link.toNode));
    }
    LayoutOptions options;
    options.columnSpacing = kNodeWidth + 140.0f;
    options.rowSpacing = 140.0f;
    const auto positions = layeredLayout(static_cast<int>(graph_.nodes().size()), edges, options);
    for (auto& node : graph_.nodes()) {
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

    const int highlightOwner = selectedNode_;
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
        if (highlighted) {
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

    if (Node* node = graph_.findNode(selectedNode_)) {
        drawNodeInspector(*node);
    } else {
        ImGui::TextDisabled("Select a node to edit it.");
        ImGui::SeparatorText("Graph");
        ImGui::Text("%zu nodes, %zu links", graph_.nodes().size(), graph_.links().size());
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

    // Parameters that need new pipelines touch the graph's revision; the
    // compiler decides whether a rebuild is really needed.
    bool changed = false;
    auto& vc = engine_.getVulkanContext();

    switch (node.kind) {
    case NodeKind::Scene: {
        auto& p = node.as<SceneParams>();
        ImGui::SeparatorText("Scene");
        changed |= inputText("File", p.path);
        if (ImGui::BeginCombo("Samples", "choose...")) {
            for (const char* sample : {"3rdparty/spz/samples/racoonfamily.spz", "3rdparty/spz/samples/hornedlizard.spz"}) {
                if (ImGui::Selectable(std::filesystem::path(sample).filename().string().c_str())) {
                    p.path = sample;
                    changed = true;
                }
            }
            ImGui::EndCombo();
        }
        if (auto resolved = resolveInputPath(p.path)) {
            ImGui::TextDisabled("%s", resolved->string().c_str());
            if (auto it = models_.find(resolved->string()); it != models_.end()) {
                ImGui::Text("%u Gaussians loaded", it->second->count());
            }
        } else if (!p.path.empty()) {
            ImGui::TextColored(severityColor(Severity::Error), "File not found");
        }
        break;
    }
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

        if (appliedPlan_ && appliedPlan_->splattingNode == node.id && !compiled_.empty()) {
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
        ImGui::TextWrapped("Converts the rendered offscreen image into a 1x3xHxW tensor (values in [0, 1]) with "
                           "klartraum's image_to_tensor shader.");
        break;
    case NodeKind::OnnxModel: {
        auto& p = node.as<OnnxModelParams>();
        ImGui::SeparatorText("Model");
        changed |= inputText("File", p.path);
        if (ImGui::BeginCombo("Samples", "choose...")) {
            for (const char* sample : {kSampleEncoder, kSampleDecoder}) {
                if (ImGui::Selectable(fileName(sample).c_str())) {
                    p.path = sample;
                    changed = true;
                }
            }
            ImGui::EndCombo();
        }
        std::string error;
        if (const auto info = p.path.empty() ? nullptr : onnxInfo(p.path, error)) {
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
        } else if (!p.path.empty()) {
            ImGui::TextColored(severityColor(Severity::Error), "%s", error.c_str());
        }
        break;
    }
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
        graph_.touch();
        modified_ = true;
    }

    ImGui::SeparatorText("Pins");
    for (int slot = 0; slot < static_cast<int>(info.inputs.size()); ++slot) {
        const Node* src = graph_.inputNode(node.id, slot);
        ImGui::BulletText("in  %s <- %s", std::string(info.inputs[slot].name).c_str(),
                          src ? src->title.c_str() : "(not connected)");
    }
    for (const auto& pin : info.outputs) {
        ImGui::BulletText("out %s (%s)", std::string(pin.name).c_str(), std::string(pinTypeName(pin.type)).c_str());
    }

    drawDiagnostics(node.id);
    ImGui::Spacing();
    if (ImGui::Button("Delete node")) {
        graph_.removeNode(node.id);
        selectedNode_ = -1;
        modified_ = true;
    }
}

void StudioApp::drawElementInspector(const ElementNode& element) {
    ImGui::PushStyleColor(ImGuiCol_Text, ImGui::ColorConvertU32ToFloat4(brighten(categoryColor(element.category), 0.35f)));
    ImGui::TextUnformatted(element.label().c_str());
    ImGui::PopStyleColor();
    ImGui::TextDisabled("Compiled klartraum element (read-only)");
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
