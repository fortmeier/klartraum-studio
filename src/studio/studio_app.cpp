#include "studio/studio_app.hpp"

#include <algorithm>
#include <format>
#include <set>

#include <imgui.h>
#include <imnodes.h>

#include "klartraum/gaussian_data_standard.hpp"
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "klartraum/klartraum_core.hpp"
#include "klartraum/interface_camera_orbit.hpp"

#include "studio/graph_layout.hpp"
#include "studio/graph_serialization.hpp"

namespace kstudio {

namespace {

constexpr double kProfileInterval = 0.5;  // seconds between profiling readbacks

// Attribute ids in the compiled-graph editor: per element, input slots count
// up from the element's base id, the single output uses the last one.
constexpr int kElementAttrStride = 512;
int elementInputAttr(int element, int slot) { return element * kElementAttrStride + slot; }
int elementOutputAttr(int element) { return element * kElementAttrStride + kElementAttrStride - 1; }

ImU32 rgb(int r, int g, int b, int a = 255) { return IM_COL32(r, g, b, a); }

ImU32 pinColor(PinType type) {
    switch (type) {
    case PinType::Gaussians: return rgb(236, 178, 72);
    case PinType::Camera: return rgb(110, 196, 140);
    case PinType::Image: return rgb(96, 160, 240);
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

ImNodesPinShape pinShape(PinType type) {
    switch (type) {
    case PinType::Gaussians: return ImNodesPinShape_QuadFilled;
    case PinType::Camera: return ImNodesPinShape_TriangleFilled;
    case PinType::Image: return ImNodesPinShape_CircleFilled;
    }
    return ImNodesPinShape_CircleFilled;
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

std::string defaultScenePath() {
    return "3rdparty/spz/samples/racoonfamily.spz";
}

} // namespace

StudioApp::StudioApp(klartraum::KlartraumEngine& engine, StudioOptions options, GLFWwindow* window)
    : engine_(engine), window_(window), options_(std::move(options)) {
    profiling_ = options_.profiling;

    imnodes_ = ImNodes::CreateContext();
    authoringEditor_ = ImNodes::EditorContextCreate();
    compiledEditor_ = ImNodes::EditorContextCreate();
    setupStyle();

    camera_ = std::make_shared<klartraum::InterfaceCameraOrbit>(klartraum::InterfaceCameraOrbit::UpDirection::Y);
    engine_.setInterfaceCamera(camera_);
    pushCameraParams(CameraParams{});

    if (!options_.graphFile.empty()) {
        if (!openGraph(options_.graphFile)) {
            newDefaultGraph();
        }
    } else {
        newDefaultGraph();
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
    ImNodes::EditorContextFree(compiledEditor_);
    ImNodes::EditorContextFree(authoringEditor_);
    ImNodes::DestroyContext(imnodes_);
}

// ---------------------------------------------------------------------------
// Graph documents

void StudioApp::newDefaultGraph() {
    const std::string scene = options_.scenePath.empty() ? defaultScenePath() : options_.scenePath;
    setGraph(makeGaussianSplattingGraph(scene, options_.backend), {});
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
    ImNodes::EditorContextSet(authoringEditor_);
    ImNodes::ClearNodeSelection();
    ImNodes::ClearLinkSelection();
    ImNodes::EditorContextResetPanning(ImVec2(0.0f, 0.0f));
    // A new camera node's view replaces the current one.
    if (appliedPlan_) {
        appliedPlan_->cameraNode = -1;
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

std::optional<std::filesystem::path> StudioApp::resolveScenePath(const std::string& path) const {
    namespace fs = std::filesystem;
    if (path.empty()) {
        return std::nullopt;
    }
    // Relative paths are tried against the working directory, the graph
    // file's directory and the klartraum sources (for the bundled samples).
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

// ---------------------------------------------------------------------------
// Compilation

void StudioApp::updatePlan() {
    if (graph_.revision() != plannedRevision_) {
        plan_ = planGraph(graph_);
        plannedRevision_ = graph_.revision();
        if (plan_.ok() && !resolveScenePath(plan_.splatting->scenePath)) {
            plan_.diagnostics.push_back(Diagnostic{Severity::Error, plan_.splatting->sceneNode,
                                                   "Scene file not found: " + plan_.splatting->scenePath});
            plan_.splatting.reset();
        }
    }
    if (!plan_.ok()) {
        return;
    }
    const SplattingPlan& pending = *plan_.splatting;

    const bool rebuild = !appliedPlan_ || pending.needsRebuildFrom(*appliedPlan_);
    if (rebuild) {
        const bool alreadyFailed = failedPlan_ && !pending.needsRebuildFrom(*failedPlan_);
        const bool userIsEditing = ImGui::GetCurrentContext() && ImGui::IsAnyItemActive();
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
    const auto resolved = resolveScenePath(path);
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
        // A builder that adds nothing keeps the window resizable.
        engine_.setGraphBuilder([](klartraum::KlartraumEngine&) {});
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
    const auto keep = resolveScenePath(plan.scenePath);
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

    ImNodes::StyleColorsDark();
    ImNodesStyle& nodes = ImNodes::GetStyle();
    nodes.Flags |= ImNodesStyleFlags_GridLines | ImNodesStyleFlags_GridLinesPrimary;
    nodes.NodeCornerRounding = 5.0f;
    nodes.LinkThickness = 2.5f;
    nodes.PinCircleRadius = 5.0f;
    nodes.Colors[ImNodesCol_GridBackground] = rgb(24, 24, 30, 235);
    nodes.Colors[ImNodesCol_GridLine] = rgb(50, 50, 60, 200);
    nodes.Colors[ImNodesCol_GridLinePrimary] = rgb(64, 64, 76, 220);
    nodes.Colors[ImNodesCol_NodeBackground] = rgb(40, 40, 48, 245);
    nodes.Colors[ImNodesCol_NodeBackgroundHovered] = rgb(48, 48, 58, 245);
    nodes.Colors[ImNodesCol_NodeBackgroundSelected] = rgb(54, 54, 66, 245);

    // Ctrl+click on a link end detaches it.
    ImNodes::GetIO().LinkDetachWithModifierClick.Modifier = &ImGui::GetIO().KeyCtrl;
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

void StudioApp::drawGui() {
    syncCamera();

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
        if (ImGui::MenuItem("New Gaussian Splatting Graph")) {
            newDefaultGraph();
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
        if (auto resolved = resolveScenePath(appliedPlan_->scenePath)) {
            if (auto it = models_.find(resolved->string()); it != models_.end()) {
                ImGui::Text("Gaussians: %u", it->second->count());
            }
        }
        ImGui::Text("Compiled elements: %zu", compiled_.nodes.size());
    } else {
        ImGui::TextDisabled("No graph compiled");
    }

    ImGui::SeparatorText("Compile");
    const bool pending = plan_.ok() && (!appliedPlan_ || plan_.splatting->needsRebuildFrom(*appliedPlan_));
    if (!plan_.ok()) {
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
    ImGui::Checkbox("Auto-apply", &autoApply_);
    ImGui::SameLine();
    ImGui::BeginDisabled(!plan_.ok());
    if (ImGui::Button("Apply")) {
        applyRequested_ = true;
        failedPlan_.reset();
    }
    ImGui::EndDisabled();

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

void StudioApp::drawGraphWindow() {
    const ImGuiIO& io = ImGui::GetIO();
    const float height = std::max(260.0f, io.DisplaySize.y * 0.44f);
    ImGui::SetNextWindowPos(ImVec2(0.0f, io.DisplaySize.y - height), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(io.DisplaySize.x - 360.0f, height), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Graph", nullptr, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse)) {
        ImGui::End();
        return;
    }
    if (ImGui::BeginTabBar("graph_tabs")) {
        const ImGuiTabItemFlags authoringFlags = requestedTab_ == 0 ? ImGuiTabItemFlags_SetSelected : 0;
        const ImGuiTabItemFlags compiledFlags = requestedTab_ == 1 ? ImGuiTabItemFlags_SetSelected : 0;
        requestedTab_ = -1;
        if (ImGui::BeginTabItem("Authoring graph", nullptr, authoringFlags)) {
            activeTab_ = 0;
            drawAuthoringEditor();
            ImGui::EndTabItem();
        }
        const std::string compiledLabel = std::format("Compiled graph ({})###compiled", compiled_.nodes.size());
        if (ImGui::BeginTabItem(compiledLabel.c_str(), nullptr, compiledFlags)) {
            activeTab_ = 1;
            drawCompiledEditor();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
    ImGui::End();
}

void StudioApp::drawAuthoringEditor() {
    std::set<int> errorNodes;
    for (const auto& d : plan_.diagnostics) {
        if (d.severity == Severity::Error && d.node >= 0) {
            errorNodes.insert(d.node);
        }
    }
    std::set<int> unusedNodes;
    for (const auto& d : plan_.diagnostics) {
        if (d.severity == Severity::Info && d.node >= 0) {
            unusedNodes.insert(d.node);
        }
    }

    ImGui::TextDisabled("Right-click: add node   Drag pin to pin: connect   Del: delete   Ctrl+click link end: detach");
    ImNodes::EditorContextSet(authoringEditor_);
    ImNodes::BeginNodeEditor();

    auto& vc = engine_.getVulkanContext();
    for (auto& node : graph_.nodes()) {
        if (auto it = std::find(nodesToPlace_.begin(), nodesToPlace_.end(), node.id); it != nodesToPlace_.end()) {
            if (addNodeScreenPos_) {
                ImNodes::SetNodeScreenSpacePos(node.id, ImVec2(addNodeScreenPos_->x, addNodeScreenPos_->y));
                addNodeScreenPos_.reset();
            } else {
                ImNodes::SetNodeGridSpacePos(node.id, ImVec2(node.position.x, node.position.y));
            }
            nodesToPlace_.erase(it);
        }

        const ImU32 color = kindColor(node.kind);
        ImNodes::PushColorStyle(ImNodesCol_TitleBar, color);
        ImNodes::PushColorStyle(ImNodesCol_TitleBarHovered, brighten(color, 0.08f));
        ImNodes::PushColorStyle(ImNodesCol_TitleBarSelected, brighten(color, 0.15f));
        const bool hasError = errorNodes.contains(node.id);
        if (hasError) {
            ImNodes::PushColorStyle(ImNodesCol_NodeOutline, rgb(240, 90, 80));
        }
        const bool unused = unusedNodes.contains(node.id);
        if (unused) {
            ImGui::PushStyleVar(ImGuiStyleVar_Alpha, 0.55f);
        }

        ImNodes::BeginNode(node.id);
        ImNodes::BeginNodeTitleBar();
        ImGui::TextUnformatted(node.title.c_str());
        ImNodes::EndNodeTitleBar();

        // A short summary; the parameters are edited in the inspector.
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.75f, 0.75f, 0.8f, 1.0f));
        switch (node.kind) {
        case NodeKind::Scene: {
            const auto& path = node.as<SceneParams>().path;
            ImGui::TextUnformatted(path.empty() ? "(no file)" : std::filesystem::path(path).filename().string().c_str());
            if (auto resolved = resolveScenePath(path)) {
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
        }
        ImGui::PopStyleColor();
        ImGui::Dummy(ImVec2(150.0f, 0.0f));

        const auto& info = kindInfo(node.kind);
        for (int slot = 0; slot < static_cast<int>(info.inputs.size()); ++slot) {
            const auto& pin = info.inputs[slot];
            ImNodes::PushColorStyle(ImNodesCol_Pin, pinColor(pin.type));
            ImNodes::BeginInputAttribute(pinId({node.id, PinDirection::Input, slot}), pinShape(pin.type));
            ImGui::TextUnformatted(pin.name.data(), pin.name.data() + pin.name.size());
            ImNodes::EndInputAttribute();
            ImNodes::PopColorStyle();
        }
        for (int slot = 0; slot < static_cast<int>(info.outputs.size()); ++slot) {
            const auto& pin = info.outputs[slot];
            ImNodes::PushColorStyle(ImNodesCol_Pin, pinColor(pin.type));
            ImNodes::BeginOutputAttribute(pinId({node.id, PinDirection::Output, slot}), pinShape(pin.type));
            const float textWidth = ImGui::CalcTextSize(pin.name.data(), pin.name.data() + pin.name.size()).x;
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 150.0f - textWidth);
            ImGui::TextUnformatted(pin.name.data(), pin.name.data() + pin.name.size());
            ImNodes::EndOutputAttribute();
            ImNodes::PopColorStyle();
        }
        ImNodes::EndNode();

        if (unused) {
            ImGui::PopStyleVar();
        }
        if (hasError) {
            ImNodes::PopColorStyle();
        }
        ImNodes::PopColorStyle();
        ImNodes::PopColorStyle();
        ImNodes::PopColorStyle();
    }

    for (const auto& link : graph_.links()) {
        const Node* from = graph_.findNode(link.fromNode);
        const PinType type = kindInfo(from->kind).outputs[link.fromSlot].type;
        ImNodes::PushColorStyle(ImNodesCol_Link, pinColor(type));
        ImNodes::PushColorStyle(ImNodesCol_LinkHovered, brighten(pinColor(type), 0.2f));
        ImNodes::PushColorStyle(ImNodesCol_LinkSelected, brighten(pinColor(type), 0.3f));
        ImNodes::Link(link.id, pinId({link.fromNode, PinDirection::Output, link.fromSlot}),
                      pinId({link.toNode, PinDirection::Input, link.toSlot}));
        ImNodes::PopColorStyle();
        ImNodes::PopColorStyle();
        ImNodes::PopColorStyle();
    }

    // Right-click context menus: on a node, or on the canvas to add one.
    int hoveredNode = -1;
    const bool editorHovered = ImNodes::IsEditorHovered();
    const bool nodeHovered = ImNodes::IsNodeHovered(&hoveredNode);
    if (editorHovered && ImGui::IsMouseReleased(ImGuiMouseButton_Right)) {
        if (nodeHovered) {
            selectedNode_ = hoveredNode;
            ImGui::OpenPopup("node_menu");
        } else {
            addNodeScreenPos_ = Vec2{ImGui::GetMousePos().x, ImGui::GetMousePos().y};
            ImGui::OpenPopup("add_node");
        }
    }
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8.0f, 8.0f));
    if (ImGui::BeginPopup("add_node")) {
        ImGui::TextDisabled("Add node");
        ImGui::Separator();
        for (const auto& info : allKinds()) {
            if (ImGui::MenuItem(std::string(info.title).c_str())) {
                const int id = graph_.addNode(info.kind);
                if (info.kind == NodeKind::Scene) {
                    graph_.findNode(id)->as<SceneParams>().path = defaultScenePath();
                }
                nodesToPlace_.push_back(id);
                modified_ = true;
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("%s", std::string(info.description).c_str());
            }
        }
        ImGui::EndPopup();
    } else if (nodesToPlace_.empty()) {
        addNodeScreenPos_.reset();
    }
    if (ImGui::BeginPopup("node_menu")) {
        if (Node* node = graph_.findNode(selectedNode_)) {
            ImGui::TextDisabled("%s", node->title.c_str());
            ImGui::Separator();
            if (ImGui::MenuItem("Delete")) {
                graph_.removeNode(node->id);
                selectedNode_ = -1;
                modified_ = true;
            }
        }
        ImGui::EndPopup();
    }
    ImGui::PopStyleVar();

    ImNodes::MiniMap(0.15f, ImNodesMiniMapLocation_BottomRight);
    ImNodes::EndNodeEditor();

    // Editor interactions are only reported after EndNodeEditor.
    int startAttr = 0, endAttr = 0;
    if (ImNodes::IsLinkCreated(&startAttr, &endAttr)) {
        if (auto error = graph_.connect(pinFromId(startAttr), pinFromId(endAttr))) {
            setStatus(*error, true);
        } else {
            modified_ = true;
        }
    }
    int destroyedLink = 0;
    if (ImNodes::IsLinkDestroyed(&destroyedLink)) {
        graph_.removeLink(destroyedLink);
        modified_ = true;
    }

    const int selectedCount = ImNodes::NumSelectedNodes();
    if (selectedCount > 0) {
        std::vector<int> selected(selectedCount);
        ImNodes::GetSelectedNodes(selected.data());
        if (std::find(selected.begin(), selected.end(), selectedNode_) == selected.end()) {
            selectedNode_ = selected.front();
        }
    } else if (editorHovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !nodeHovered) {
        selectedNode_ = -1;
    }

    if (editorHovered && !ImGui::GetIO().WantTextInput &&
        (ImGui::IsKeyPressed(ImGuiKey_Delete) || ImGui::IsKeyPressed(ImGuiKey_Backspace))) {
        deleteSelection();
    }

    // Remember where the user put things, for saving.
    for (auto& node : graph_.nodes()) {
        if (std::find(nodesToPlace_.begin(), nodesToPlace_.end(), node.id) == nodesToPlace_.end()) {
            const ImVec2 pos = ImNodes::GetNodeGridSpacePos(node.id);
            const Vec2 position{pos.x, pos.y};
            if (!(position == node.position)) {
                node.position = position;
                modified_ = true;
            }
        }
    }
}

void StudioApp::deleteSelection() {
    ImNodes::EditorContextSet(authoringEditor_);
    const int linkCount = ImNodes::NumSelectedLinks();
    if (linkCount > 0) {
        std::vector<int> links(linkCount);
        ImNodes::GetSelectedLinks(links.data());
        for (int id : links) {
            graph_.removeLink(id);
        }
    }
    const int nodeCount = ImNodes::NumSelectedNodes();
    if (nodeCount > 0) {
        std::vector<int> nodes(nodeCount);
        ImNodes::GetSelectedNodes(nodes.data());
        for (int id : nodes) {
            graph_.removeNode(id);
        }
    }
    if (linkCount + nodeCount > 0) {
        ImNodes::ClearNodeSelection();
        ImNodes::ClearLinkSelection();
        selectedNode_ = -1;
        modified_ = true;
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
    options.columnSpacing = 300.0f;
    options.rowSpacing = 140.0f;
    const auto positions = layeredLayout(static_cast<int>(graph_.nodes().size()), edges, options);
    for (auto& node : graph_.nodes()) {
        node.position = {40.0f + positions[index.at(node.id)].x, 20.0f + positions[index.at(node.id)].y};
        nodesToPlace_.push_back(node.id);
    }
    modified_ = true;
}

void StudioApp::layoutCompiledGraph(bool measured) {
    std::vector<int> visible;
    std::map<int, int> index;
    for (const auto& node : compiled_.nodes) {
        if (hideBuffers_ && node.category == ElementCategory::Buffer) {
            continue;
        }
        index.emplace(node.id, static_cast<int>(visible.size()));
        visible.push_back(node.id);
    }
    std::vector<std::pair<int, int>> edges;
    for (const auto& edge : compiled_.edges) {
        if (index.contains(edge.from) && index.contains(edge.to)) {
            edges.emplace_back(index.at(edge.from), index.at(edge.to));
        }
    }
    LayoutOptions options;
    options.columnSpacing = 280.0f;
    options.rowSpacing = 120.0f;
    if (measured) {
        for (int id : visible) {
            options.heights.push_back(ImNodes::GetNodeDimensions(id).y);
        }
    }
    const auto positions = layeredLayout(static_cast<int>(visible.size()), edges, options);
    for (size_t i = 0; i < visible.size(); ++i) {
        ImNodes::SetNodeGridSpacePos(visible[i], ImVec2(20.0f + positions[i].x, 20.0f + positions[i].y));
    }
}

void StudioApp::drawCompiledEditor() {
    if (compiled_.empty()) {
        ImGui::TextDisabled("Nothing compiled yet.");
        return;
    }
    if (ImGui::Checkbox("Hide buffers", &hideBuffers_)) {
        compiledLayoutDirty_ = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Arrange")) {
        compiledLayoutDirty_ = true;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("Read-only view of the elements klartraum compiled; select one to inspect it.");

    ImNodes::EditorContextSet(compiledEditor_);
    if (compiledLayoutDirty_) {
        layoutCompiledGraph(false);
        ImNodes::EditorContextResetPanning(ImVec2(0.0f, 0.0f));
        compiledLayoutDirty_ = false;
        compiledLayoutNeedsMeasure_ = true;
    }

    ImNodes::BeginNodeEditor();
    const int highlightOwner = selectedNode_;
    float maxMs = 0.0f;
    for (const auto& [label, ms] : timings_) {
        maxMs = std::max(maxMs, ms);
    }

    for (const auto& node : compiled_.nodes) {
        if (hideBuffers_ && node.category == ElementCategory::Buffer) {
            continue;
        }
        const ImU32 color = categoryColor(node.category);
        ImNodes::PushColorStyle(ImNodesCol_TitleBar, color);
        ImNodes::PushColorStyle(ImNodesCol_TitleBarHovered, brighten(color, 0.08f));
        ImNodes::PushColorStyle(ImNodesCol_TitleBarSelected, brighten(color, 0.15f));
        const bool highlighted = highlightOwner >= 0 && node.owner == highlightOwner;
        ImNodes::PushColorStyle(ImNodesCol_NodeOutline, highlighted ? rgb(255, 214, 90) : rgb(90, 90, 100));

        ImNodes::BeginNode(node.id);
        ImNodes::BeginNodeTitleBar();
        ImGui::TextUnformatted(node.label().c_str());
        ImNodes::EndNodeTitleBar();

        ImGui::TextDisabled("%s", node.type.c_str());
        if (auto it = timings_.find(node.label()); it != timings_.end() && it->second > 0.0f) {
            const float t = maxMs > 0.0f ? it->second / maxMs : 0.0f;
            ImGui::TextColored(ImVec4(0.6f + 0.4f * t, 0.9f - 0.5f * t, 0.5f - 0.3f * t, 1.0f), "%.3f ms", it->second);
        }

        // One pin per input slot; hidden producers are listed by name.
        for (const auto& edge : compiled_.edges) {
            if (edge.to != node.id) {
                continue;
            }
            ImNodes::BeginInputAttribute(elementInputAttr(node.id, edge.slot), ImNodesPinShape_Circle);
            const ElementNode* from = compiled_.find(edge.from);
            const bool hidden = hideBuffers_ && from && from->category == ElementCategory::Buffer;
            if (hidden) {
                ImGui::TextDisabled("%d: %s", edge.slot, from->label().c_str());
            } else {
                ImGui::Dummy(ImVec2(1.0f, ImGui::GetTextLineHeight()));
            }
            ImNodes::EndInputAttribute();
        }
        if (!node.outputs.empty()) {
            ImNodes::BeginOutputAttribute(elementOutputAttr(node.id), ImNodesPinShape_CircleFilled);
            ImGui::Dummy(ImVec2(150.0f, 1.0f));
            ImNodes::EndOutputAttribute();
        } else {
            ImGui::Dummy(ImVec2(150.0f, 1.0f));
        }
        ImNodes::EndNode();
        ImNodes::PopColorStyle();
        ImNodes::PopColorStyle();
        ImNodes::PopColorStyle();
        ImNodes::PopColorStyle();
    }

    for (const auto& edge : compiled_.edges) {
        const ElementNode* from = compiled_.find(edge.from);
        const ElementNode* to = compiled_.find(edge.to);
        if (!from || !to) {
            continue;
        }
        if (hideBuffers_ && (from->category == ElementCategory::Buffer || to->category == ElementCategory::Buffer)) {
            continue;
        }
        const bool highlighted = highlightOwner >= 0 && (from->owner == highlightOwner || to->owner == highlightOwner);
        ImNodes::PushColorStyle(ImNodesCol_Link, highlighted ? rgb(255, 214, 90, 220) : rgb(150, 150, 170, 180));
        ImNodes::Link(edge.id, elementOutputAttr(edge.from), elementInputAttr(edge.to, edge.slot));
        ImNodes::PopColorStyle();
    }

    ImNodes::MiniMap(0.18f, ImNodesMiniMapLocation_BottomRight);
    ImNodes::EndNodeEditor();

    // Node sizes are known once the nodes have been drawn.
    if (compiledLayoutNeedsMeasure_) {
        layoutCompiledGraph(true);
        compiledLayoutNeedsMeasure_ = false;
    }

    // The view is read-only: links dragged here are ignored.
    int ignoredStart = 0, ignoredEnd = 0;
    (void)ImNodes::IsLinkCreated(&ignoredStart, &ignoredEnd);

    const int selectedCount = ImNodes::NumSelectedNodes();
    if (selectedCount > 0) {
        std::vector<int> selected(selectedCount);
        ImNodes::GetSelectedNodes(selected.data());
        selectedElement_ = selected.front();
    } else {
        selectedElement_ = -1;
    }
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
        if (const ElementNode* element = compiled_.find(selectedElement_)) {
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
        if (auto resolved = resolveScenePath(p.path)) {
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
    if (auto it = timings_.find(element.label()); it != timings_.end()) {
        ImGui::Text("GPU time: %.3f ms", it->second);
    }

    ImGui::SeparatorText("Inputs");
    for (const auto& edge : compiled_.edges) {
        if (edge.to == element.id) {
            const ElementNode* from = compiled_.find(edge.from);
            ImGui::BulletText("%d: %s (%s)", edge.slot, from->label().c_str(), from->type.c_str());
        }
    }
    ImGui::SeparatorText("Consumers");
    for (int id : element.outputs) {
        const ElementNode* to = compiled_.find(id);
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
