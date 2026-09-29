#pragma once

#include <chrono>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <imgui.h>
#include <imgui_node_editor.h>

#include "studio/graph_compiler.hpp"
#include "studio/graph_introspection.hpp"
#include "studio/graph_model.hpp"
#include "studio/graph_runner.hpp"
#include "studio/onnx_info.hpp"
#include "studio/preview_texture.hpp"

struct GLFWwindow;

namespace klartraum {
class KlartraumEngine;
class InterfaceCameraOrbit;
class GaussianDataStandard;
} // namespace klartraum

namespace kstudio {

// The example graphs offered in File > Examples.
enum class Example { GaussianSplatting, Autoencoder, SplatAutoencoder, CombinedScenes, AnimatedScenes, StableDiffusion };
std::optional<Example> exampleFromName(std::string_view name);

struct StudioOptions {
    std::filesystem::path graphFile;  // empty: the example graph
    Example example = Example::GaussianSplatting;
    std::string scenePath;            // scene for the default graph
    SplattingBackend backend = SplattingBackend::Raster;
    bool profiling = false;
};

// The studio UI: an editable authoring graph, the compiled klartraum graph it
// produces, and the rendered result behind them. Create it after the ImGui
// context (e.g. the ImGuiFrontend) and destroy it before. `window`, if given,
// gets the document title and is closed by File > Quit.
class StudioApp {
public:
    StudioApp(klartraum::KlartraumEngine& engine, StudioOptions options, GLFWwindow* window = nullptr);
    ~StudioApp();

    StudioApp(const StudioApp&) = delete;
    StudioApp& operator=(const StudioApp&) = delete;

    // Builds the UI; called once per frame between ImGui's NewFrame and Render.
    void drawGui();

    // True once a graph has been compiled and added to the engine.
    bool hasAppliedGraph() const { return appliedPlan_.has_value(); }
    const ElementGraph& compiledGraph() const { return compiled_; }
    const std::string& lastError() const { return applyError_; }

    // For scripted use (e.g. the snapshot tool).
    void showTab(int tab) { requestedTab_ = tab; }
    void selectNode(int node) {
        selectedNode_ = node;
        pendingSelection_ = node;
    }
    void setHideBuffers(bool hide) { hideBuffers_ = hide; compiledLayoutDirty_ = true; }
    const Graph& graph() const { return graph_; }
    // Edits made through this reference must call Graph::touch().
    Graph& editableGraph() { return graph_; }
    void loadExample(Example example);
    // Executes the run part of the graph at the start of the next frame.
    void requestRun() { runRequested_ = true; }
    bool lastRunSucceeded() const { return lastRun_.has_value() && runError_.empty(); }
    const std::string& runError() const { return runError_; }
    // 0: the live graph, 1: the last run's graph.
    void showCompiled(int source) {
        compiledSource_ = source;
        compiledLayoutDirty_ = true;
        selectedElement_ = -1;
    }

private:
    // Graph documents
    void newDefaultGraph();
    void setGraph(Graph graph, std::filesystem::path file);
    bool openGraph(const std::filesystem::path& path);
    bool saveGraphTo(const std::filesystem::path& path);
    // Relative input paths are tried against the working directory, the graph
    // file's directory and the klartraum sources (for the bundled samples).
    std::optional<std::filesystem::path> resolveInputPath(const std::string& path) const;
    // Relative output paths are relative to the graph file, else the working
    // directory.
    std::filesystem::path resolveOutputPath(const std::string& path) const;
    std::shared_ptr<const OnnxModelInfo> onnxInfo(const std::string& path, std::string& error);

    // Compilation
    void updatePlan();
    bool apply(const LivePlan& plan, const Graph& graph);
    void installBuilder(const std::optional<LivePlan>& plan, const Graph& graph);
    // How the live graph and Run find their inputs.
    RunContext runContext();
    // A scene file's Gaussians, cached per file.
    std::shared_ptr<const std::vector<klartraum::Gaussian3D>> loadScene(const std::string& path, bool flipY);
    // Assembled and uploaded Gaussians, cached per parts.
    std::shared_ptr<klartraum::GaussianDataStandard> loadGaussians(const std::vector<GaussianPart>& parts);
    // How many Gaussians a loaded scene file has.
    std::optional<size_t> sceneCount(const SceneParams& scene);
    void syncCamera();
    void pushCameraParams(const CameraParams& params);
    void refreshProfiling();
    void run();
    // The graph the compiled view shows: the live graph or the last run's.
    const ElementGraph& shownCompiled() const {
        return compiledSource_ == 1 && lastRun_ ? lastRun_->compiled : compiled_;
    }
    const std::map<std::string, float>& shownTimings() const {
        return compiledSource_ == 1 && lastRun_ ? lastRun_->timings : timings_;
    }
    bool runIsOutdated() const { return lastRun_ && lastRunRevision_ != graph_.revision(); }

    // UI
    void setupStyle();
    void drawMenuBar();
    void drawOverview();
    void drawGraphWindow();
    void drawAuthoringEditor();
    void drawCompiledEditor();
    void drawInspector();
    void drawNodeInspector(Node& node);
    void drawElementInspector(const ElementNode& element);
    void drawDiagnostics(int node);
    void drawRunControls();
    // Draws a node's run result scaled to `width`; returns false if there is none.
    bool drawPreview(int node, float width);
    void drawFileDialog();
    void layoutAuthoringGraph();
    // Without heights, nodes are placed on a fixed grid; with heights
    // (measured from the editor), they are stacked without overlaps.
    void layoutCompiledGraph(bool measured);
    void deleteSelection();
    void drawEditorToolbar(bool compiled);
    void drawNodeHeader(ax::NodeEditor::NodeId node, ImVec2 headerMin, ImVec2 headerMax, ImU32 color);
    // Rings around a node that becomes elements of the live and/or run graph.
    void drawGraphRings(ax::NodeEditor::NodeId node, bool live, bool run);
    void drawLegend();
    // Inspector section: where and when a node runs, and what it is made of.
    void drawExecutionInfo(const Node& node);
    // The Gaussians an Upload Gaussians node of `graph` has uploaded, if any.
    std::shared_ptr<klartraum::GaussianDataStandard> uploadedGaussians(const Graph& graph, int node) const;
    void handleFit(int& pendingFrames);
    void setStatus(std::string message, bool error = false);
    void updateWindowTitle();

    klartraum::KlartraumEngine& engine_;
    GLFWwindow* window_ = nullptr;
    std::shared_ptr<klartraum::InterfaceCameraOrbit> camera_;
    StudioOptions options_;

    // Authoring graph
    Graph graph_;
    std::filesystem::path file_;
    bool modified_ = false;

    // Compilation state
    uint64_t plannedRevision_ = ~uint64_t{0};
    CompilePlan plan_;
    std::optional<LivePlan> appliedPlan_;
    Graph appliedGraph_;  // the graph appliedPlan_ was built from
    std::optional<LivePlan> failedPlan_;
    std::string applyError_;
    std::string builderError_;
    // GPU values the live graph reads, set from the graph before every frame.
    std::vector<HostBinding> liveBindings_;
    std::chrono::steady_clock::time_point start_ = std::chrono::steady_clock::now();
    void updateLiveValues();
    double secondsSinceStart() const;
    bool autoApply_ = true;
    bool applyRequested_ = false;
    std::map<std::string, std::shared_ptr<klartraum::GaussianDataStandard>> models_;  // by partsKey()
    std::map<std::pair<std::string, bool>, std::shared_ptr<const std::vector<klartraum::Gaussian3D>>> scenes_;
    // CPU work of the last live build and the last run, with durations; the
    // log being written while one of them builds.
    std::vector<std::string> liveHostSteps_;
    std::vector<std::string> runHostSteps_;
    std::vector<std::string>* hostLog_ = nullptr;
    void logHostStep(const std::string& step, double milliseconds);

    // Run
    OnnxInfoCache onnxInfo_;
    bool runRequested_ = false;
    bool autoRun_ = false;
    std::optional<RunResult> lastRun_;
    uint64_t lastRunRevision_ = 0;
    std::string runError_;
    std::map<int, std::unique_ptr<PreviewTexture>> previews_;

    // Compiled graph view
    ElementGraph compiled_;
    int compiledSource_ = 0;  // 0: live graph, 1: last run
    std::string compiledSignature_;
    bool compiledLayoutDirty_ = true;
    bool compiledLayoutNeedsMeasure_ = false;
    bool hideBuffers_ = false;
    int selectedElement_ = -1;

    // Profiling
    bool profiling_ = false;
    std::map<std::string, float> timings_;
    double lastProfileTime_ = 0.0;

    // Editors
    ax::NodeEditor::EditorContext* authoringEditor_ = nullptr;
    ax::NodeEditor::EditorContext* compiledEditor_ = nullptr;
    std::vector<int> nodesToPlace_;  // authoring nodes whose editor position must be set
    Vec2 newNodePosition_;           // canvas position for a node added from the menu
    int contextNode_ = -1;
    int contextLink_ = -1;
    int pendingSelection_ = -1;      // node to select in the editor on its next frame
    bool graphWindowHovered_ = false;
    bool fitRequested_ = true;
    int authoringFitFrames_ = 0;
    int compiledFitFrames_ = 0;
    int selectedNode_ = -1;
    int activeTab_ = 0;              // 0: authoring, 1: compiled
    int requestedTab_ = -1;

    // Dialogs / status
    enum class FileDialog { None, Open, SaveAs };
    FileDialog fileDialog_ = FileDialog::None;
    std::string fileDialogPath_;
    std::string status_;
    bool statusIsError_ = false;
    double statusTime_ = 0.0;
    std::string windowTitle_;
};

} // namespace kstudio
