#pragma once

#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "studio/graph_compiler.hpp"
#include "studio/graph_introspection.hpp"
#include "studio/graph_model.hpp"

struct GLFWwindow;
struct ImNodesContext;
struct ImNodesEditorContext;

namespace klartraum {
class KlartraumEngine;
class InterfaceCameraOrbit;
class GaussianDataStandard;
} // namespace klartraum

namespace kstudio {

struct StudioOptions {
    std::filesystem::path graphFile;  // empty: the default Gaussian-splatting graph
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
    void selectNode(int node) { selectedNode_ = node; }
    void setHideBuffers(bool hide) { hideBuffers_ = hide; compiledLayoutDirty_ = true; }
    const Graph& graph() const { return graph_; }
    // Edits made through this reference must call Graph::touch().
    Graph& editableGraph() { return graph_; }

private:
    // Graph documents
    void newDefaultGraph();
    void setGraph(Graph graph, std::filesystem::path file);
    bool openGraph(const std::filesystem::path& path);
    bool saveGraphTo(const std::filesystem::path& path);
    std::optional<std::filesystem::path> resolveScenePath(const std::string& path) const;

    // Compilation
    void updatePlan();
    bool apply(const SplattingPlan& plan);
    void installBuilder(const std::optional<SplattingPlan>& plan,
                        const std::shared_ptr<klartraum::GaussianDataStandard>& model);
    std::shared_ptr<klartraum::GaussianDataStandard> loadModel(const std::string& path);
    void syncCamera();
    void pushCameraParams(const CameraParams& params);
    void refreshProfiling();

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
    void drawFileDialog();
    void layoutAuthoringGraph();
    // Without heights, nodes are placed on a fixed grid; with heights
    // (measured from the editor), they are stacked without overlaps.
    void layoutCompiledGraph(bool measured);
    void deleteSelection();
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
    std::optional<SplattingPlan> appliedPlan_;
    std::optional<SplattingPlan> failedPlan_;
    std::string applyError_;
    std::string builderError_;
    bool autoApply_ = true;
    bool applyRequested_ = false;
    std::map<std::string, std::shared_ptr<klartraum::GaussianDataStandard>> models_;

    // Compiled graph view
    ElementGraph compiled_;
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
    ImNodesContext* imnodes_ = nullptr;
    ImNodesEditorContext* authoringEditor_ = nullptr;
    ImNodesEditorContext* compiledEditor_ = nullptr;
    std::vector<int> nodesToPlace_;  // authoring nodes whose editor position must be set
    std::optional<Vec2> addNodeScreenPos_;
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
