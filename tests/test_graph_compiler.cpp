/**
 * TESTS:
 * - planDefaultGraph: the default graph plans Present and everything feeding it, depth first
 * - planFailsOnErrors: a graph with errors yields no plan but keeps the diagnostics
 * - cameraChangesDoNotRebuild: camera parameters and node ids do not require new pipelines
 * - pipelineChangesRebuild: scene path, splatting parameters and new nodes require new pipelines
 * - planProcessedLiveGraph: a swapchain rendering resampled, encoded, decoded and presented is a
 *   valid live plan; a Preview reading from the swapchain is an error on the Preview
 * - gsplatConfigMapping: SplattingParams map field by field onto klartraum::GsplatConfig
 * - planRunGraph: a graph with sinks plans the nodes feeding them in dependency order
 * - planLiveAndRunIndependently: an error in the run part leaves the live plan intact and vice versa
 * - planReportsMissingFiles: missing scene and image files are errors on their nodes
 **/

#include <gtest/gtest.h>

#include <algorithm>
#include <functional>

#include "klartraum/gaussian_splatting_factory.hpp"
#include "klartraum/vulkan_gaussian_splatting_types.hpp"

#include "studio/graph_compiler.hpp"

using namespace kstudio;

namespace {

int findKind(const Graph& graph, NodeKind kind) {
    for (const auto& node : graph.nodes()) {
        if (node.kind == kind) {
            return node.id;
        }
    }
    return -1;
}

PinRef out(int node, int slot = 0) { return {node, PinDirection::Output, slot}; }
PinRef in(int node, int slot = 0) { return {node, PinDirection::Input, slot}; }

LivePlan livePlan(const Graph& graph) {
    const CompilePlan plan = planGraph(graph);
    EXPECT_TRUE(plan.live.has_value());
    return plan.live.value_or(LivePlan{});
}

} // namespace

TEST(GraphCompiler, planDefaultGraph) {
    const Graph graph = makeGaussianSplattingGraph("scene.spz", SplattingBackend::Raster);
    const CompilePlan plan = planGraph(graph);
    ASSERT_TRUE(plan.ok());
    const LivePlan& p = *plan.live;
    EXPECT_EQ(p.presentNode, findKind(graph, NodeKind::Present));
    EXPECT_EQ(p.cameraNode, findKind(graph, NodeKind::Camera));
    const std::vector<int> expected = {
        findKind(graph, NodeKind::Present), findKind(graph, NodeKind::GaussianSplatting),
        findKind(graph, NodeKind::UploadGaussians), findKind(graph, NodeKind::Scene),
        findKind(graph, NodeKind::Camera), findKind(graph, NodeKind::SwapchainTarget)};
    EXPECT_EQ(p.nodes, expected);
    EXPECT_TRUE(p.contains(findKind(graph, NodeKind::Scene)));
}

TEST(GraphCompiler, planFailsOnErrors) {
    Graph graph = makeGaussianSplattingGraph("scene.spz");
    graph.removeNode(findKind(graph, NodeKind::Scene));
    const CompilePlan plan = planGraph(graph);
    EXPECT_FALSE(plan.ok());
    EXPECT_FALSE(plan.diagnostics.empty());
}

TEST(GraphCompiler, cameraChangesDoNotRebuild) {
    Graph graph = makeGaussianSplattingGraph("scene.spz");
    const LivePlan before = livePlan(graph);

    graph.findNode(findKind(graph, NodeKind::Camera))->as<CameraParams>().distance = 5.0f;
    const int otherCamera = graph.addNode(NodeKind::Camera);
    graph.connect(out(otherCamera), in(findKind(graph, NodeKind::GaussianSplatting), 1));
    const LivePlan after = livePlan(graph);

    EXPECT_EQ(after.cameraNode, otherCamera);
    EXPECT_FALSE(after.needsRebuildFrom(before));
    // Corresponding nodes sit at the same positions.
    ASSERT_EQ(after.nodes.size(), before.nodes.size());
    EXPECT_EQ(after.nodes[4], otherCamera);
    EXPECT_EQ(before.nodes[4], findKind(graph, NodeKind::Camera));
}

TEST(GraphCompiler, pipelineChangesRebuild) {
    const Graph graph = makeGaussianSplattingGraph("scene.spz");
    const LivePlan base = livePlan(graph);
    auto changed = [&](const std::function<void(Graph&)>& change) {
        Graph other = graph;
        change(other);
        return livePlan(other).needsRebuildFrom(base);
    };
    auto splatting = [](Graph& g) -> SplattingParams& {
        return g.findNode(findKind(g, NodeKind::GaussianSplatting))->as<SplattingParams>();
    };

    EXPECT_TRUE(changed([](Graph& g) { g.findNode(findKind(g, NodeKind::Scene))->as<SceneParams>().path = "o.spz"; }));
    EXPECT_TRUE(changed([&](Graph& g) {
        auto& p = splatting(g);
        p.backend = p.backend == SplattingBackend::Raster ? SplattingBackend::Compute : SplattingBackend::Raster;
    }));
    EXPECT_TRUE(changed([&](Graph& g) { splatting(g).spreadMultiplier = 3.0f; }));
    EXPECT_TRUE(changed([&](Graph& g) { splatting(g).shDegree = 0; }));
    // A Resample between the rendering and Present.
    EXPECT_TRUE(changed([](Graph& g) {
        const int resample = g.addNode(NodeKind::Resample);
        g.connect(out(findKind(g, NodeKind::GaussianSplatting)), in(resample));
        g.connect(out(resample), in(findKind(g, NodeKind::Present)));
    }));
    EXPECT_FALSE(changed([](Graph& g) { g.findNode(findKind(g, NodeKind::Scene))->title = "Renamed"; }));
}

TEST(GraphCompiler, planProcessedLiveGraph) {
    // Rendering -> Resample -> Image to Tensor -> encoder -> decoder ->
    // Tensor to Image -> Present, all on the swapchain rendering.
    Graph graph = makeGaussianSplattingGraph("scene.spz");
    const int splatting = findKind(graph, NodeKind::GaussianSplatting);
    const int present = findKind(graph, NodeKind::Present);
    const int resample = graph.addNode(NodeKind::Resample);
    const int toTensor = graph.addNode(NodeKind::ImageToTensor);
    const int encoder = graph.addNode(NodeKind::OnnxModel);
    const int decoder = graph.addNode(NodeKind::OnnxModel);
    const int toImage = graph.addNode(NodeKind::TensorToImage);
    graph.findNode(encoder)->as<OnnxModelParams>().path = "enc.onnx";
    graph.findNode(decoder)->as<OnnxModelParams>().path = "dec.onnx";
    ASSERT_FALSE(graph.connect(out(splatting), in(resample)).has_value());
    ASSERT_FALSE(graph.connect(out(resample), in(toTensor)).has_value());
    ASSERT_FALSE(graph.connect(out(toTensor), in(encoder)).has_value());
    ASSERT_FALSE(graph.connect(out(encoder), in(decoder)).has_value());
    ASSERT_FALSE(graph.connect(out(decoder), in(toImage)).has_value());
    ASSERT_FALSE(graph.connect(out(toImage), in(present)).has_value());

    const CompilePlan plan = planGraph(graph);
    for (const auto& d : plan.diagnostics) {
        EXPECT_NE(d.severity, Severity::Error) << d.message;
    }
    ASSERT_TRUE(plan.live.has_value());
    EXPECT_EQ(plan.live->nodes.size(), 11u);
    EXPECT_FALSE(plan.run.has_value());

    // Run cannot read the swapchain.
    const int preview = graph.addNode(NodeKind::Preview);
    ASSERT_FALSE(graph.connect(out(resample), in(preview)).has_value());
    const CompilePlan withPreview = planGraph(graph);
    EXPECT_TRUE(withPreview.live.has_value());
    EXPECT_FALSE(withPreview.run.has_value());
    EXPECT_TRUE(std::any_of(withPreview.diagnostics.begin(), withPreview.diagnostics.end(), [&](const Diagnostic& d) {
        return d.severity == Severity::Error && d.node == preview;
    }));
}

TEST(GraphCompiler, gsplatConfigMapping) {
    SplattingParams params;
    params.spreadMultiplier = 3.5f;
    params.maxMod = 3;
    params.numSortWGsCap = 64;
    params.splatTileX = 16;
    params.splatTileY = 4;
    params.shDegree = 2;
    params.alphaCullThreshold = 0.1f;
    params.useMeshShader = true;

    const klartraum::GsplatConfig config = toGsplatConfig(params);
    EXPECT_EQ(config.spreadMultiplier, 3.5f);
    EXPECT_EQ(config.maxMod, 3u);
    EXPECT_EQ(config.numSortWGsCap, 64u);
    EXPECT_EQ(config.splatTileX, 16u);
    EXPECT_EQ(config.splatTileY, 4u);
    EXPECT_EQ(config.shDegree, 2);
    EXPECT_EQ(config.alphaCullThreshold, 0.1f);
    EXPECT_TRUE(config.useMeshShader);

    // Defaults agree with klartraum's.
    const klartraum::GsplatConfig defaults = toGsplatConfig(SplattingParams{});
    const klartraum::GsplatConfig reference;
    EXPECT_EQ(defaults.spreadMultiplier, reference.spreadMultiplier);
    EXPECT_EQ(defaults.maxMod, reference.maxMod);
    EXPECT_EQ(defaults.numSortWGsCap, reference.numSortWGsCap);
    EXPECT_EQ(defaults.splatTileX, reference.splatTileX);
    EXPECT_EQ(defaults.splatTileY, reference.splatTileY);
    EXPECT_EQ(defaults.shDegree, reference.shDegree);
    EXPECT_EQ(defaults.alphaCullThreshold, reference.alphaCullThreshold);
    EXPECT_EQ(defaults.useMeshShader, reference.useMeshShader);

    EXPECT_EQ(toGsplatBackend(SplattingBackend::Compute), klartraum::GsplatBackend::Compute);
    EXPECT_EQ(toGsplatBackend(SplattingBackend::Raster), klartraum::GsplatBackend::Raster);
}

TEST(GraphCompiler, planRunGraph) {
    const Graph graph = makeAutoencoderGraph("in.png", "enc.onnx", "dec.onnx", "out.png");
    const CompilePlan plan = planGraph(graph);
    EXPECT_FALSE(plan.live.has_value());
    ASSERT_TRUE(plan.run.has_value());
    EXPECT_EQ(plan.run->nodes.size(), 5u);
    EXPECT_EQ(plan.run->sinks.size(), 2u);
    EXPECT_EQ(plan.run->nodes.front(), findKind(graph, NodeKind::ImageFile));
}

TEST(GraphCompiler, planLiveAndRunIndependently) {
    // The live Gaussian-splatting graph plus a run part fed by an image file.
    Graph graph = makeGaussianSplattingGraph("scene.spz");
    const int image = graph.addNode(NodeKind::ImageFile);
    const int preview = graph.addNode(NodeKind::Preview);
    graph.connect({image, PinDirection::Output, 0}, {preview, PinDirection::Input, 0});

    // The image file has no path: only the run part is affected.
    CompilePlan plan = planGraph(graph);
    EXPECT_TRUE(plan.live.has_value());
    EXPECT_FALSE(plan.run.has_value());

    graph.findNode(image)->as<ImageFileParams>().path = "in.png";
    graph.findNode(findKind(graph, NodeKind::Scene))->as<SceneParams>().path.clear();
    graph.touch();
    plan = planGraph(graph);
    EXPECT_FALSE(plan.live.has_value());
    EXPECT_TRUE(plan.run.has_value());
}

TEST(GraphCompiler, planReportsMissingFiles) {
    const Graph graph = makeAutoencoderGraph("missing.png", "enc.onnx", "dec.onnx", "out.png");
    const auto exists = [](const std::string& path) { return path != "missing.png"; };
    const CompilePlan plan = planGraph(graph, {}, exists);
    EXPECT_FALSE(plan.run.has_value());
    EXPECT_TRUE(std::any_of(plan.diagnostics.begin(), plan.diagnostics.end(), [&](const Diagnostic& d) {
        return d.severity == Severity::Error && d.node == findKind(graph, NodeKind::ImageFile);
    }));
    EXPECT_TRUE(planGraph(graph, {}, [](const std::string&) { return true; }).run.has_value());
}
