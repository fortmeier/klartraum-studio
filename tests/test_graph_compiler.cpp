/**
 * TESTS:
 * - planDefaultGraph: the default graph plans the chain Scene/Camera/Target -> Splatting -> Present
 * - planFailsOnErrors: a graph with errors yields no plan but keeps the diagnostics
 * - cameraChangesDoNotRebuild: camera parameters and node ids do not require new pipelines
 * - pipelineChangesRebuild: scene path, backend and splatting parameters require new pipelines
 * - gsplatConfigMapping: SplattingParams map field by field onto klartraum::GsplatConfig
 **/

#include <gtest/gtest.h>

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

} // namespace

TEST(GraphCompiler, planDefaultGraph) {
    const Graph graph = makeGaussianSplattingGraph("scene.spz", SplattingBackend::Raster);
    const CompilePlan plan = planGraph(graph);
    ASSERT_TRUE(plan.ok());
    const SplattingPlan& p = *plan.splatting;
    EXPECT_EQ(p.presentNode, findKind(graph, NodeKind::Present));
    EXPECT_EQ(p.splattingNode, findKind(graph, NodeKind::GaussianSplatting));
    EXPECT_EQ(p.sceneNode, findKind(graph, NodeKind::Scene));
    EXPECT_EQ(p.cameraNode, findKind(graph, NodeKind::Camera));
    EXPECT_EQ(p.targetNode, findKind(graph, NodeKind::SwapchainTarget));
    EXPECT_EQ(p.scenePath, "scene.spz");
    EXPECT_EQ(p.params.backend, SplattingBackend::Raster);
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
    const SplattingPlan before = *planGraph(graph).splatting;

    graph.findNode(findKind(graph, NodeKind::Camera))->as<CameraParams>().distance = 5.0f;
    const int otherCamera = graph.addNode(NodeKind::Camera);
    graph.connect({otherCamera, PinDirection::Output, 0},
                  {findKind(graph, NodeKind::GaussianSplatting), PinDirection::Input, 1});
    const SplattingPlan after = *planGraph(graph).splatting;

    EXPECT_EQ(after.cameraNode, otherCamera);
    EXPECT_FALSE(after.needsRebuildFrom(before));
}

TEST(GraphCompiler, pipelineChangesRebuild) {
    const Graph graph = makeGaussianSplattingGraph("scene.spz");
    const SplattingPlan base = *planGraph(graph).splatting;

    SplattingPlan scene = base;
    scene.scenePath = "other.spz";
    EXPECT_TRUE(scene.needsRebuildFrom(base));

    SplattingPlan backend = base;
    backend.params.backend = base.params.backend == SplattingBackend::Raster ? SplattingBackend::Compute
                                                                             : SplattingBackend::Raster;
    EXPECT_TRUE(backend.needsRebuildFrom(base));

    SplattingPlan spread = base;
    spread.params.spreadMultiplier = 3.0f;
    EXPECT_TRUE(spread.needsRebuildFrom(base));

    SplattingPlan sh = base;
    sh.params.shDegree = 0;
    EXPECT_TRUE(sh.needsRebuildFrom(base));
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
