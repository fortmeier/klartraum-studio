/**
 * TESTS:
 * - categorizeElementTypes: klartraum element type names map to display categories
 * - introspectEmpty: a null root yields an empty graph
 * - introspectChain: a linear chain yields producers before consumers with slot-indexed edges
 * - introspectSharedInputs: an element used by several consumers appears once with all its consumers
 * - introspectGroupOutputs: a group is traversed through the output elements it reports as inputs
 * - introspectOwners: listed elements get their owner, unconsumed unlisted ones the default owner
 * - introspectOwnersPropagate: unlisted elements inherit the owner of their consumer
 * - builtGraphMatchesCompiledElements: for both Gaussian-splatting backends, the introspected
 *   graph holds exactly the elements klartraum compiles and renders (GPU)
 **/

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>

#include "klartraum/computegraph/computegraphgroup.hpp"
#include "klartraum/gaussian_data_standard.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/vulkan_helpers.hpp"

#include "studio/graph_compiler.hpp"
#include "studio/graph_introspection.hpp"
#include "studio/graph_runner.hpp"

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

class FakeElement : public klartraum::ComputeGraphElement {
public:
    FakeElement(std::string name, const char* type = "GeneralComputation") : type_(type) { setName(name); }
    const char* getType() const override { return type_; }
    void checkInput(klartraum::ComputeGraphElementPtr, int) override {}

private:
    const char* type_;
};

class FakeGroup : public klartraum::ComputeGraphGroup {
public:
    explicit FakeGroup(klartraum::ComputeGraphElementPtr output) {
        setName("Group");
        outputElements[0] = std::move(output);
    }
};

std::shared_ptr<FakeElement> element(const std::string& name, const char* type = "GeneralComputation") {
    return std::make_shared<FakeElement>(name, type);
}

int idOf(const ElementGraph& graph, const std::string& label) {
    for (const auto& node : graph.nodes) {
        if (node.label() == label) {
            return node.id;
        }
    }
    return -1;
}

} // namespace

TEST(GraphIntrospection, categorizeElementTypes) {
    EXPECT_EQ(categorize("BufferElement"), ElementCategory::Buffer);
    EXPECT_EQ(categorize("BufferElementSinglePath"), ElementCategory::Buffer);
    EXPECT_EQ(categorize("TensorElement"), ElementCategory::Buffer);
    EXPECT_EQ(categorize("UniformBufferObject"), ElementCategory::Uniform);
    EXPECT_EQ(categorize("ImageViewSrc"), ElementCategory::Image);
    EXPECT_EQ(categorize("GeneralComputation"), ElementCategory::Compute);
    EXPECT_EQ(categorize("BufferTransformation"), ElementCategory::Compute);
    EXPECT_EQ(categorize("RenderPass"), ElementCategory::Graphics);
    EXPECT_EQ(categorize("BufferToGraphicsBarrier"), ElementCategory::Sync);
    EXPECT_EQ(categorize("GaussianSplattingRaster"), ElementCategory::Group);
    EXPECT_EQ(categorize("Teapot"), ElementCategory::Other);
}

TEST(GraphIntrospection, introspectEmpty) {
    const ElementGraph graph = introspect(nullptr);
    EXPECT_TRUE(graph.empty());
    EXPECT_EQ(graph.root, -1);
}

TEST(GraphIntrospection, introspectChain) {
    auto buffer = element("Buf", "BufferElement");
    auto a = element("A");
    auto b = element("B");
    a->setInput(buffer, 0);
    b->setInput(a, 1);

    const ElementGraph graph = introspect(b);
    ASSERT_EQ(graph.nodes.size(), 3u);
    EXPECT_EQ(graph.nodes[0].label(), "Buf");
    EXPECT_EQ(graph.nodes[1].label(), "A");
    EXPECT_EQ(graph.nodes[2].label(), "B");
    EXPECT_EQ(graph.root, 2);
    EXPECT_EQ(graph.nodes[0].category, ElementCategory::Buffer);

    ASSERT_EQ(graph.edges.size(), 2u);
    EXPECT_EQ(graph.edges[1].from, 1);
    EXPECT_EQ(graph.edges[1].to, 2);
    EXPECT_EQ(graph.edges[1].slot, 1);
    EXPECT_EQ(graph.nodes[1].outputs, std::vector<int>{2});
}

TEST(GraphIntrospection, introspectSharedInputs) {
    auto shared = element("Shared", "UniformBufferObject");
    auto a = element("A");
    auto b = element("B");
    auto c = element("C");
    a->setInput(shared, 0);
    b->setInput(shared, 0);
    c->setInput(a, 0);
    c->setInput(b, 1);

    const ElementGraph graph = introspect(c);
    EXPECT_EQ(graph.nodes.size(), 4u);
    const ElementNode* node = graph.find(idOf(graph, "Shared"));
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->outputs.size(), 2u);
    EXPECT_EQ(graph.edges.size(), 4u);
}

TEST(GraphIntrospection, introspectGroupOutputs) {
    auto inner = element("Inner");
    auto last = element("Last");
    last->setInput(inner, 0);
    auto group = std::make_shared<FakeGroup>(last);

    const ElementGraph graph = introspect(group);
    ASSERT_EQ(graph.nodes.size(), 3u);
    EXPECT_EQ(graph.nodes[graph.root].label(), "Group");
    EXPECT_EQ(graph.nodes[graph.root].type, "ComputeGraphGroup");
    EXPECT_GE(idOf(graph, "Inner"), 0);
}

TEST(GraphIntrospection, introspectOwners) {
    auto buffer = element("Buf", "BufferElement");
    auto op = element("Op");
    op->setInput(buffer, 0);

    const ElementGraph graph = introspect(op, {{buffer.get(), 7}}, 3);
    EXPECT_EQ(graph.find(idOf(graph, "Buf"))->owner, 7);
    EXPECT_EQ(graph.find(idOf(graph, "Op"))->owner, 3);
}

TEST(GraphIntrospection, introspectOwnersPropagate) {
    auto buffer = element("Buf", "BufferElement");
    auto inner = element("Inner");
    inner->setInput(buffer, 0);
    auto group = std::make_shared<FakeGroup>(inner);
    auto consumer = element("Consumer");
    consumer->setInput(group, 0);

    const ElementGraph graph = introspect(consumer, {{group.get(), 5}, {consumer.get(), 9}});
    EXPECT_EQ(graph.find(idOf(graph, "Group"))->owner, 5);
    EXPECT_EQ(graph.find(idOf(graph, "Inner"))->owner, 5);
    EXPECT_EQ(graph.find(idOf(graph, "Buf"))->owner, 5);
    EXPECT_EQ(graph.find(idOf(graph, "Consumer"))->owner, 9);
}

TEST(GraphIntrospection, builtGraphMatchesCompiledElements) {
    const std::filesystem::path root = KLARTRAUM_SOURCE_DIR;
    const std::string scene = (root / "3rdparty/spz/samples/racoonfamily.spz").string();
    if (!std::filesystem::exists(scene)) {
        GTEST_SKIP() << "SPZ sample not found: " << scene;
    }
    klartraum::setAssetRoot(root.string());

    for (SplattingBackend backend : {SplattingBackend::Compute, SplattingBackend::Raster}) {
        SCOPED_TRACE(std::string(backendName(backend)));
        const Graph graph = makeGaussianSplattingGraph(scene, backend);
        const LivePlan plan = *planGraph(graph).live;

        klartraum::HeadlessFrontend frontend;
        auto& engine = frontend.getKlartraumEngine();
        engine.enableProfiling();
        auto model = std::make_shared<klartraum::GaussianDataStandard>(engine.getVulkanContext(), scene);
        RunContext context;
        context.loadScene = [&](const std::string&) { return model; };

        const BuiltGraph built = buildLiveGraph(engine, graph, plan, context);
        const ElementGraph introspected = introspect(built.root, built.owners, plan.presentNode);
        engine.step();

        // Profiling reports one entry per element klartraum compiled, in
        // its execution order.
        std::vector<std::string> compiled;
        for (const auto& [label, ms] : engine.getProfilingResults()) {
            compiled.push_back(label);
        }
        std::vector<std::string> labels;
        for (const auto& node : introspected.nodes) {
            labels.push_back(node.label());
        }
        std::sort(compiled.begin(), compiled.end());
        std::sort(labels.begin(), labels.end());
        EXPECT_EQ(labels, compiled);
        EXPECT_GT(introspected.nodes.size(), 10u);

        auto countOwned = [&](int owner) {
            return std::count_if(introspected.nodes.begin(), introspected.nodes.end(),
                                 [&](const ElementNode& n) { return n.owner == owner; });
        };
        EXPECT_EQ(countOwned(findKind(graph, NodeKind::Scene)), 7);
        EXPECT_EQ(countOwned(plan.cameraNode), 1);
        EXPECT_EQ(countOwned(findKind(graph, NodeKind::SwapchainTarget)), 1);
        // A rendering into the swapchain is presented as it is.
        EXPECT_EQ(introspected.nodes[introspected.root].owner, findKind(graph, NodeKind::GaussianSplatting));

        vkDeviceWaitIdle(engine.getVulkanContext().getDevice());
        engine.clearComputeGraphs();
    }
    klartraum::setAssetRoot("");
}
