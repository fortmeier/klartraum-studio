/**
 * TESTS:
 * - pinIdRoundTrip: pin ids encode node, direction and slot and decode back
 * - addAndRemoveNodes: nodes get unique ids and removing one drops its links
 * - connectMatchingTypes: an output connects to an input of the same pin type
 * - connectAcceptsReversedPins: dragging from input to output yields the same link
 * - connectRejectsTypeMismatch: pins of different types cannot be linked
 * - connectReplacesInputLink: linking into an occupied input replaces the old link
 * - connectRejectsCycles: self links and links closing a cycle are rejected
 * - revisionTracksChanges: edits increase the graph revision
 * - defaultGraphIsValid: the Gaussian-splatting graph validates without errors
 * - validateMissingPresent: a graph without Present is an error
 * - validateUnconnectedInput: a missing input link is reported on its node
 * - validateTargetMustBeSwapchain: splatting into another node's image is an error
 * - validateReportsUnusedNodes: nodes not feeding Present are reported as info
 **/

#include <gtest/gtest.h>

#include <algorithm>

#include "studio/graph_model.hpp"

using namespace kstudio;

namespace {

PinRef out(int node, int slot = 0) { return {node, PinDirection::Output, slot}; }
PinRef in(int node, int slot = 0) { return {node, PinDirection::Input, slot}; }

bool hasDiagnostic(const std::vector<Diagnostic>& diagnostics, Severity severity, int node) {
    return std::any_of(diagnostics.begin(), diagnostics.end(),
                       [&](const Diagnostic& d) { return d.severity == severity && d.node == node; });
}

int findKind(const Graph& graph, NodeKind kind) {
    for (const auto& node : graph.nodes()) {
        if (node.kind == kind) {
            return node.id;
        }
    }
    return -1;
}

} // namespace

TEST(GraphModel, pinIdRoundTrip) {
    for (int node : {1, 7, 1000}) {
        for (auto direction : {PinDirection::Input, PinDirection::Output}) {
            for (int slot = 0; slot < kMaxPinsPerDirection; ++slot) {
                const PinRef pin{node, direction, slot};
                EXPECT_EQ(pinFromId(pinId(pin)), pin);
            }
        }
    }
    EXPECT_NE(pinId(in(1, 0)), pinId(out(1, 0)));
}

TEST(GraphModel, addAndRemoveNodes) {
    Graph graph;
    const int a = graph.addNode(NodeKind::Scene);
    const int b = graph.addNode(NodeKind::GaussianSplatting);
    EXPECT_NE(a, b);
    ASSERT_FALSE(graph.connect(out(a), in(b, 0)).has_value());
    EXPECT_EQ(graph.links().size(), 1u);

    EXPECT_TRUE(graph.removeNode(a));
    EXPECT_EQ(graph.nodes().size(), 1u);
    EXPECT_TRUE(graph.links().empty());
    EXPECT_FALSE(graph.removeNode(a));

    // Ids are not reused.
    EXPECT_GT(graph.addNode(NodeKind::Scene), b);
}

TEST(GraphModel, connectMatchingTypes) {
    Graph graph;
    const int camera = graph.addNode(NodeKind::Camera);
    const int splatting = graph.addNode(NodeKind::GaussianSplatting);
    EXPECT_FALSE(graph.connect(out(camera), in(splatting, 1)).has_value());
    const Link* link = graph.inputLink(splatting, 1);
    ASSERT_NE(link, nullptr);
    EXPECT_EQ(link->fromNode, camera);
    EXPECT_EQ(graph.inputNode(splatting, 1)->id, camera);
}

TEST(GraphModel, connectAcceptsReversedPins) {
    Graph graph;
    const int camera = graph.addNode(NodeKind::Camera);
    const int splatting = graph.addNode(NodeKind::GaussianSplatting);
    EXPECT_FALSE(graph.connect(in(splatting, 1), out(camera)).has_value());
    ASSERT_NE(graph.inputLink(splatting, 1), nullptr);
    EXPECT_EQ(graph.inputLink(splatting, 1)->fromNode, camera);
}

TEST(GraphModel, connectRejectsTypeMismatch) {
    Graph graph;
    const int camera = graph.addNode(NodeKind::Camera);
    const int splatting = graph.addNode(NodeKind::GaussianSplatting);
    EXPECT_TRUE(graph.connect(out(camera), in(splatting, 0)).has_value());
    EXPECT_TRUE(graph.connect(out(camera), out(splatting)).has_value());
    EXPECT_TRUE(graph.links().empty());
}

TEST(GraphModel, connectReplacesInputLink) {
    Graph graph;
    const int sceneA = graph.addNode(NodeKind::Scene);
    const int sceneB = graph.addNode(NodeKind::Scene);
    const int splatting = graph.addNode(NodeKind::GaussianSplatting);
    ASSERT_FALSE(graph.connect(out(sceneA), in(splatting, 0)).has_value());
    ASSERT_FALSE(graph.connect(out(sceneB), in(splatting, 0)).has_value());
    EXPECT_EQ(graph.links().size(), 1u);
    EXPECT_EQ(graph.inputLink(splatting, 0)->fromNode, sceneB);
}

TEST(GraphModel, connectRejectsCycles) {
    Graph graph;
    const int a = graph.addNode(NodeKind::GaussianSplatting);
    const int b = graph.addNode(NodeKind::GaussianSplatting);
    EXPECT_TRUE(graph.connect(out(a), in(a, 2)).has_value());
    ASSERT_FALSE(graph.connect(out(a), in(b, 2)).has_value());
    EXPECT_TRUE(graph.connect(out(b), in(a, 2)).has_value());
    EXPECT_EQ(graph.links().size(), 1u);
}

TEST(GraphModel, revisionTracksChanges) {
    Graph graph;
    auto revision = graph.revision();
    const int scene = graph.addNode(NodeKind::Scene);
    EXPECT_GT(graph.revision(), revision);
    revision = graph.revision();
    const int splatting = graph.addNode(NodeKind::GaussianSplatting);
    graph.connect(out(scene), in(splatting, 0));
    EXPECT_GT(graph.revision(), revision);
    revision = graph.revision();
    graph.removeLink(graph.links().front().id);
    EXPECT_GT(graph.revision(), revision);
    revision = graph.revision();
    EXPECT_FALSE(graph.removeLink(12345));
    EXPECT_EQ(graph.revision(), revision);
}

TEST(GraphModel, defaultGraphIsValid) {
    const Graph graph = makeGaussianSplattingGraph("scene.spz");
    EXPECT_EQ(graph.nodes().size(), 5u);
    EXPECT_EQ(graph.links().size(), 4u);
    for (const auto& d : graph.validate()) {
        EXPECT_NE(d.severity, Severity::Error) << d.message;
    }
    EXPECT_FALSE(graph.hasErrors());
}

TEST(GraphModel, validateMissingPresent) {
    Graph graph = makeGaussianSplattingGraph("scene.spz");
    graph.removeNode(findKind(graph, NodeKind::Present));
    EXPECT_TRUE(hasDiagnostic(graph.validate(), Severity::Error, -1));
}

TEST(GraphModel, validateUnconnectedInput) {
    Graph graph = makeGaussianSplattingGraph("scene.spz");
    graph.removeNode(findKind(graph, NodeKind::Camera));
    EXPECT_TRUE(hasDiagnostic(graph.validate(), Severity::Error, findKind(graph, NodeKind::GaussianSplatting)));
}

TEST(GraphModel, validateTargetMustBeSwapchain) {
    Graph graph = makeGaussianSplattingGraph("scene.spz");
    const int first = findKind(graph, NodeKind::GaussianSplatting);
    const int second = graph.addNode(NodeKind::GaussianSplatting);
    graph.connect(out(findKind(graph, NodeKind::Scene)), in(second, 0));
    graph.connect(out(findKind(graph, NodeKind::Camera)), in(second, 1));
    ASSERT_FALSE(graph.connect(out(first), in(second, 2)).has_value());
    ASSERT_FALSE(graph.connect(out(second), in(findKind(graph, NodeKind::Present))).has_value());
    EXPECT_TRUE(hasDiagnostic(graph.validate(), Severity::Error, second));
}

TEST(GraphModel, validateReportsUnusedNodes) {
    Graph graph = makeGaussianSplattingGraph("scene.spz");
    const int extra = graph.addNode(NodeKind::Camera);
    const auto diagnostics = graph.validate();
    EXPECT_TRUE(hasDiagnostic(diagnostics, Severity::Info, extra));
    EXPECT_FALSE(graph.hasErrors());
}
