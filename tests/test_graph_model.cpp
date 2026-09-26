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
 * - runOnlyGraphIsValid: a graph with sinks but no Present validates without errors
 * - validateNothingToDo: a graph without Present and sinks is an error
 * - validateOffscreenRules: Present needs a swapchain rendering, Image to Tensor an offscreen one
 * - validateMissingPaths: image, model and output files must be set
 * - upstreamAndOrder: upstreamOf collects dependencies; topologicalOrder places producers first
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

TEST(GraphModel, runOnlyGraphIsValid) {
    const Graph graph = makeAutoencoderGraph("in.png", "enc.onnx", "dec.onnx", "out.png");
    for (const auto& d : graph.validate()) {
        EXPECT_EQ(d.severity, Severity::Info) << d.message;
    }
    const Graph splat = makeSplatAutoencoderGraph("scene.spz", "enc.onnx", "dec.onnx");
    EXPECT_FALSE(splat.hasErrors());
}

TEST(GraphModel, validateNothingToDo) {
    Graph graph;
    graph.addNode(NodeKind::Scene);
    EXPECT_TRUE(hasDiagnostic(graph.validate(), Severity::Error, -1));
}

TEST(GraphModel, validateOffscreenRules) {
    // Present fed by a splatting that renders offscreen.
    Graph graph = makeGaussianSplattingGraph("scene.spz");
    const int offscreen = graph.addNode(NodeKind::OffscreenTarget);
    const int splatting = findKind(graph, NodeKind::GaussianSplatting);
    ASSERT_FALSE(graph.connect(out(offscreen), in(splatting, 2)).has_value());
    EXPECT_TRUE(hasDiagnostic(graph.validate(), Severity::Error, findKind(graph, NodeKind::Present)));

    // Image to Tensor fed by a splatting that renders into the swapchain.
    Graph other = makeGaussianSplattingGraph("scene.spz");
    const int toTensor = other.addNode(NodeKind::ImageToTensor);
    const int preview = other.addNode(NodeKind::Preview);
    ASSERT_FALSE(other.connect(out(findKind(other, NodeKind::GaussianSplatting)), in(toTensor)).has_value());
    ASSERT_FALSE(other.connect(out(toTensor), in(preview)).has_value());
    EXPECT_TRUE(hasDiagnostic(other.validate(), Severity::Error, toTensor));
    EXPECT_FALSE(hasDiagnostic(other.validate(), Severity::Error, findKind(other, NodeKind::Present)));
}

TEST(GraphModel, validateMissingPaths) {
    Graph graph = makeAutoencoderGraph("", "", "dec.onnx", "");
    const auto diagnostics = graph.validate();
    EXPECT_TRUE(hasDiagnostic(diagnostics, Severity::Error, findKind(graph, NodeKind::ImageFile)));
    EXPECT_TRUE(hasDiagnostic(diagnostics, Severity::Error, findKind(graph, NodeKind::OnnxModel)));
    EXPECT_TRUE(hasDiagnostic(diagnostics, Severity::Error, findKind(graph, NodeKind::ImageFileWriter)));
}

TEST(GraphModel, upstreamAndOrder) {
    const Graph graph = makeAutoencoderGraph("in.png", "enc.onnx", "dec.onnx", "out.png");
    const int image = findKind(graph, NodeKind::ImageFile);
    const int preview = findKind(graph, NodeKind::Preview);
    const int writer = findKind(graph, NodeKind::ImageFileWriter);

    auto upstream = graph.upstreamOf(preview);
    std::sort(upstream.begin(), upstream.end());
    EXPECT_EQ(upstream.size(), 4u);
    EXPECT_FALSE(std::binary_search(upstream.begin(), upstream.end(), writer));

    const auto order = graph.topologicalOrder();
    ASSERT_EQ(order.size(), graph.nodes().size());
    auto position = [&](int id) { return std::find(order.begin(), order.end(), id) - order.begin(); };
    for (const auto& link : graph.links()) {
        EXPECT_LT(position(link.fromNode), position(link.toNode));
    }
    EXPECT_EQ(order.front(), image);
}

TEST(GraphModel, validateReportsUnusedNodes) {
    Graph graph = makeGaussianSplattingGraph("scene.spz");
    const int extra = graph.addNode(NodeKind::Camera);
    const auto diagnostics = graph.validate();
    EXPECT_TRUE(hasDiagnostic(diagnostics, Severity::Info, extra));
    EXPECT_FALSE(graph.hasErrors());
}
