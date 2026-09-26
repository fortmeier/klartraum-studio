/**
 * TESTS:
 * - roundTripDefaultGraph: the default graph survives toJson/fromJson unchanged
 * - roundTripParameters: every node parameter is saved and restored
 * - missingFieldsKeepDefaults: parameters absent from a file keep their defaults
 * - rejectsMalformedFiles: wrong format tags, unknown kinds, bad links and invalid JSON throw
 * - saveAndLoadFile: saveGraph/loadGraph write and read a file on disk
 **/

#include <gtest/gtest.h>

#include <filesystem>

#include "studio/graph_serialization.hpp"

using namespace kstudio;

namespace {

void expectSameGraph(const Graph& a, const Graph& b) {
    ASSERT_EQ(a.nodes().size(), b.nodes().size());
    for (const auto& node : a.nodes()) {
        const Node* other = b.findNode(node.id);
        ASSERT_NE(other, nullptr);
        EXPECT_EQ(other->kind, node.kind);
        EXPECT_EQ(other->title, node.title);
        EXPECT_EQ(other->position, node.position);
        EXPECT_EQ(other->params.index(), node.params.index());
    }
    ASSERT_EQ(a.links().size(), b.links().size());
    for (const auto& link : a.links()) {
        const Link* other = b.inputLink(link.toNode, link.toSlot);
        ASSERT_NE(other, nullptr);
        EXPECT_EQ(other->fromNode, link.fromNode);
        EXPECT_EQ(other->fromSlot, link.fromSlot);
    }
}

} // namespace

TEST(GraphSerialization, roundTripDefaultGraph) {
    const Graph graph = makeGaussianSplattingGraph("3rdparty/spz/samples/racoonfamily.spz");
    const Graph loaded = fromJson(toJson(graph));
    expectSameGraph(graph, loaded);
    EXPECT_EQ(toJson(loaded), toJson(graph));
}

TEST(GraphSerialization, roundTripParameters) {
    Graph graph = makeGaussianSplattingGraph("a/b.spz", SplattingBackend::Raster);
    for (auto& node : graph.nodes()) {
        node.position = {12.5f, -3.0f};
        node.title += " (edited)";
        if (node.kind == NodeKind::Camera) {
            auto& p = node.as<CameraParams>();
            p.azimuth = 1.25f;
            p.elevation = 0.5f;
            p.distance = 3.0f;
            p.target = {1.0f, 2.0f, 3.0f};
            p.up = UpAxis::Z;
        } else if (node.kind == NodeKind::GaussianSplatting) {
            auto& p = node.as<SplattingParams>();
            p.spreadMultiplier = 3.25f;
            p.maxMod = 4;
            p.numSortWGsCap = 128;
            p.splatTileX = 16;
            p.splatTileY = 4;
            p.shDegree = 1;
            p.alphaCullThreshold = 0.05f;
            p.useMeshShader = true;
        }
    }
    const Graph loaded = fromJson(toJson(graph));
    expectSameGraph(graph, loaded);
    for (const auto& node : graph.nodes()) {
        const Node* other = loaded.findNode(node.id);
        switch (node.kind) {
        case NodeKind::Scene: EXPECT_EQ(other->as<SceneParams>().path, "a/b.spz"); break;
        case NodeKind::Camera: EXPECT_EQ(other->as<CameraParams>(), node.as<CameraParams>()); break;
        case NodeKind::GaussianSplatting: EXPECT_EQ(other->as<SplattingParams>(), node.as<SplattingParams>()); break;
        default: break;
        }
    }
}

TEST(GraphSerialization, missingFieldsKeepDefaults) {
    const Graph graph = fromJson(R"({
        "format": "klartraum-studio-graph", "version": 1,
        "nodes": [ {"id": 3, "kind": "gaussian_splatting", "params": {"backend": "raster"}},
                   {"id": 4, "kind": "camera"} ],
        "links": [] })");
    const Node* splatting = graph.findNode(3);
    ASSERT_NE(splatting, nullptr);
    SplattingParams expected;
    expected.backend = SplattingBackend::Raster;
    EXPECT_EQ(splatting->as<SplattingParams>(), expected);
    EXPECT_EQ(splatting->title, "Gaussian Splatting");
    EXPECT_EQ(graph.findNode(4)->as<CameraParams>(), CameraParams{});
}

TEST(GraphSerialization, rejectsMalformedFiles) {
    EXPECT_THROW(fromJson("not json"), std::runtime_error);
    EXPECT_THROW(fromJson(R"({"format": "something-else", "nodes": [], "links": []})"), std::runtime_error);
    EXPECT_THROW(fromJson(R"({"format": "klartraum-studio-graph", "version": 99, "nodes": [], "links": []})"),
                 std::runtime_error);
    EXPECT_THROW(fromJson(R"({"format": "klartraum-studio-graph", "nodes": [{"id": 1, "kind": "teapot"}],
                              "links": []})"),
                 std::runtime_error);
    EXPECT_THROW(fromJson(R"({"format": "klartraum-studio-graph",
                              "nodes": [{"id": 1, "kind": "camera"}, {"id": 1, "kind": "scene"}], "links": []})"),
                 std::runtime_error);
    // Camera output into a Gaussians input.
    EXPECT_THROW(fromJson(R"({"format": "klartraum-studio-graph",
                              "nodes": [{"id": 1, "kind": "camera"}, {"id": 2, "kind": "gaussian_splatting"}],
                              "links": [{"from": [1, 0], "to": [2, 0]}]})"),
                 std::runtime_error);
}

TEST(GraphSerialization, saveAndLoadFile) {
    const auto dir = std::filesystem::temp_directory_path() / "klartraum_studio_tests";
    std::filesystem::create_directories(dir);
    const auto path = dir / "graph.ktgraph.json";

    const Graph graph = makeGaussianSplattingGraph("scene.spz");
    saveGraph(graph, path);
    expectSameGraph(graph, loadGraph(path));

    std::filesystem::remove_all(dir);
    EXPECT_THROW(loadGraph(path), std::runtime_error);
}
