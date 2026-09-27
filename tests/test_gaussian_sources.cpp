/**
 * TESTS:
 * - toSimilarityRotatesXThenYThenZ: Euler angles apply about X first, then Y, then Z
 * - composeAppliesInnerFirst: composing similarities equals applying them one after the other
 * - partsOfCombinedScenes: the combined-scenes example yields the raccoon as it is and the
 *   lantern flipped and placed, in merge order
 * - partsOfNestedTransforms: transforms along a chain compose, outermost last
 * - partsKeyTellsPartsApart: different files, flips or placements give different keys
 * - assembleConcatenatesPlacedParts: parts are loaded, placed and concatenated in order
 **/

#include <gtest/gtest.h>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "studio/gaussian_sources.hpp"

using namespace kstudio;

namespace {

glm::vec3 apply(const Similarity& s, const glm::vec3& p) {
    const glm::quat q(s.rotation[3], s.rotation[0], s.rotation[1], s.rotation[2]);
    return s.scale * (q * p) + glm::vec3(s.translation[0], s.translation[1], s.translation[2]);
}

void expectNear(const glm::vec3& a, const glm::vec3& b) {
    EXPECT_NEAR(a.x, b.x, 1e-5f);
    EXPECT_NEAR(a.y, b.y, 1e-5f);
    EXPECT_NEAR(a.z, b.z, 1e-5f);
}

int findKind(const Graph& graph, NodeKind kind, int skip = 0) {
    for (const auto& node : graph.nodes()) {
        if (node.kind == kind && skip-- == 0) {
            return node.id;
        }
    }
    return -1;
}

PinRef out(int node) { return {node, PinDirection::Output, 0}; }
PinRef in(int node, int slot = 0) { return {node, PinDirection::Input, slot}; }

} // namespace

TEST(GaussianSources, toSimilarityRotatesXThenYThenZ) {
    TransformGaussiansParams params;
    params.rotation = {90.0f, 90.0f, 0.0f};
    // X by 90 degrees takes +Y to +Z; Y by 90 degrees then takes +Z to +X.
    expectNear(apply(toSimilarity(params), {0.0f, 1.0f, 0.0f}), {1.0f, 0.0f, 0.0f});

    params = {};
    params.translation = {1.0f, 2.0f, 3.0f};
    params.scale = 2.0f;
    expectNear(apply(toSimilarity(params), {1.0f, 1.0f, 1.0f}), {3.0f, 4.0f, 5.0f});
}

TEST(GaussianSources, composeAppliesInnerFirst) {
    TransformGaussiansParams a, b;
    a.translation = {0.5f, -1.0f, 2.0f};
    a.rotation = {10.0f, 20.0f, 30.0f};
    a.scale = 1.5f;
    b.translation = {-2.0f, 0.0f, 1.0f};
    b.rotation = {-40.0f, 5.0f, 70.0f};
    b.scale = 0.25f;
    const Similarity inner = toSimilarity(a), outer = toSimilarity(b);
    for (const glm::vec3 p : {glm::vec3(1.0f, 2.0f, 3.0f), glm::vec3(-0.5f, 0.0f, 4.0f)}) {
        expectNear(apply(compose(outer, inner), p), apply(outer, apply(inner, p)));
    }
}

TEST(GaussianSources, partsOfCombinedScenes) {
    TransformGaussiansParams placement;
    placement.translation = {0.5f, -1.0f, 0.0f};
    const Graph graph = makeCombinedScenesGraph("raccoon.spz", SceneParams{"lantern.spz", true}, placement);
    EXPECT_FALSE(graph.hasErrors());
    const auto parts = gaussianParts(graph, findKind(graph, NodeKind::MergeGaussians));
    ASSERT_EQ(parts.size(), 2u);
    EXPECT_EQ(parts[0], (GaussianPart{"raccoon.spz", false, {}}));
    EXPECT_EQ(parts[1].path, "lantern.spz");
    EXPECT_TRUE(parts[1].flipY);
    EXPECT_EQ(parts[1].transform, toSimilarity(placement));
}

TEST(GaussianSources, partsOfNestedTransforms) {
    Graph graph;
    const int scene = graph.addNode(NodeKind::Scene);
    const int first = graph.addNode(NodeKind::TransformGaussians);
    const int second = graph.addNode(NodeKind::TransformGaussians);
    graph.findNode(scene)->as<SceneParams>().path = "a.spz";
    auto& a = graph.findNode(first)->as<TransformGaussiansParams>();
    a.rotation = {0.0f, 90.0f, 0.0f};
    auto& b = graph.findNode(second)->as<TransformGaussiansParams>();
    b.translation = {0.0f, 0.0f, 5.0f};
    b.scale = 2.0f;
    ASSERT_FALSE(graph.connect(out(scene), in(first)).has_value());
    ASSERT_FALSE(graph.connect(out(first), in(second)).has_value());

    const auto parts = gaussianParts(graph, second);
    ASSERT_EQ(parts.size(), 1u);
    // +X turns to -Z, is doubled and moved by +5 along Z.
    expectNear(apply(parts[0].transform, {1.0f, 0.0f, 0.0f}), {0.0f, 0.0f, 3.0f});

    const int unconnected = graph.addNode(NodeKind::MergeGaussians);
    EXPECT_THROW(gaussianParts(graph, unconnected), std::runtime_error);
}

TEST(GaussianSources, partsKeyTellsPartsApart) {
    const std::vector<GaussianPart> base{{"a.spz", false, {}}};
    auto flipped = base;
    flipped[0].flipY = true;
    auto moved = base;
    moved[0].transform.translation[1] = 0.1f;
    auto twice = base;
    twice.push_back(base[0]);
    EXPECT_NE(partsKey(base), partsKey(flipped));
    EXPECT_NE(partsKey(base), partsKey(moved));
    EXPECT_NE(partsKey(base), partsKey(twice));
    EXPECT_EQ(partsKey(base), partsKey(std::vector<GaussianPart>{{"a.spz", false, {}}}));
}

TEST(GaussianSources, assembleConcatenatesPlacedParts) {
    auto scene = [](float x) {
        klartraum::Gaussian3D g{};
        g.position = {x, 0.0f, 0.0f};
        g.rotation = {0.0f, 0.0f, 0.0f, 1.0f};
        g.scale = {1.0f, 1.0f, 1.0f};
        return std::make_shared<const std::vector<klartraum::Gaussian3D>>(1, g);
    };
    int loads = 0;
    const GaussianLoader load = [&](const std::string& path, bool) {
        ++loads;
        return scene(path == "a.spz" ? 1.0f : 2.0f);
    };
    Similarity moved;
    moved.translation = {0.0f, 10.0f, 0.0f};
    moved.scale = 3.0f;
    const auto gaussians = assembleGaussians({{"a.spz", false, {}}, {"b.spz", false, moved}}, load);
    EXPECT_EQ(loads, 2);
    ASSERT_EQ(gaussians.size(), 2u);
    EXPECT_EQ(gaussians[0].position, (std::array<float, 3>{1.0f, 0.0f, 0.0f}));
    EXPECT_EQ(gaussians[1].position, (std::array<float, 3>{6.0f, 10.0f, 0.0f}));
    EXPECT_EQ(gaussians[1].scale, (std::array<float, 3>{3.0f, 3.0f, 3.0f}));
}
