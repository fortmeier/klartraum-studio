/**
 * TESTS:
 * - imageShapes: 1x1xHxW and 1x3xHxW are images, other shapes are not
 * - autoencoderShapes: shapes flow from the image file through encoder and decoder
 * - offscreenShapes: Image to Tensor takes its size from the Offscreen Target
 * - mismatchedModelInput: a tensor of the wrong shape into a model is reported on the model
 * - sinkNeedsImage: a Preview fed with a non-image tensor is reported on the Preview
 * - unreadableModel: a model the provider cannot read is reported with the provider's error
 * - unsupportedOperators: unsupported operators are listed on the model node
 **/

#include <gtest/gtest.h>

#include <algorithm>

#include "studio/tensor_shapes.hpp"

using namespace kstudio;

namespace {

OnnxModelInfo model(TensorShape in, TensorShape out, std::vector<std::string> unsupported = {}) {
    OnnxModelInfo info;
    info.inputs = {{"input", std::move(in)}};
    info.outputs = {{"output", std::move(out)}};
    info.unsupportedOps = std::move(unsupported);
    return info;
}

// Serves the two sample models by name.
OnnxInfoProvider provider(std::map<std::string, OnnxModelInfo> models) {
    return [models = std::move(models)](const std::string& path, std::string& error)
               -> std::shared_ptr<const OnnxModelInfo> {
        auto it = models.find(path);
        if (it == models.end()) {
            error = "cannot read " + path;
            return nullptr;
        }
        return std::make_shared<OnnxModelInfo>(it->second);
    };
}

OnnxInfoProvider sampleModels() {
    return provider({{"enc", model({1, 3, 128, 128}, {1, 128, 16, 16})},
                     {"dec", model({1, 128, 16, 16}, {1, 3, 128, 128})}});
}

int findKind(const Graph& graph, NodeKind kind, int skip = 0) {
    for (const auto& node : graph.nodes()) {
        if (node.kind == kind && skip-- == 0) {
            return node.id;
        }
    }
    return -1;
}

bool hasErrorOn(const ShapeInference& result, int node) {
    return std::any_of(result.diagnostics.begin(), result.diagnostics.end(),
                       [&](const Diagnostic& d) { return d.severity == Severity::Error && d.node == node; });
}

} // namespace

TEST(TensorShapes, imageShapes) {
    EXPECT_TRUE(isImageShape({1, 3, 8, 8}));
    EXPECT_TRUE(isImageShape({1, 1, 8, 4}));
    EXPECT_FALSE(isImageShape({1, 128, 16, 16}));
    EXPECT_FALSE(isImageShape({2, 3, 8, 8}));
    EXPECT_FALSE(isImageShape({3, 8, 8}));
}

TEST(TensorShapes, autoencoderShapes) {
    const Graph graph = makeAutoencoderGraph("in.png", "enc", "dec", "out.png");
    const ShapeInference result = inferTensorShapes(graph, sampleModels());
    EXPECT_TRUE(result.diagnostics.empty());
    EXPECT_EQ(result.shapes.at(findKind(graph, NodeKind::ImageFile)), (TensorShape{1, 3, 128, 128}));
    EXPECT_EQ(result.shapes.at(findKind(graph, NodeKind::OnnxModel, 0)), (TensorShape{1, 128, 16, 16}));
    EXPECT_EQ(result.shapes.at(findKind(graph, NodeKind::OnnxModel, 1)), (TensorShape{1, 3, 128, 128}));
}

TEST(TensorShapes, offscreenShapes) {
    Graph graph = makeSplatAutoencoderGraph("scene.spz", "enc", "dec");
    auto& target = graph.findNode(findKind(graph, NodeKind::OffscreenTarget))->as<OffscreenTargetParams>();
    target.width = 64;
    target.height = 32;
    const ShapeInference result = inferTensorShapes(graph, sampleModels());
    EXPECT_EQ(result.shapes.at(findKind(graph, NodeKind::ImageToTensor)), (TensorShape{1, 3, 32, 64}));
    // 64x32 does not fit the 128x128 encoder.
    EXPECT_TRUE(hasErrorOn(result, findKind(graph, NodeKind::OnnxModel, 0)));
}

TEST(TensorShapes, mismatchedModelInput) {
    Graph graph = makeAutoencoderGraph("in.png", "enc", "dec", "out.png");
    graph.findNode(findKind(graph, NodeKind::ImageFile))->as<ImageFileParams>().width = 64;
    const ShapeInference result = inferTensorShapes(graph, sampleModels());
    EXPECT_TRUE(hasErrorOn(result, findKind(graph, NodeKind::OnnxModel, 0)));
    EXPECT_FALSE(hasErrorOn(result, findKind(graph, NodeKind::OnnxModel, 1)));
}

TEST(TensorShapes, sinkNeedsImage) {
    Graph graph = makeAutoencoderGraph("in.png", "enc", "dec", "out.png");
    const int preview = findKind(graph, NodeKind::Preview);
    graph.connect({findKind(graph, NodeKind::OnnxModel, 0), PinDirection::Output, 0}, {preview, PinDirection::Input, 0});
    const ShapeInference result = inferTensorShapes(graph, sampleModels());
    EXPECT_TRUE(hasErrorOn(result, preview));
    EXPECT_FALSE(hasErrorOn(result, findKind(graph, NodeKind::ImageFileWriter)));
}

TEST(TensorShapes, unreadableModel) {
    const Graph graph = makeAutoencoderGraph("in.png", "missing", "dec", "out.png");
    const ShapeInference result = inferTensorShapes(graph, sampleModels());
    const int encoder = findKind(graph, NodeKind::OnnxModel, 0);
    ASSERT_TRUE(hasErrorOn(result, encoder));
    EXPECT_NE(result.diagnostics.front().message.find("cannot read missing"), std::string::npos);
}

TEST(TensorShapes, unsupportedOperators) {
    const Graph graph = makeAutoencoderGraph("in.png", "enc", "dec", "out.png");
    const ShapeInference result = inferTensorShapes(
        graph, provider({{"enc", model({1, 3, 128, 128}, {1, 128, 16, 16}, {"Softmax", "MatMul"})},
                         {"dec", model({1, 128, 16, 16}, {1, 3, 128, 128})}}));
    const int encoder = findKind(graph, NodeKind::OnnxModel, 0);
    ASSERT_TRUE(hasErrorOn(result, encoder));
    EXPECT_NE(result.diagnostics.front().message.find("Softmax, MatMul"), std::string::npos);
}
