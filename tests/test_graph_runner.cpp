/**
 * TESTS:
 * - runAutoencoderGraph: an image file runs through the sample encoder and decoder; the
 *   preview and the written PNG hold the same 128x128 image, and it resembles the input
 *   the way the autoencoder's reconstruction should (GPU)
 * - runSplatAutoencoderGraph: a Gaussian splatting rendered offscreen runs through image
 *   to tensor, encoder and decoder; the preview is a non-black 128x128 image and the
 *   compiled graph attributes the ONNX layers to their model nodes (GPU)
 * - runReportsFailingNode: a missing input file fails the run with the node's title (GPU)
 **/

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <cmath>
#include <numeric>
#include <set>

#include "klartraum/gaussian_data_standard.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/vulkan_helpers.hpp"

#include "studio/graph_runner.hpp"

using namespace kstudio;

namespace {

const std::filesystem::path kRoot = KLARTRAUM_SOURCE_DIR;
const std::string kEncoder = (kRoot / "data/onnx/simple_encoder.onnx").string();
const std::string kDecoder = (kRoot / "data/onnx/simple_decoder.onnx").string();
const std::string kImage = (kRoot / "data/lantern.jpg").string();
const std::string kScene = (kRoot / "3rdparty/spz/samples/racoonfamily.spz").string();

class GraphRunnerTest : public ::testing::Test {
protected:
    void SetUp() override {
        for (const auto& path : {kEncoder, kDecoder, kImage, kScene}) {
            if (!std::filesystem::exists(path)) {
                GTEST_SKIP() << "sample not found: " << path;
            }
        }
        klartraum::setAssetRoot(kRoot.string());
        outputDir = std::filesystem::temp_directory_path() / "klartraum_studio_runner";
        std::filesystem::remove_all(outputDir);
        std::filesystem::create_directories(outputDir);
        frontend = std::make_unique<klartraum::HeadlessFrontend>();

        context.resolveInput = [](const std::string& path) {
            if (!std::filesystem::exists(path)) {
                throw std::runtime_error("file not found: " + path);
            }
            return std::filesystem::path(path);
        };
        context.resolveOutput = [this](const std::string& path) { return outputDir / path; };
        context.loadScene = [this](const std::string& path) {
            return std::make_shared<klartraum::GaussianDataStandard>(vc(), path);
        };
        context.onnxInfo = [this](const std::string& path, std::string& error) { return onnx.get(path, &error); };
    }

    void TearDown() override {
        frontend.reset();
        klartraum::setAssetRoot("");
        std::filesystem::remove_all(outputDir);
    }

    klartraum::VulkanContext& vc() { return frontend->getKlartraumEngine().getVulkanContext(); }

    RunPlan plan(const Graph& graph) {
        const CompilePlan compiled = planGraph(graph, context.onnxInfo);
        for (const auto& d : compiled.diagnostics) {
            EXPECT_NE(d.severity, Severity::Error) << d.message;
        }
        EXPECT_TRUE(compiled.run.has_value());
        return compiled.run.value_or(RunPlan{});
    }

    static double meanValue(const ImageRGBA8& image) {
        double sum = 0.0;
        for (size_t i = 0; i < image.pixels.size(); i += 4) {
            sum += image.pixels[i] + image.pixels[i + 1] + image.pixels[i + 2];
        }
        return sum / (3.0 * image.width * image.height);
    }

    static bool isConstant(const ImageRGBA8& image) {
        return std::all_of(image.pixels.begin(), image.pixels.end(), [&](uint8_t v) { return v == image.pixels[0]; });
    }

    std::unique_ptr<klartraum::HeadlessFrontend> frontend;
    std::filesystem::path outputDir;
    OnnxInfoCache onnx;
    RunContext context;
};

int findKind(const Graph& graph, NodeKind kind, int skip = 0) {
    for (const auto& node : graph.nodes()) {
        if (node.kind == kind && skip-- == 0) {
            return node.id;
        }
    }
    return -1;
}

} // namespace

TEST_F(GraphRunnerTest, runAutoencoderGraph) {
    const Graph graph = makeAutoencoderGraph(kImage, kEncoder, kDecoder, "decoded.png");
    const RunResult result = runGraph(vc(), graph, plan(graph), context);

    const int preview = findKind(graph, NodeKind::Preview);
    const int writer = findKind(graph, NodeKind::ImageFileWriter);
    ASSERT_TRUE(result.images.contains(preview));
    ASSERT_TRUE(result.images.contains(writer));
    const ImageRGBA8& image = result.images.at(preview);
    EXPECT_EQ(image.width, 128u);
    EXPECT_EQ(image.height, 128u);
    EXPECT_FALSE(isConstant(image));
    EXPECT_EQ(result.images.at(writer).pixels, image.pixels);

    // The sample autoencoder reconstructs its input blurred: close on average,
    // far closer than a uniform image of the input's mean colour.
    const ImageRGBA8 input = resizeImage(loadImage(kImage), 128, 128);
    const double inputMean = meanValue(input);
    double reconstructionError = 0.0, uniformError = 0.0;
    for (size_t i = 0; i < input.pixels.size(); i += 4) {
        for (size_t c = 0; c < 3; ++c) {
            reconstructionError += std::abs(double(image.pixels[i + c]) - input.pixels[i + c]);
            uniformError += std::abs(inputMean - input.pixels[i + c]);
        }
    }
    const double pixels = 3.0 * 128 * 128;
    EXPECT_LT(reconstructionError / pixels, 20.0) << "mean absolute error per channel";
    EXPECT_LT(reconstructionError, 0.5 * uniformError);

    ASSERT_EQ(result.written.size(), 1u);
    EXPECT_EQ(result.written[0], outputDir / "decoded.png");
    const ImageRGBA8 written = loadImage(result.written[0]);
    EXPECT_EQ(written.pixels, image.pixels);

    // Every compiled element belongs to one of the graph's nodes (or the run root).
    std::set<int> owners;
    for (const auto& element : result.compiled.nodes) {
        owners.insert(element.owner);
    }
    EXPECT_TRUE(owners.contains(findKind(graph, NodeKind::OnnxModel, 0)));
    EXPECT_TRUE(owners.contains(findKind(graph, NodeKind::OnnxModel, 1)));
    EXPECT_TRUE(owners.contains(findKind(graph, NodeKind::ImageFile)));
}

TEST_F(GraphRunnerTest, runSplatAutoencoderGraph) {
    const Graph graph = makeSplatAutoencoderGraph(kScene, kEncoder, kDecoder);
    const RunResult result = runGraph(vc(), graph, plan(graph), context);

    const ImageRGBA8& image = result.images.at(findKind(graph, NodeKind::Preview));
    EXPECT_EQ(image.width, 128u);
    EXPECT_EQ(image.height, 128u);
    EXPECT_FALSE(isConstant(image));
    EXPECT_GT(meanValue(image), 5.0) << "the decoded splatting is black";
    EXPECT_TRUE(result.written.empty());

    const int encoder = findKind(graph, NodeKind::OnnxModel, 0);
    const auto convs = std::count_if(result.compiled.nodes.begin(), result.compiled.nodes.end(),
                                     [&](const ElementNode& n) { return n.owner == encoder && n.name.starts_with("Conv"); });
    EXPECT_EQ(convs, 3);
    EXPECT_TRUE(std::any_of(result.compiled.nodes.begin(), result.compiled.nodes.end(), [&](const ElementNode& n) {
        return n.owner == findKind(graph, NodeKind::GaussianSplatting) && n.category == ElementCategory::Group;
    }));
}

TEST_F(GraphRunnerTest, runReportsFailingNode) {
    Graph graph = makeAutoencoderGraph(kImage, kEncoder, kDecoder, "decoded.png");
    auto& image = *graph.findNode(findKind(graph, NodeKind::ImageFile));
    image.title = "Holiday photo";
    image.as<ImageFileParams>().path = (outputDir / "missing.png").string();
    try {
        runGraph(vc(), graph, plan(graph), context);
        FAIL() << "expected the run to fail";
    } catch (const std::runtime_error& e) {
        EXPECT_TRUE(std::string(e.what()).starts_with("Holiday photo: ")) << e.what();
    }
}
