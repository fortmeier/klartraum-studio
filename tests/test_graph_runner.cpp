/**
 * TESTS:
 * - runAutoencoderGraph: an image file runs through the sample encoder and decoder; the
 *   preview and the written PNG hold the same 128x128 image, and it resembles the input
 *   the way the autoencoder's reconstruction should (GPU)
 * - runSplatAutoencoderGraph: a Gaussian splatting rendered offscreen runs through image
 *   to tensor, encoder and decoder; the preview is a non-black 128x128 image and the
 *   compiled graph attributes the ONNX layers to their model nodes (GPU)
 * - runTensorToImage: an image file through Tensor to Image into a writer, and back through
 *   Image to Tensor into a preview, reproduces the resized input (GPU)
 * - runWritesOffscreenRendering: a writer fed directly with an offscreen splatting writes a
 *   non-black image of the target's size (GPU)
 * - runResample: an image resampled to twice its size with nearest filtering repeats every
 *   pixel (GPU)
 * - liveProcessedGraph: the Gaussian splatting rendered into the swapchain, resampled, encoded,
 *   decoded and presented builds as the live graph and renders frames; Present's resample is
 *   marked as added by the studio, the nodes' own elements are not (GPU)
 * - liveReportsShapeMismatch: a model fed with the window-sized rendering fails the live build
 *   with the model's title (GPU)
 * - liveCombinedScenes: two scenes, one of them transformed, merge into one Gaussian
 *   Splatting that holds the Gaussians of both and renders frames (GPU)
 * - runAnimatedScenes: the lantern swung by Time -> Sine -> Upload Number -> Make Transform on
 *   the GPU looks different after 1 s and the same after a full swing of 4 s (GPU)
 * - liveAnimatedScenes: the animated graph builds live with one binding per uploaded number and
 *   unconnected transform input, the yaw binding follows the sine, and frames render (GPU)
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
#include "klartraum/computegraph/hostvalues.hpp"
#include "klartraum/klartraum_core.hpp"
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
        context.loadGaussians = [this](const std::vector<GaussianPart>& parts) {
            return std::make_shared<klartraum::GaussianDataStandard>(
                vc(), assembleGaussians(parts, [](const std::string& path, bool flipY) {
                    return std::make_shared<const std::vector<klartraum::Gaussian3D>>(
                        klartraum::loadGaussiansSpz(path, flipY));
                }));
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

TEST_F(GraphRunnerTest, runTensorToImage) {
    Graph graph;
    const int file = graph.addNode(NodeKind::ImageFile);
    const int toImage = graph.addNode(NodeKind::TensorToImage);
    const int toTensor = graph.addNode(NodeKind::ImageToTensor);
    const int writer = graph.addNode(NodeKind::ImageFileWriter);
    const int preview = graph.addNode(NodeKind::Preview);
    auto& p = graph.findNode(file)->as<ImageFileParams>();
    p.path = kImage;
    p.width = 96;
    p.height = 64;
    graph.findNode(writer)->as<ImageFileWriterParams>().path = "roundtrip.png";
    auto out = [](int node) { return PinRef{node, PinDirection::Output, 0}; };
    auto in = [](int node) { return PinRef{node, PinDirection::Input, 0}; };
    ASSERT_FALSE(graph.connect(out(file), in(toImage)).has_value());
    ASSERT_FALSE(graph.connect(out(toImage), in(writer)).has_value());
    ASSERT_FALSE(graph.connect(out(toImage), in(toTensor)).has_value());
    ASSERT_FALSE(graph.connect(out(toTensor), in(preview)).has_value());

    const RunResult result = runGraph(vc(), graph, plan(graph), context);

    const ImageRGBA8 input = resizeImage(loadImage(kImage), 96, 64);
    for (int sink : {writer, preview}) {
        const ImageRGBA8& image = result.images.at(sink);
        ASSERT_EQ(image.width, 96u);
        ASSERT_EQ(image.height, 64u);
        int maxError = 0;
        for (size_t i = 0; i < input.pixels.size(); ++i) {
            if (i % 4 != 3) {
                maxError = std::max(maxError, std::abs(int(image.pixels[i]) - int(input.pixels[i])));
            }
        }
        EXPECT_LE(maxError, 1) << "sink " << sink;
    }
    ASSERT_EQ(result.written.size(), 1u);
    EXPECT_EQ(loadImage(result.written[0]).pixels, result.images.at(writer).pixels);
}

TEST_F(GraphRunnerTest, runWritesOffscreenRendering) {
    Graph graph = makeSplatAutoencoderGraph(kScene, kEncoder, kDecoder);
    auto& target = graph.findNode(findKind(graph, NodeKind::OffscreenTarget))->as<OffscreenTargetParams>();
    target.width = 80;
    target.height = 48;
    // Only the rendering is written; the autoencoder is left out.
    for (NodeKind kind : {NodeKind::Preview, NodeKind::OnnxModel, NodeKind::OnnxModel, NodeKind::ImageToTensor}) {
        graph.removeNode(findKind(graph, kind));
    }
    const int writer = graph.addNode(NodeKind::ImageFileWriter);
    graph.findNode(writer)->as<ImageFileWriterParams>().path = "splats.png";
    ASSERT_FALSE(graph.connect({findKind(graph, NodeKind::GaussianSplatting), PinDirection::Output, 0},
                               {writer, PinDirection::Input, 0})
                     .has_value());

    const RunResult result = runGraph(vc(), graph, plan(graph), context);

    const ImageRGBA8& image = result.images.at(writer);
    EXPECT_EQ(image.width, 80u);
    EXPECT_EQ(image.height, 48u);
    EXPECT_FALSE(isConstant(image));
    EXPECT_GT(meanValue(image), 5.0) << "the splatting is black";
    ASSERT_EQ(result.written.size(), 1u);
    EXPECT_EQ(result.written[0], outputDir / "splats.png");
}

TEST_F(GraphRunnerTest, runResample) {
    Graph graph;
    const int file = graph.addNode(NodeKind::ImageFile);
    const int toImage = graph.addNode(NodeKind::TensorToImage);
    const int resample = graph.addNode(NodeKind::Resample);
    const int preview = graph.addNode(NodeKind::Preview);
    auto& p = graph.findNode(file)->as<ImageFileParams>();
    p.path = kImage;
    p.width = 24;
    p.height = 16;
    auto& r = graph.findNode(resample)->as<ResampleParams>();
    r.width = 48;
    r.height = 32;
    r.filter = ResampleFilter::Nearest;
    auto out = [](int node) { return PinRef{node, PinDirection::Output, 0}; };
    auto in = [](int node) { return PinRef{node, PinDirection::Input, 0}; };
    ASSERT_FALSE(graph.connect(out(file), in(toImage)).has_value());
    ASSERT_FALSE(graph.connect(out(toImage), in(resample)).has_value());
    ASSERT_FALSE(graph.connect(out(resample), in(preview)).has_value());

    const RunResult result = runGraph(vc(), graph, plan(graph), context);

    const ImageRGBA8 input = resizeImage(loadImage(kImage), 24, 16);
    const ImageRGBA8& image = result.images.at(preview);
    ASSERT_EQ(image.width, 48u);
    ASSERT_EQ(image.height, 32u);
    int maxError = 0;
    for (uint32_t y = 0; y < 32; ++y) {
        for (uint32_t x = 0; x < 48; ++x) {
            for (uint32_t c = 0; c < 3; ++c) {
                const int actual = image.pixels[(y * 48 + x) * 4 + c];
                const int expected = input.pixels[((y / 2) * 24 + x / 2) * 4 + c];
                maxError = std::max(maxError, std::abs(actual - expected));
            }
        }
    }
    EXPECT_LE(maxError, 1);
}

namespace {

// The Gaussian splatting graph with the rendering resampled to the
// encoder's size, encoded, decoded and presented; without `resample`, the
// window-sized rendering goes to the encoder directly.
Graph processedLiveGraph(bool resample) {
    Graph graph = makeGaussianSplattingGraph(kScene);
    const int splatting = findKind(graph, NodeKind::GaussianSplatting);
    const int toTensor = graph.addNode(NodeKind::ImageToTensor);
    const int encoder = graph.addNode(NodeKind::OnnxModel);
    const int decoder = graph.addNode(NodeKind::OnnxModel);
    const int toImage = graph.addNode(NodeKind::TensorToImage);
    graph.findNode(encoder)->as<OnnxModelParams>().path = kEncoder;
    graph.findNode(encoder)->title = "Encoder";
    graph.findNode(decoder)->as<OnnxModelParams>().path = kDecoder;
    auto out = [](int node) { return PinRef{node, PinDirection::Output, 0}; };
    auto in = [](int node) { return PinRef{node, PinDirection::Input, 0}; };
    if (resample) {
        const int node = graph.addNode(NodeKind::Resample);
        graph.connect(out(splatting), in(node));
        graph.connect(out(node), in(toTensor));
    } else {
        graph.connect(out(splatting), in(toTensor));
    }
    graph.connect(out(toTensor), in(encoder));
    graph.connect(out(encoder), in(decoder));
    graph.connect(out(decoder), in(toImage));
    graph.connect(out(toImage), in(findKind(graph, NodeKind::Present)));
    return graph;
}

} // namespace

TEST_F(GraphRunnerTest, liveProcessedGraph) {
    const Graph graph = processedLiveGraph(true);
    const CompilePlan compiled = planGraph(graph, context.onnxInfo);
    for (const auto& d : compiled.diagnostics) {
        EXPECT_NE(d.severity, Severity::Error) << d.message;
    }
    ASSERT_TRUE(compiled.live.has_value());

    auto& engine = frontend->getKlartraumEngine();
    const BuiltGraph built = buildLiveGraph(engine, graph, *compiled.live, context);
    const ElementGraph elements =
        introspect(built.root, built.owners, compiled.live->presentNode, built.inserted);
    for (int i = 0; i < 3; ++i) {
        engine.step();
    }
    vkDeviceWaitIdle(vc().getDevice());

    auto owns = [&](NodeKind kind, std::string_view type) {
        return std::any_of(elements.nodes.begin(), elements.nodes.end(), [&](const ElementNode& n) {
            return n.owner == findKind(graph, kind) && n.type == type;
        });
    };
    EXPECT_TRUE(owns(NodeKind::Resample, "ImageResample"));
    EXPECT_TRUE(owns(NodeKind::Present, "ImageResample"));
    EXPECT_TRUE(owns(NodeKind::OnnxModel, "OnnxNetwork"));
    EXPECT_TRUE(owns(NodeKind::SwapchainTarget, "ImageViewSrc"));
    // Present's resample into the swapchain is added by the studio; the
    // Resample node's own is what the node stands for.
    auto added = [&](NodeKind kind, std::string_view type) {
        return std::any_of(elements.nodes.begin(), elements.nodes.end(), [&](const ElementNode& n) {
            return n.owner == findKind(graph, kind) && n.type == type && n.inserted;
        });
    };
    EXPECT_TRUE(added(NodeKind::Present, "ImageResample"));
    EXPECT_FALSE(added(NodeKind::Resample, "ImageResample"));
    EXPECT_FALSE(added(NodeKind::OnnxModel, "OnnxNetwork"));
    engine.clearComputeGraphs();
}

TEST_F(GraphRunnerTest, liveReportsShapeMismatch) {
    const Graph graph = processedLiveGraph(false);
    const CompilePlan compiled = planGraph(graph, context.onnxInfo);
    ASSERT_TRUE(compiled.live.has_value());
    try {
        buildLiveGraph(frontend->getKlartraumEngine(), graph, *compiled.live, context);
        FAIL() << "expected the build to fail";
    } catch (const std::runtime_error& e) {
        EXPECT_TRUE(std::string(e.what()).starts_with("Encoder: the model expects a")) << e.what();
    }
}

TEST_F(GraphRunnerTest, liveCombinedScenes) {
    const std::string lantern = (kRoot / "data/lantern.spz").string();
    if (!std::filesystem::exists(lantern)) {
        GTEST_SKIP() << "sample not found: " << lantern;
    }
    TransformGaussiansParams placement;
    placement.translation = {0.6f, -1.0f, -0.6f};
    placement.rotation = {0.0f, 30.0f, 0.0f};
    const Graph graph = makeCombinedScenesGraph(kScene, SceneParams{lantern, true}, placement);
    const CompilePlan compiled = planGraph(graph);
    ASSERT_TRUE(compiled.live.has_value());

    std::vector<std::vector<GaussianPart>> requested;
    std::shared_ptr<klartraum::GaussianDataStandard> model;
    RunContext recording = context;
    recording.loadGaussians = [&](const std::vector<GaussianPart>& parts) {
        requested.push_back(parts);
        model = context.loadGaussians(parts);
        return model;
    };
    auto& engine = frontend->getKlartraumEngine();
    const BuiltGraph built = buildLiveGraph(engine, graph, *compiled.live, recording);
    for (int i = 0; i < 2; ++i) {
        engine.step();
    }
    vkDeviceWaitIdle(vc().getDevice());

    ASSERT_EQ(requested.size(), 1u);
    ASSERT_EQ(requested[0].size(), 2u);
    EXPECT_EQ(model->count(), klartraum::loadGaussiansSpz(kScene).size() +
                                  klartraum::loadGaussiansSpz(lantern, true).size());
    // The scene buffers belong to the Upload Gaussians node; the CPU nodes
    // before it build no elements.
    const ElementGraph elements = introspect(built.root, built.owners, compiled.live->presentNode);
    auto owned = [&](NodeKind kind) {
        return std::count_if(elements.nodes.begin(), elements.nodes.end(),
                             [&](const ElementNode& n) { return n.owner == findKind(graph, kind); });
    };
    EXPECT_EQ(owned(NodeKind::UploadGaussians), 7);
    EXPECT_EQ(owned(NodeKind::MergeGaussians), 0);
    EXPECT_EQ(owned(NodeKind::TransformGaussians), 0);
    EXPECT_EQ(owned(NodeKind::Scene), 0);
    engine.clearComputeGraphs();
}

namespace {

// The animated example, rendered offscreen into a Preview for Run.
Graph animatedRunGraph(const std::string& lantern) {
    MakeTransformParams placement;
    placement.translation = {0.68f, -1.0f, -0.68f};
    placement.scale = 1.6f;
    CameraParams camera;
    camera.azimuth = 1.57f;
    camera.elevation = -0.35f;
    camera.distance = 1.6f;
    camera.target = {0.6f, -0.9f, -0.7f};
    Graph graph = makeAnimatedScenesGraph(kScene, SceneParams{lantern, true}, placement,
                                          SineParams{30.0f, 0.25f, 0.0f, 20.0f}, camera);
    graph.removeNode(findKind(graph, NodeKind::Present));
    graph.removeNode(findKind(graph, NodeKind::SwapchainTarget));
    const int target = graph.addNode(NodeKind::OffscreenTarget);
    graph.findNode(target)->as<OffscreenTargetParams>() = {160, 120};
    const int preview = graph.addNode(NodeKind::Preview);
    const int splatting = findKind(graph, NodeKind::GaussianSplatting);
    graph.connect({target, PinDirection::Output, 0}, {splatting, PinDirection::Input, 2});
    graph.connect({splatting, PinDirection::Output, 0}, {preview, PinDirection::Input, 0});
    return graph;
}

int maxDifference(const ImageRGBA8& a, const ImageRGBA8& b) {
    int result = 0;
    for (size_t i = 0; i < a.pixels.size(); ++i) {
        result = std::max(result, std::abs(int(a.pixels[i]) - int(b.pixels[i])));
    }
    return result;
}

} // namespace

TEST_F(GraphRunnerTest, runAnimatedScenes) {
    const std::string lantern = (kRoot / "data/lantern.spz").string();
    if (!std::filesystem::exists(lantern)) {
        GTEST_SKIP() << "sample not found: " << lantern;
    }
    const Graph graph = animatedRunGraph(lantern);
    const RunPlan runPlan = plan(graph);
    const int preview = findKind(graph, NodeKind::Preview);
    auto renderAt = [&](double time) {
        RunContext timed = context;
        timed.time = time;
        return runGraph(vc(), graph, runPlan, timed).images.at(preview);
    };
    const ImageRGBA8 start = renderAt(0.0);
    EXPECT_GT(meanValue(start), 5.0) << "the rendering is black";
    EXPECT_GT(maxDifference(renderAt(1.0), start), 40) << "the lantern did not turn";
    EXPECT_LE(maxDifference(renderAt(4.0), start), 3) << "a full swing should end where it started";
}

TEST_F(GraphRunnerTest, liveAnimatedScenes) {
    const std::string lantern = (kRoot / "data/lantern.spz").string();
    if (!std::filesystem::exists(lantern)) {
        GTEST_SKIP() << "sample not found: " << lantern;
    }
    const Graph graph = makeAnimatedScenesGraph(kScene, SceneParams{lantern, true}, MakeTransformParams{},
                                                SineParams{30.0f, 0.25f, 0.0f, 20.0f});
    const CompilePlan compiled = planGraph(graph);
    for (const auto& d : compiled.diagnostics) {
        EXPECT_NE(d.severity, Severity::Error) << d.message;
    }
    ASSERT_TRUE(compiled.live.has_value());
    auto& engine = frontend->getKlartraumEngine();
    const BuiltGraph built = buildLiveGraph(engine, graph, *compiled.live, context);

    // The uploaded yaw, and the six Make Transform inputs taking the node's values.
    ASSERT_EQ(built.bindings.size(), 7u);
    const int upload = findKind(graph, NodeKind::UploadNumber);
    const auto yaw = std::find_if(built.bindings.begin(), built.bindings.end(),
                                  [&](const HostBinding& b) { return b.node == upload; });
    ASSERT_NE(yaw, built.bindings.end());
    applyBindings(graph, built.bindings, 1.0);
    EXPECT_NEAR(yaw->values->values()[0], 50.0f, 1e-3f);
    for (int i = 0; i < 3; ++i) {
        applyBindings(graph, built.bindings, i * 0.5);
        engine.step();
    }
    vkDeviceWaitIdle(vc().getDevice());
    engine.clearComputeGraphs();
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
