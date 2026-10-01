/**
 * TESTS:
 * - storeReusesStorage: storing a tensor, CPU values or an image again with the same type copies into
 *   the kept storage (same generation); another shape or element type makes new storage (next
 *   generation); every kept value reads back as stored
 * - runSplatsOverImages: in a Run, a Gaussian Splatting renders the lantern over an image drawn onto
 *   its target by a Composite (linked from the image file's tensor through the Tensor to Image node
 *   connectConverting inserts), and over an Offscreen Target cleared red by a Clear Image; away from
 *   the lantern each shows its background unchanged (GPU)
 * - liveReadsRunResult: the Stable Diffusion image, decoded by a Run and kept, is drawn onto the
 *   swapchain by a live Composite, and a live Gaussian Splatting renders over it; building the live graph before the Run
 *   fails with a message asking for it (GPU; skipped without the 128 px export)
 **/

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>

#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/gaussian_data_standard.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/klartraum_core.hpp"
#include "klartraum/offscreen_target.hpp"
#include "klartraum/vulkan_helpers.hpp"

#include "studio/graph_compiler.hpp"
#include "studio/graph_runner.hpp"
#include "studio/meta_nodes.hpp"
#include "studio/retained_results.hpp"

using namespace kstudio;

namespace {

const std::filesystem::path kRoot = KLARTRAUM_SOURCE_DIR;
const std::string kImage = (kRoot / "data/lantern.jpg").string();
const std::string kLantern = (kRoot / "data/lantern.spz").string();
const std::filesystem::path kSd128 = kRoot / "data/onnx/sd15_denoiser_128";

PinRef out(int node, int slot = 0) { return {node, PinDirection::Output, slot}; }
PinRef in(int node, int slot = 0) { return {node, PinDirection::Input, slot}; }

int findKind(const Graph& graph, NodeKind kind) {
    for (const auto& node : graph.nodes()) {
        if (node.kind == kind) {
            return node.id;
        }
    }
    return -1;
}

// An image's pixels as BGRA bytes (the swapchain format).
std::vector<uint8_t> readImage(klartraum::VulkanContext& vc, VkImage image, VkImageLayout layout,
                               VkExtent2D extent) {
    const VkDeviceSize bytes = VkDeviceSize{extent.width} * extent.height * 4;
    VkBuffer buffer;
    VkDeviceMemory memory;
    vc.createBuffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, buffer, memory);
    vc.submitImmediate([&](VkCommandBuffer commandBuffer) {
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {extent.width, extent.height, 1};
        vkCmdCopyImageToBuffer(commandBuffer, image, layout, buffer, 1, &region);
    });
    void* data;
    vkMapMemory(vc.getDevice(), memory, 0, bytes, 0, &data);
    std::vector<uint8_t> pixels(static_cast<uint8_t*>(data), static_cast<uint8_t*>(data) + bytes);
    vkUnmapMemory(vc.getDevice(), memory);
    vkDestroyBuffer(vc.getDevice(), buffer, nullptr);
    vkFreeMemory(vc.getDevice(), memory, nullptr);
    return pixels;
}

// Where the lantern's Gaussians are, and how far they spread (median
// distance), for pointing a camera at them.
struct Framing {
    std::array<float, 3> center{};
    float spread = 1.0f;
};

Framing frameLantern() {
    const auto gaussians = klartraum::loadGaussiansSpz(kLantern, true);
    Framing framing;
    for (int axis = 0; axis < 3; ++axis) {
        std::vector<float> values;
        for (const auto& g : gaussians) values.push_back(g.position[axis]);
        std::nth_element(values.begin(), values.begin() + values.size() / 2, values.end());
        framing.center[axis] = values[values.size() / 2];
    }
    std::vector<float> distances;
    for (const auto& g : gaussians) {
        float d = 0.0f;
        for (int axis = 0; axis < 3; ++axis) d += std::pow(g.position[axis] - framing.center[axis], 2.0f);
        distances.push_back(std::sqrt(d));
    }
    std::nth_element(distances.begin(), distances.begin() + distances.size() / 2, distances.end());
    framing.spread = distances[distances.size() / 2];
    return framing;
}

class RetainedTest : public ::testing::Test {
protected:
    void SetUp() override {
        klartraum::setAssetRoot(kRoot.string());
        frontend = std::make_unique<klartraum::HeadlessFrontend>();
        outputDir = std::filesystem::temp_directory_path() / "klartraum_studio_retained";
        std::filesystem::create_directories(outputDir);
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

    // A lantern Gaussian Splatting over `target` (input 2 still to connect),
    // with a camera that looks at the lantern.
    int addLanternSplatting(Graph& graph) {
        const Framing framing = frameLantern();
        const int scene = graph.addNode(NodeKind::Scene);
        graph.findNode(scene)->as<SceneParams>() = SceneParams{kLantern, true};
        const int upload = graph.addNode(NodeKind::UploadGaussians);
        const int camera = graph.addNode(NodeKind::Camera);
        auto& c = graph.findNode(camera)->as<CameraParams>();
        c.distance = 6.0f * framing.spread;
        // The orbit camera looks at minus its target.
        c.target = {-framing.center[0], -framing.center[1], -framing.center[2]};
        const int splatting = graph.addNode(NodeKind::GaussianSplatting);
        graph.connect(out(scene), in(upload));
        graph.connect(out(upload), in(splatting, 0));
        graph.connect(out(camera), in(splatting, 1));
        return splatting;
    }

    std::unique_ptr<klartraum::HeadlessFrontend> frontend;
    std::filesystem::path outputDir;
    OnnxInfoCache onnx;
    RunContext context;
};

// Mean absolute difference per channel over the pixels where `mask` is set.
double difference(const ImageRGBA8& a, const ImageRGBA8& b, const std::vector<bool>& mask) {
    double sum = 0.0;
    size_t count = 0;
    for (size_t pixel = 0; pixel < mask.size(); ++pixel) {
        if (!mask[pixel]) continue;
        for (int c = 0; c < 3; ++c) {
            sum += std::abs(int(a.pixels[pixel * 4 + c]) - int(b.pixels[pixel * 4 + c]));
        }
        count += 3;
    }
    return count == 0 ? 0.0 : sum / count;
}

} // namespace

TEST_F(RetainedTest, storeReusesStorage) {
    RetainedResults retained(vc());
    EXPECT_EQ(retained.find({1, 0}), nullptr);
    EXPECT_FALSE(retained.has({{1, 0}}));
    EXPECT_TRUE(retained.has({}));

    // A float tensor, copied on the GPU.
    auto tensor = vc().create<klartraum::TensorElement<float>>(
        std::vector<uint32_t>{1, 3, 2, 2},
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    tensor->_setup(vc(), 1);
    std::vector<float> values(12);
    for (size_t i = 0; i < values.size(); ++i) values[i] = 0.5f * static_cast<float>(i);
    tensor->getDataBuffer(0).memcopyFrom(values);
    retained.storeTensor({1, 0}, *tensor, "t");
    const uint64_t first = retained.generation();
    const RetainedValue* value = retained.find({1, 0});
    ASSERT_NE(value, nullptr);
    EXPECT_EQ(value->type, (TensorType{{1, 3, 2, 2}}));
    EXPECT_TRUE(value->tensor->isSinglePathStorage());
    auto kept = std::dynamic_pointer_cast<klartraum::TensorElement<float>>(value->tensor);
    std::vector<float> read(12);
    kept->getDataBuffer(0).memcopyTo(read);
    EXPECT_EQ(read, values);

    // The same type again: copied into the kept tensor.
    values.assign(12, 7.0f);
    tensor->getDataBuffer(0).memcopyFrom(values);
    retained.storeTensor({1, 0}, *tensor, "t");
    EXPECT_EQ(retained.generation(), first);
    EXPECT_EQ(retained.find({1, 0})->tensor, kept);
    kept->getDataBuffer(0).memcopyTo(read);
    EXPECT_EQ(read, values);

    // CPU values of another shape: new storage.
    retained.storeHost({1, 0}, HostTensor{{2, 3}, {1, 2, 3, 4, 5, 6}}, "t");
    EXPECT_GT(retained.generation(), first);
    EXPECT_EQ(retained.find({1, 0})->type, (TensorType{{2, 3}}));

    // An int64 tensor keeps its element type.
    auto ids = vc().create<klartraum::TensorElement<int64_t>>(
        std::vector<uint32_t>{2, 77},
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    ids->_setup(vc(), 1);
    retained.storeTensor({2, 1}, *ids, "ids");
    EXPECT_EQ(retained.find({2, 1})->type, (TensorType{{2, 77}, ElementType::Int64}));

    // An image, copied on the GPU; the same extent again keeps it.
    auto source = std::make_shared<klartraum::OffscreenTarget>(vc(), VkExtent2D{8, 8}, 1);
    vc().submitImmediate([&](VkCommandBuffer cmd) {
        klartraum::recordClearImage(cmd, source->getImage(0), {{0.0f, 1.0f, 0.0f, 1.0f}});
    });
    retained.storeImage({3, 0}, source->getImage(0), VK_IMAGE_LAYOUT_GENERAL, {8, 8}, "image");
    const uint64_t withImage = retained.generation();
    auto image = retained.find({3, 0})->image;
    ASSERT_NE(image, nullptr);
    retained.storeImage({3, 0}, source->getImage(0), VK_IMAGE_LAYOUT_GENERAL, {8, 8}, "image");
    EXPECT_EQ(retained.generation(), withImage);
    EXPECT_EQ(retained.find({3, 0})->image, image);
    const auto pixels = readImage(vc(), image->getImage(0), VK_IMAGE_LAYOUT_GENERAL, {8, 8});
    EXPECT_EQ(pixels[1], 255);  // green
    EXPECT_EQ(pixels[0], 0);

    EXPECT_TRUE(retained.has({{1, 0}, {2, 1}, {3, 0}}));
    retained.clear();
    EXPECT_FALSE(retained.has({{1, 0}}));
    EXPECT_GT(retained.generation(), withImage);
}

TEST_F(RetainedTest, runSplatsOverImages) {
    if (!std::filesystem::exists(kImage) || !std::filesystem::exists(kLantern)) {
        GTEST_SKIP() << "samples not found";
    }
    Graph graph;
    const int file = graph.addNode(NodeKind::ImageFile);
    graph.findNode(file)->as<ImageFileParams>() = ImageFileParams{kImage, 96, 96};
    // The image drawn onto a target: the tensor gets converted on the way.
    const int canvas = graph.addNode(NodeKind::OffscreenTarget);
    graph.findNode(canvas)->as<OffscreenTargetParams>() = OffscreenTargetParams{96, 96, true};
    const int drawn = graph.addNode(NodeKind::Composite);
    graph.connect(out(canvas), in(drawn, 0));
    int converted = -1;
    ASSERT_FALSE(graph.connectConverting(out(file), in(drawn, 1), &converted));
    ASSERT_GE(converted, 0);
    // A red target.
    const int target = graph.addNode(NodeKind::OffscreenTarget);
    graph.findNode(target)->as<OffscreenTargetParams>() = OffscreenTargetParams{96, 96, false};
    const int red = graph.addNode(NodeKind::ClearImage);
    graph.findNode(red)->as<ClearImageParams>().color = {1.0f, 0.0f, 0.0f, 1.0f};
    graph.connect(out(target), in(red));

    std::vector<int> previews;
    for (int background : {drawn, red}) {
        const int splatting = addLanternSplatting(graph);
        ASSERT_FALSE(graph.connect(out(background), in(splatting, 2)));
        const int preview = graph.addNode(NodeKind::Preview);
        graph.connect(out(splatting), in(preview));
        previews.push_back(preview);
    }
    // The image on its own.
    const int plainPreview = graph.addNode(NodeKind::Preview);
    graph.connect(out(file), in(plainPreview));

    const CompilePlan plan = planGraph(graph, context.onnxInfo);
    for (const auto& d : plan.diagnostics) {
        ASSERT_NE(d.severity, Severity::Error) << d.message;
    }
    const RunResult result = runGraph(vc(), plan.flat.graph, *plan.run, context);
    const ImageRGBA8& plain = result.images.at(plainPreview);
    ImageRGBA8 redImage = plain;
    for (size_t i = 0; i < redImage.pixels.size(); i += 4) {
        redImage.pixels[i] = 255;
        redImage.pixels[i + 1] = 0;
        redImage.pixels[i + 2] = 0;
    }

    // Where the lantern is: pixels the splats change over the red background.
    const ImageRGBA8& overRed = result.images.at(previews[1]);
    std::vector<bool> lantern(96 * 96), elsewhere(96 * 96);
    size_t covered = 0;
    for (size_t pixel = 0; pixel < lantern.size(); ++pixel) {
        const bool changed = std::abs(int(overRed.pixels[pixel * 4]) - 255) > 8 || overRed.pixels[pixel * 4 + 1] > 8 ||
                             overRed.pixels[pixel * 4 + 2] > 8;
        lantern[pixel] = changed;
        elsewhere[pixel] = !changed;
        covered += changed ? 1 : 0;
    }
    EXPECT_GT(covered, 50u) << "the lantern should be visible";
    EXPECT_LT(covered, lantern.size() / 2) << "the lantern should leave the background visible";

    // Away from the lantern, each shows its background.
    EXPECT_LT(difference(result.images.at(previews[0]), plain, elsewhere), 2.0);
    EXPECT_LT(difference(overRed, redImage, elsewhere), 2.0);
    // Over it, the image background differs from plain.
    EXPECT_GT(difference(result.images.at(previews[0]), plain, lantern), 10.0);
}

TEST_F(RetainedTest, liveReadsRunResult) {
    if (!std::filesystem::exists(kSd128 / "sd15_unet.onnx") || !std::filesystem::exists(kLantern)) {
        GTEST_SKIP() << "no export in " << kSd128;
    }
    Graph graph = makeStableDiffusionGraph(kSd128.string(), 128,
                                           "a realistic photograph of a green garden", "blurry", "sd.png");
    graph.findNode(findKind(graph, NodeKind::DdimSampler))->as<DdimSamplerParams>().steps = 2;
    int decoder = -1;
    for (const auto& node : graph.nodes()) {
        if (node.kind == NodeKind::Meta && node.as<MetaParams>().definition == kVaeDecoderDefinition) {
            decoder = node.id;
        }
    }
    const int toImage = graph.addNode(NodeKind::TensorToImage);
    graph.connect(out(decoder), in(toImage));
    const int swapchain = graph.addNode(NodeKind::SwapchainTarget);
    const int background = graph.addNode(NodeKind::Composite);  // stretched over the swapchain
    graph.connect(out(swapchain), in(background, 0));
    graph.connect(out(toImage), in(background, 1));
    const int splatting = addLanternSplatting(graph);
    ASSERT_FALSE(graph.connect(out(background), in(splatting, 2)));
    const int present = graph.addNode(NodeKind::Present);
    ASSERT_FALSE(graph.connect(out(splatting), in(present)));
    const int preview = findKind(graph, NodeKind::Preview);

    const CompilePlan plan = planGraph(graph, context.onnxInfo);
    for (const auto& d : plan.diagnostics) {
        ASSERT_NE(d.severity, Severity::Error) << d.message;
    }
    ASSERT_TRUE(plan.live && plan.run);
    ASSERT_EQ(plan.live->retained.size(), 1u);

    RetainedResults retained(vc());
    context.retained = &retained;
    auto& engine = frontend->getKlartraumEngine();
    // Without a Run, the live graph cannot be built.
    try {
        buildLiveGraph(engine, plan.flat.graph, *plan.live, context);
        FAIL() << "expected the live build to fail";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("press Run"), std::string::npos) << e.what();
    }
    engine.clearComputeGraphs();

    const RunResult result = runGraph(vc(), plan.flat.graph, *plan.run, context);
    ASSERT_TRUE(retained.has(plan.live->retained));
    const RetainedValue* value = retained.find(plan.live->retained[0]);
    ASSERT_NE(value->image, nullptr);
    EXPECT_EQ(value->image->extent().width, 128u);
    // What the Run kept is the image the Preview shows.
    const ImageRGBA8& decoded = result.images.at(preview);
    const auto kept = readImage(vc(), value->image->getImage(0), VK_IMAGE_LAYOUT_GENERAL, {128, 128});
    double keptDifference = 0.0;
    for (size_t pixel = 0; pixel < 128 * 128; ++pixel) {
        // BGRA against RGBA.
        keptDifference += std::abs(int(kept[pixel * 4]) - int(decoded.pixels[pixel * 4 + 2])) +
                          std::abs(int(kept[pixel * 4 + 1]) - int(decoded.pixels[pixel * 4 + 1])) +
                          std::abs(int(kept[pixel * 4 + 2]) - int(decoded.pixels[pixel * 4]));
    }
    EXPECT_LT(keptDifference / (128.0 * 128 * 3), 1.0);

    // The live graph renders the lantern over it, in every frame.
    const BuiltGraph built = buildLiveGraph(engine, plan.flat.graph, *plan.live, context);
    for (int frame = 0; frame < 4; ++frame) {
        engine.step();
    }
    vkDeviceWaitIdle(vc().getDevice());
    // Its elements read the kept image.
    const ElementGraph compiled = introspect(built.root);
    EXPECT_TRUE(std::any_of(compiled.nodes.begin(), compiled.nodes.end(),
                            [](const ElementNode& e) { return e.type == "SinglePathImage"; }));
    // The swapchain's corner shows the decoded image's corner (the lantern
    // is in the middle), stretched over the window by the Composite.
    const VkExtent2D extent = vc().getSwapChainExtent();
    const auto frame = readImage(vc(), vc().getSwapChainImage(0), VK_IMAGE_LAYOUT_GENERAL, extent);
    for (int c = 0; c < 3; ++c) {
        EXPECT_NEAR(frame[2 - c], decoded.pixels[c], 24) << "channel " << c;
    }
    engine.clearComputeGraphs();
}
