/**
 * TESTS:
 * - ddimScheduleLeadingSpacing: 8 steps give diffusers' timesteps 876 ... 1, each step leading to the
 *   next one's alpha and the last one to the first training step's
 * - ddimScheduleMatchesExports: the schedule matches the timesteps and alphas export_denoiser.py
 *   wrote next to each exported model (skipped without exports)
 * - ddimScheduleRejectsStepCounts: 0 and more than 1000 steps throw
 * - latentNoiseIsStandardNormal: a seed gives the same noise every time, with mean 0 and variance 1;
 *   other seeds give other noise
 * - ddimStepGuidesAndSteps: with the true noise predicted, a step recovers the clean latents and
 *   noises them to the previous alpha; guidance extrapolates from the negative prediction
 * - readFloatsChecksSize: a file of the wrong size is rejected with its size
 * - stagedNodesOnlyRun: DDIM Sampler and VAE Decoder feeding Present are errors on those nodes; the
 *   same nodes feeding a Preview are not
 * - validateStableDiffusionParams: missing files, latent sizes that are not multiples of 8 and step
 *   counts outside 1..1000 are errors
 * - promptTokensAreInt64Tensors: the Prompt's ids and mask are 2x77 int64 tensor pins; they connect
 *   like any tensor, and a layer that needs float32 reports them
 * - runStagesSplitAtStagedNodes: nodes before the sampler run in stage 0, the decoder in 1 and the
 *   sinks after it in 2
 * - roundTripStableDiffusionParams: prompt, noise, sampler and decoder parameters survive
 *   toJson/fromJson
 * - planChecksModelShapes: with the exported models, the example graph plans without errors, shapes
 *   propagate to 1x3xHxW, and a latent size the UNet was not exported for is an error (skipped
 *   without exports)
 * - runReproducesReference: the 128x128 graph with the exported initial latents and 8 steps
 *   reproduces the Python/ONNX Runtime image; the compiled graph holds the sampler's and the
 *   decoder's networks (GPU; skipped without exports)
 **/

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <numeric>

#include "klartraum/headless_frontend.hpp"
#include "klartraum/klartraum_core.hpp"
#include "klartraum/vulkan_helpers.hpp"

#include "studio/graph_compiler.hpp"
#include "studio/graph_runner.hpp"
#include "studio/graph_serialization.hpp"
#include "studio/stable_diffusion.hpp"
#include "studio/tensor_shapes.hpp"

using namespace kstudio;

namespace {

const std::filesystem::path kRoot = KLARTRAUM_SOURCE_DIR;

// Written by klartraum's scripts/sd15_onnx/export_denoiser.py; not committed.
std::filesystem::path exportDir(uint32_t size) {
    return kRoot / "data/onnx" / ("sd15_denoiser_" + std::to_string(size));
}

bool hasExport(uint32_t size) {
    return std::filesystem::exists(exportDir(size) / "sd15_unet.onnx");
}

// The prompts export_denoiser.py renders its reference with.
const std::string kPrompt =
    "a realistic photograph of a traditional Japanese stone lantern in a green garden, "
    "single gray granite garden lantern, centered, moss, natural daylight";
const std::string kNegativePrompt =
    "person, building, house, flower pot, collage, multiple images, metal, painting, "
    "illustration, abstract, blurry, distorted, oversaturated, text";

template <typename T> std::vector<T> readAll(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    std::vector<T> values(static_cast<size_t>(in.tellg()) / sizeof(T));
    in.seekg(0);
    in.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(T)));
    return values;
}

int findKind(const Graph& graph, NodeKind kind) {
    for (const auto& node : graph.nodes()) {
        if (node.kind == kind) {
            return node.id;
        }
    }
    return -1;
}

bool hasError(const std::vector<Diagnostic>& diagnostics, int node) {
    return std::any_of(diagnostics.begin(), diagnostics.end(),
                       [&](const Diagnostic& d) { return d.node == node && d.severity == Severity::Error; });
}

PinRef out(int node, int slot = 0) { return {node, PinDirection::Output, slot}; }
PinRef in(int node, int slot = 0) { return {node, PinDirection::Input, slot}; }

} // namespace

TEST(StableDiffusion, ddimScheduleLeadingSpacing) {
    const DdimSchedule schedule = ddimSchedule(8);
    EXPECT_EQ(schedule.timesteps, (std::vector<int64_t>{876, 751, 626, 501, 376, 251, 126, 1}));
    ASSERT_EQ(schedule.alphas.size(), 8u);
    ASSERT_EQ(schedule.previousAlphas.size(), 8u);
    for (size_t i = 0; i + 1 < schedule.alphas.size(); ++i) {
        EXPECT_FLOAT_EQ(schedule.previousAlphas[i], schedule.alphas[i + 1]);
        EXPECT_LT(schedule.alphas[i], schedule.alphas[i + 1]);  // less noise with every step
    }
    // After t=1 comes the first training step, alpha = 1 - beta_0.
    EXPECT_NEAR(schedule.previousAlphas.back(), 1.0 - 0.00085, 1e-7);

    const DdimSchedule single = ddimSchedule(1);
    EXPECT_EQ(single.timesteps, (std::vector<int64_t>{1}));
}

TEST(StableDiffusion, ddimScheduleMatchesExports) {
    int checked = 0;
    for (uint32_t size : {128u, 256u, 512u}) {
        const auto dir = exportDir(size);
        if (!std::filesystem::exists(dir / "scheduler_timesteps_i64.bin")) {
            continue;
        }
        const auto timesteps = readAll<int64_t>(dir / "scheduler_timesteps_i64.bin");
        const auto alphaPairs = readAll<float>(dir / "scheduler_alphas_f32.bin");
        const DdimSchedule schedule = ddimSchedule(static_cast<uint32_t>(timesteps.size()));
        EXPECT_EQ(schedule.timesteps, timesteps) << dir;
        ASSERT_EQ(alphaPairs.size(), 2 * timesteps.size());
        for (size_t i = 0; i < timesteps.size(); ++i) {
            EXPECT_NEAR(schedule.alphas[i], alphaPairs[2 * i], 1e-5f * alphaPairs[2 * i]) << "step " << i;
            EXPECT_NEAR(schedule.previousAlphas[i], alphaPairs[2 * i + 1], 1e-5f * alphaPairs[2 * i + 1])
                << "step " << i;
        }
        ++checked;
    }
    if (checked == 0) {
        GTEST_SKIP() << "no SD 1.5 exports below " << (kRoot / "data/onnx");
    }
}

TEST(StableDiffusion, ddimScheduleRejectsStepCounts) {
    EXPECT_THROW(ddimSchedule(0), std::invalid_argument);
    EXPECT_THROW(ddimSchedule(1001), std::invalid_argument);
    EXPECT_EQ(ddimSchedule(1000).timesteps.size(), 1000u);
}

TEST(StableDiffusion, latentNoiseIsStandardNormal) {
    const auto noise = latentNoise(42, 100001);
    EXPECT_EQ(noise, latentNoise(42, 100001));
    EXPECT_NE(noise, latentNoise(43, 100001));
    const double mean = std::accumulate(noise.begin(), noise.end(), 0.0) / noise.size();
    double variance = 0.0;
    for (float v : noise) {
        variance += (v - mean) * (v - mean);
    }
    variance /= noise.size();
    EXPECT_NEAR(mean, 0.0, 0.01);
    EXPECT_NEAR(variance, 1.0, 0.02);
}

TEST(StableDiffusion, ddimStepGuidesAndSteps) {
    const float alpha = 0.3f;
    const float previousAlpha = 0.7f;
    const std::vector<float> clean{0.5f, -1.0f, 2.0f};
    const std::vector<float> noise{1.0f, 0.25f, -0.5f};
    std::vector<float> latents(3);
    std::vector<float> expected(3);
    for (size_t i = 0; i < 3; ++i) {
        latents[i] = std::sqrt(alpha) * clean[i] + std::sqrt(1.0f - alpha) * noise[i];
        expected[i] = std::sqrt(previousAlpha) * clean[i] + std::sqrt(1.0f - previousAlpha) * noise[i];
    }

    // Both prompts predict the true noise: guidance changes nothing.
    std::vector<float> prediction = noise;
    prediction.insert(prediction.end(), noise.begin(), noise.end());
    std::vector<float> stepped = latents;
    ddimStep(stepped, prediction, 7.5f, alpha, previousAlpha);
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(stepped[i], expected[i], 1e-5f);
    }

    // Guidance 2 extrapolates from the negative prediction through the
    // positive one: negative n - d and positive n - d/2 give n.
    std::vector<float> guided(6);
    for (size_t i = 0; i < 3; ++i) {
        guided[i] = noise[i] - 0.4f;
        guided[3 + i] = noise[i] - 0.2f;
    }
    stepped = latents;
    ddimStep(stepped, guided, 2.0f, alpha, previousAlpha);
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(stepped[i], expected[i], 1e-5f);
    }

    EXPECT_THROW(ddimStep(stepped, noise, 1.0f, alpha, previousAlpha), std::invalid_argument);
}

TEST(StableDiffusion, readFloatsChecksSize) {
    const auto path = std::filesystem::temp_directory_path() / "klartraum_studio_floats.bin";
    {
        std::ofstream file(path, std::ios::binary);
        const float values[3] = {1.0f, 2.0f, 3.0f};
        file.write(reinterpret_cast<const char*>(values), sizeof(values));
    }
    EXPECT_EQ(readFloats(path, 3), (std::vector<float>{1.0f, 2.0f, 3.0f}));
    try {
        readFloats(path, 4);
        ADD_FAILURE() << "expected an exception";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("12 bytes"), std::string::npos) << e.what();
    }
    EXPECT_THROW(readFloats(path.string() + ".missing", 3), std::runtime_error);
    std::filesystem::remove(path);
}

TEST(StableDiffusion, stagedNodesOnlyRun) {
    Graph graph = makeStableDiffusionGraph("models", 256, "a lantern", "", "out.png");
    const int sampler = findKind(graph, NodeKind::DdimSampler);
    const int decoder = findKind(graph, NodeKind::VaeDecoder);
    EXPECT_FALSE(hasError(graph.validate(), sampler));
    EXPECT_FALSE(hasError(graph.validate(), decoder));
    EXPECT_FALSE(graph.hasErrors());

    const int toImage = graph.addNode(NodeKind::TensorToImage);
    const int present = graph.addNode(NodeKind::Present);
    ASSERT_FALSE(graph.connect(out(decoder), in(toImage)));
    ASSERT_FALSE(graph.connect(out(toImage), in(present)));
    const auto diagnostics = graph.validate();
    EXPECT_TRUE(hasError(diagnostics, sampler));
    EXPECT_TRUE(hasError(diagnostics, decoder));
    EXPECT_FALSE(hasError(diagnostics, toImage));
    EXPECT_FALSE(hasError(diagnostics, findKind(graph, NodeKind::Prompt)));
}

TEST(StableDiffusion, validateStableDiffusionParams) {
    Graph graph = makeStableDiffusionGraph("models", 256, "a lantern", "", "out.png");
    const int prompt = findKind(graph, NodeKind::Prompt);
    const int noise = findKind(graph, NodeKind::LatentNoise);
    const int sampler = findKind(graph, NodeKind::DdimSampler);
    graph.findNode(prompt)->as<PromptParams>().vocabulary.clear();
    graph.findNode(noise)->as<LatentNoiseParams>().width = 100;
    graph.findNode(sampler)->as<DdimSamplerParams>().steps = 0;
    const auto diagnostics = graph.validate();
    EXPECT_TRUE(hasError(diagnostics, prompt));
    EXPECT_TRUE(hasError(diagnostics, noise));
    EXPECT_TRUE(hasError(diagnostics, sampler));

    graph.findNode(sampler)->as<DdimSamplerParams>().steps = 1001;
    EXPECT_TRUE(hasError(graph.validate(), sampler));
    graph.findNode(sampler)->as<DdimSamplerParams>().steps = 1000;
    graph.findNode(sampler)->as<DdimSamplerParams>().path.clear();
    EXPECT_TRUE(hasError(graph.validate(), sampler));
    for (NodeKind kind : {NodeKind::TextEncoder, NodeKind::VaeDecoder}) {
        Graph empty;
        const int id = empty.addNode(kind);
        EXPECT_TRUE(hasError(empty.validate(), id)) << kindInfo(kind).title;
    }
}

TEST(StableDiffusion, promptTokensAreInt64Tensors) {
    Graph graph;
    const int prompt = graph.addNode(NodeKind::Prompt);
    const int encoder = graph.addNode(NodeKind::TextEncoder);
    const int multiply = graph.addNode(NodeKind::Multiply);
    const auto outputs = graph.outputPins(*graph.findNode(prompt));
    ASSERT_EQ(outputs.size(), 2u);
    EXPECT_EQ(outputs[0].type, PinType::Tensor);
    EXPECT_EQ(outputs[1].type, PinType::Tensor);
    EXPECT_FALSE(graph.connect(out(prompt, 0), in(encoder, 0)));
    EXPECT_FALSE(graph.connect(out(prompt, 1), in(encoder, 1)));
    EXPECT_FALSE(graph.connect(out(prompt, 1), in(multiply, 0)));

    const ShapeInference types = inferTensorShapes(graph, {});
    EXPECT_EQ(types.types.at({prompt, 0}), (TensorType{{2, 77}, ElementType::Int64}));
    EXPECT_EQ(types.types.at({prompt, 1}), (TensorType{{2, 77}, ElementType::Int64}));
    EXPECT_TRUE(std::any_of(types.diagnostics.begin(), types.diagnostics.end(), [&](const Diagnostic& d) {
        return d.node == multiply && d.message.find("float32") != std::string::npos;
    }));
    EXPECT_FALSE(types.types.contains({multiply, 0}));
}

TEST(StableDiffusion, runStagesSplitAtStagedNodes) {
    const Graph graph = makeStableDiffusionGraph("models", 256, "a lantern", "", "out.png");
    const auto stages = runStages(graph, graph.topologicalOrder());
    EXPECT_EQ(stages.at(findKind(graph, NodeKind::Prompt)), 0);
    EXPECT_EQ(stages.at(findKind(graph, NodeKind::TextEncoder)), 0);
    EXPECT_EQ(stages.at(findKind(graph, NodeKind::LatentNoise)), 0);
    EXPECT_EQ(stages.at(findKind(graph, NodeKind::DdimSampler)), 0);
    EXPECT_EQ(stages.at(findKind(graph, NodeKind::VaeDecoder)), 1);
    EXPECT_EQ(stages.at(findKind(graph, NodeKind::Preview)), 2);
    EXPECT_EQ(stages.at(findKind(graph, NodeKind::ImageFileWriter)), 2);
}

TEST(StableDiffusion, roundTripStableDiffusionParams) {
    Graph graph = makeStableDiffusionGraph("models/sd", 256, "a \"quoted\" lantern\nat night", "blurry", "out.png");
    auto& noise = graph.findNode(findKind(graph, NodeKind::LatentNoise))->as<LatentNoiseParams>();
    noise.seed = 1234;
    noise.path = "latents.bin";
    auto& sampler = graph.findNode(findKind(graph, NodeKind::DdimSampler))->as<DdimSamplerParams>();
    sampler.steps = 30;
    sampler.guidanceScale = 5.5f;

    const Graph loaded = fromJson(toJson(graph));
    EXPECT_EQ(toJson(loaded), toJson(graph));
    for (const auto& node : graph.nodes()) {
        EXPECT_EQ(loaded.findNode(node.id)->params, node.params) << node.title;
    }
    const auto& prompt = loaded.findNode(findKind(loaded, NodeKind::Prompt))->as<PromptParams>();
    EXPECT_EQ(prompt.prompt, "a \"quoted\" lantern\nat night");
    EXPECT_EQ(prompt.negativePrompt, "blurry");
    EXPECT_EQ(prompt.vocabulary, "models/sd/vocab.json");
}

TEST(StableDiffusion, planChecksModelShapes) {
    if (!hasExport(256)) {
        GTEST_SKIP() << "no export in " << exportDir(256);
    }
    OnnxInfoCache cache;
    const OnnxInfoProvider info = [&](const std::string& path, std::string& error) { return cache.get(path, &error); };
    Graph graph = makeStableDiffusionGraph(exportDir(256).string(), 256, kPrompt, kNegativePrompt, "out.png");
    const CompilePlan plan = planGraph(graph, info, [](const std::string& path) {
        return std::filesystem::exists(path);
    });
    for (const auto& d : plan.diagnostics) {
        EXPECT_NE(d.severity, Severity::Error) << d.message;
    }
    ASSERT_TRUE(plan.run.has_value());
    EXPECT_EQ(plan.run->types.at({findKind(graph, NodeKind::TextEncoder), 0}).shape, (TensorShape{2, 77, 768}));
    EXPECT_EQ(plan.run->types.at({findKind(graph, NodeKind::DdimSampler), 0}).shape, (TensorShape{1, 4, 32, 32}));
    EXPECT_EQ(plan.run->types.at({findKind(graph, NodeKind::VaeDecoder), 0}).shape, (TensorShape{1, 3, 256, 256}));

    const int noise = findKind(graph, NodeKind::LatentNoise);
    graph.findNode(noise)->as<LatentNoiseParams>().width = 512;
    EXPECT_TRUE(hasError(planGraph(graph, info).diagnostics, findKind(graph, NodeKind::DdimSampler)));
}

class StableDiffusionRunTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!hasExport(128)) {
            GTEST_SKIP() << "no export in " << exportDir(128);
        }
        klartraum::setAssetRoot(kRoot.string());
        frontend = std::make_unique<klartraum::HeadlessFrontend>();
        outputDir = std::filesystem::temp_directory_path() / "klartraum_studio_sd";
        std::filesystem::create_directories(outputDir);
        context.resolveInput = [](const std::string& path) {
            if (!std::filesystem::exists(path)) {
                throw std::runtime_error("file not found: " + path);
            }
            return std::filesystem::path(path);
        };
        context.resolveOutput = [this](const std::string& path) { return outputDir / path; };
        context.onnxInfo = [this](const std::string& path, std::string& error) { return onnx.get(path, &error); };
        context.hostStep = [this](const std::string& step, double) { steps.push_back(step); };
    }

    void TearDown() override {
        frontend.reset();
        klartraum::setAssetRoot("");
        std::filesystem::remove_all(outputDir);
    }

    std::unique_ptr<klartraum::HeadlessFrontend> frontend;
    std::filesystem::path outputDir;
    OnnxInfoCache onnx;
    RunContext context;
    std::vector<std::string> steps;
};

TEST_F(StableDiffusionRunTest, runReproducesReference) {
    const auto dir = exportDir(128);
    Graph graph = makeStableDiffusionGraph(dir.string(), 128, kPrompt, kNegativePrompt, "sd.png");
    const int noise = findKind(graph, NodeKind::LatentNoise);
    const int sampler = findKind(graph, NodeKind::DdimSampler);
    const int decoder = findKind(graph, NodeKind::VaeDecoder);
    const int preview = findKind(graph, NodeKind::Preview);
    graph.findNode(noise)->as<LatentNoiseParams>().path = (dir / "initial_latents_f32.bin").string();
    const auto timesteps = readAll<int64_t>(dir / "scheduler_timesteps_i64.bin");
    graph.findNode(sampler)->as<DdimSamplerParams>().steps = static_cast<uint32_t>(timesteps.size());

    const CompilePlan plan = planGraph(graph, context.onnxInfo);
    for (const auto& d : plan.diagnostics) {
        ASSERT_NE(d.severity, Severity::Error) << d.message;
    }
    const RunResult result = runGraph(frontend->getKlartraumEngine().getVulkanContext(), graph, *plan.run, context);

    // The reference is the decoder's [-1, 1] output.
    std::vector<float> reference = readAll<float>(dir / "pipeline_reference_f32.bin");
    ASSERT_EQ(reference.size(), 3u * 128 * 128);
    for (float& v : reference) {
        v = 0.5f * (v + 1.0f);
    }
    const ImageRGBA8 expected = tensorToImage(reference, 3, 128, 128);
    ASSERT_TRUE(result.images.contains(preview));
    const ImageRGBA8& image = result.images.at(preview);
    ASSERT_EQ(image.width, 128u);
    ASSERT_EQ(image.height, 128u);
    double difference = 0.0;
    for (size_t i = 0; i < image.pixels.size(); ++i) {
        difference += std::abs(int(image.pixels[i]) - int(expected.pixels[i]));
    }
    difference /= image.pixels.size();
    EXPECT_LT(difference, 4.0) << "mean absolute difference per channel, in 8-bit steps";
    ASSERT_EQ(result.written.size(), 1u);
    EXPECT_TRUE(std::filesystem::exists(result.written[0]));

    // Every stage's graph and the staged nodes' networks are in the view.
    auto owned = [&](int node, std::string_view type) {
        return std::any_of(result.compiled.nodes.begin(), result.compiled.nodes.end(), [&](const ElementNode& e) {
            return e.owner == node && e.type == type;
        });
    };
    EXPECT_TRUE(owned(findKind(graph, NodeKind::TextEncoder), "OnnxNetwork"));
    EXPECT_TRUE(owned(sampler, "OnnxNetwork"));
    EXPECT_TRUE(owned(decoder, "OnnxNetwork"));
    const auto stepLogs = std::count_if(steps.begin(), steps.end(), [](const std::string& s) {
        return s.find("step ") != std::string::npos;
    });
    EXPECT_EQ(stepLogs, static_cast<long>(timesteps.size()));
}
