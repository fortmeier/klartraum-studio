#include "studio/stable_diffusion.hpp"

#include <chrono>
#include <cmath>
#include <format>
#include <fstream>
#include <numbers>
#include <random>
#include <stdexcept>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/onnx/onnx_network.hpp"

namespace kstudio {

namespace {

using FloatTensor = klartraum::TensorElement<float>;

double elapsedMs(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

void log(const HostStepLog& hostStep, const std::string& step, std::chrono::steady_clock::time_point start) {
    if (hostStep) {
        hostStep(step, elapsedMs(start));
    }
}

const OnnxTensorDesc& modelInput(const OnnxModelInfo& info, const std::string& name, const char* model) {
    for (const auto& input : info.inputs) {
        if (input.name == name) {
            return input;
        }
    }
    throw std::runtime_error(std::format("the {} has no input '{}'", model, name));
}

std::shared_ptr<FloatTensor> floatOutput(const klartraum::OnnxNetwork& network, const std::string& name) {
    auto output = std::dynamic_pointer_cast<FloatTensor>(network.getOutputElement(name));
    if (!output) {
        throw std::runtime_error("the model's output '" + name + "' is not a float tensor");
    }
    return output;
}

std::map<std::string, float> timings(const klartraum::ComputeGraph& graph) {
    std::map<std::string, float> result;
    for (const auto& [label, ms] : graph.getProfilingResults()) {
        result.emplace(label, static_cast<float>(ms));
    }
    return result;
}

} // namespace

DdimSchedule ddimSchedule(uint32_t steps) {
    constexpr int kTrainingSteps = 1000;
    if (steps == 0 || steps > kTrainingSteps) {
        throw std::invalid_argument("DDIM needs between 1 and 1000 steps");
    }
    // Betas are spaced linearly in their square root ("scaled linear").
    std::vector<double> cumulative(kTrainingSteps);
    const double first = std::sqrt(0.00085);
    const double last = std::sqrt(0.012);
    double product = 1.0;
    for (int t = 0; t < kTrainingSteps; ++t) {
        const double root = first + (last - first) * t / (kTrainingSteps - 1);
        product *= 1.0 - root * root;
        cumulative[t] = product;
    }

    const int ratio = kTrainingSteps / static_cast<int>(steps);
    DdimSchedule schedule;
    for (int i = static_cast<int>(steps) - 1; i >= 0; --i) {
        const int t = i * ratio + 1;
        const int previous = t - ratio;
        schedule.timesteps.push_back(t);
        schedule.alphas.push_back(static_cast<float>(cumulative[t]));
        schedule.previousAlphas.push_back(static_cast<float>(cumulative[std::max(previous, 0)]));
    }
    return schedule;
}

std::vector<float> latentNoise(uint32_t seed, size_t count) {
    std::mt19937 engine(seed);
    // Uniform in (0, 1), so the logarithm below stays finite.
    auto uniform = [&] { return (static_cast<double>(engine()) + 0.5) / 4294967296.0; };
    std::vector<float> values(count);
    for (size_t i = 0; i < count; i += 2) {
        const double radius = std::sqrt(-2.0 * std::log(uniform()));
        const double angle = 2.0 * std::numbers::pi * uniform();
        values[i] = static_cast<float>(radius * std::cos(angle));
        if (i + 1 < count) {
            values[i + 1] = static_cast<float>(radius * std::sin(angle));
        }
    }
    return values;
}

std::vector<float> readFloats(const std::filesystem::path& path, size_t count) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error("cannot read " + path.string());
    }
    const auto bytes = static_cast<size_t>(in.tellg());
    if (bytes != count * sizeof(float)) {
        throw std::runtime_error(std::format("{} holds {} bytes; {} floats need {}", path.filename().string(), bytes,
                                             count, count * sizeof(float)));
    }
    in.seekg(0);
    std::vector<float> values(count);
    in.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(bytes));
    if (!in) {
        throw std::runtime_error("cannot read " + path.string());
    }
    return values;
}

void ddimStep(std::vector<float>& latents, const std::vector<float>& prediction, float guidanceScale, float alpha,
              float previousAlpha) {
    const size_t n = latents.size();
    if (prediction.size() != 2 * n) {
        throw std::invalid_argument("ddimStep: the prediction must hold two noise estimates of the latents' size");
    }
    const float sqrtAlpha = std::sqrt(alpha);
    const float sqrtBeta = std::sqrt(1.0f - alpha);
    const float sqrtPreviousAlpha = std::sqrt(previousAlpha);
    const float sqrtPreviousBeta = std::sqrt(1.0f - previousAlpha);
    for (size_t i = 0; i < n; ++i) {
        const float noise = prediction[i] + guidanceScale * (prediction[n + i] - prediction[i]);
        const float original = (latents[i] - sqrtBeta * noise) / sqrtAlpha;
        latents[i] = sqrtPreviousAlpha * original + sqrtPreviousBeta * noise;
    }
}

StagedResult sampleDdim(klartraum::VulkanContext& vc, const std::filesystem::path& unet, const OnnxModelInfo& info,
                        const HostTensor& latents, const HostTensor& embeddings, const DdimSamplerParams& params,
                        const std::string& name, const HostStepLog& hostStep) {
    const auto& sampleInput = modelInput(info, "sample", "UNet");
    const auto& timestepInput = modelInput(info, "timestep", "UNet");
    const auto& embeddingsInput = modelInput(info, "encoder_hidden_states", "UNet");
    if (info.outputs.empty()) {
        throw std::runtime_error("the UNet has no output");
    }
    // The UNet predicts the noise for both prompts at once: a batch of two.
    TensorShape expected = sampleInput.shape;
    if (expected.size() != 4 || expected[0] != 2) {
        throw std::runtime_error(std::format("the UNet's sample input is {}, not 2x4xHxW", shapeToString(expected)));
    }
    expected[0] = 1;
    if (latents.shape != expected) {
        throw std::runtime_error(std::format("the UNet expects {} latents but gets {}", shapeToString(expected),
                                             shapeToString(latents.shape)));
    }
    if (embeddings.shape != embeddingsInput.shape) {
        throw std::runtime_error(std::format("the UNet expects {} embeddings but gets {}",
                                             shapeToString(embeddingsInput.shape), shapeToString(embeddings.shape)));
    }
    const DdimSchedule schedule = ddimSchedule(params.steps);

    auto start = std::chrono::steady_clock::now();
    auto network = vc.create<klartraum::OnnxNetwork>(unet.string());
    network->setName(name);
    auto sample = vc.create<FloatTensor>(sampleInput.shape);
    sample->setName(name + " sample");
    auto timestep = vc.create<klartraum::TensorElement<int64_t>>(timestepInput.shape);
    timestep->setName(name + " timestep");
    auto conditioning = vc.create<FloatTensor>(embeddingsInput.shape);
    conditioning->setName(name + " embeddings");
    network->setInputTensor(sampleInput.name, sample);
    network->setInputTensor(timestepInput.name, timestep);
    network->setInputTensor(embeddingsInput.name, conditioning);
    auto output = floatOutput(*network, info.outputs[0].name);
    if (output->getDataElementCount() != 2 * latents.values.size()) {
        throw std::runtime_error("the UNet's output does not have the size of its sample input");
    }

    klartraum::ComputeGraph graph(vc, 1);
    graph.enableProfiling();
    graph.compileFrom(network);
    conditioning->getDataBuffer(0).memcopyFrom(embeddings.values);
    log(hostStep, name + ": loaded the UNet", start);

    std::vector<float> current = latents.values;
    const size_t n = current.size();
    std::vector<float> batch(2 * n);
    std::vector<float> prediction(2 * n);
    for (size_t step = 0; step < schedule.timesteps.size(); ++step) {
        start = std::chrono::steady_clock::now();
        std::copy(current.begin(), current.end(), batch.begin());
        std::copy(current.begin(), current.end(), batch.begin() + static_cast<std::ptrdiff_t>(n));
        sample->getDataBuffer(0).memcopyFrom(batch);
        timestep->getDataBuffer(0).memcopyFrom(std::vector<int64_t>{schedule.timesteps[step]});
        graph.submitAndWait(vc.getGraphicsQueue(), 0);
        output->getDataBuffer(0).memcopyTo(prediction);
        ddimStep(current, prediction, params.guidanceScale, schedule.alphas[step], schedule.previousAlphas[step]);
        log(hostStep,
            std::format("{}: step {}/{} (t={})", name, step + 1, schedule.timesteps.size(), schedule.timesteps[step]),
            start);
    }

    StagedResult result;
    result.output = HostTensor{latents.shape, std::move(current)};
    result.timings = timings(graph);
    result.root = network;
    return result;
}

} // namespace kstudio
