#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "studio/graph_model.hpp"
#include "studio/onnx_info.hpp"

namespace klartraum {
class ComputeGraphElement;
class VulkanContext;
} // namespace klartraum

namespace kstudio {

// Stable Diffusion 1.5 on klartraum, with the models klartraum's
// scripts/sd15_onnx/export_denoiser.py exports; the same pipeline as
// klartraum's examples/sd15_denoiser_example.cpp.

// Latents are multiplied by this before the VAE encoder's output is used and
// divided by it before decoding.
constexpr float kVaeScalingFactor = 0.18215f;

// A tensor in CPU memory, e.g. the result of a staged node.
struct HostTensor {
    TensorShape shape;
    std::vector<float> values;
};

// The DDIM schedule for `steps` denoising steps, as diffusers' DDIMScheduler
// derives it from SD 1.5's scheduler config: 1000 training steps with
// "scaled linear" betas from 0.00085 to 0.012, "leading" timestep spacing
// with offset 1, and the first training step's alpha after the last step.
struct DdimSchedule {
    std::vector<int64_t> timesteps;     // descending
    std::vector<float> alphas;          // cumulative alpha at each timestep
    std::vector<float> previousAlphas;  // cumulative alpha of the timestep each step leads to
};
// Throws std::invalid_argument unless 1 <= steps <= 1000.
DdimSchedule ddimSchedule(uint32_t steps);

// `count` samples of standard normal noise, the same for a seed on every
// platform (std::mt19937 and the Box-Muller transform).
std::vector<float> latentNoise(uint32_t seed, size_t count);

// Reads `count` raw float32 values; throws std::runtime_error if the file
// holds a different number.
std::vector<float> readFloats(const std::filesystem::path& path, size_t count);

// One DDIM step (eta 0) with classifier-free guidance, in place.
// `prediction` holds the UNet's predicted noise for the negative prompt,
// then for the positive prompt, each the size of `latents`.
void ddimStep(std::vector<float>& latents, const std::vector<float>& prediction, float guidanceScale, float alpha,
              float previousAlpha);

// What a staged node computed, and the klartraum graph it ran for it (kept
// alive for the compiled-graph view; release it after introspecting).
struct StagedResult {
    HostTensor output;
    std::shared_ptr<klartraum::ComputeGraphElement> root;
    std::map<std::string, float> timings;  // GPU milliseconds per element label
};

// Told about CPU work and each denoising step, with its duration; may be empty.
using HostStepLog = std::function<void(const std::string& step, double milliseconds)>;

// Denoises 1x4xhxw `latents` with the UNet at `unet` (inputs sample,
// timestep and encoder_hidden_states, as `info` describes them), conditioned
// on the 2x77x768 `embeddings` of the negative and the positive prompt. The
// UNet is compiled once and submitted once per step; guidance and the DDIM
// update run on the CPU. Throws std::runtime_error if the shapes do not fit
// the model.
StagedResult sampleDdim(klartraum::VulkanContext& vulkanContext, const std::filesystem::path& unet,
                        const OnnxModelInfo& info, const HostTensor& latents, const HostTensor& embeddings,
                        const DdimSamplerParams& params, const std::string& name, const HostStepLog& hostStep);

// Decodes `latents` with the VAE decoder at `decoder` into a 1x3xHxW image
// tensor with values in [0, 1]: the latents are divided by
// kVaeScalingFactor, and the decoder's [-1, 1] output is mapped to [0, 1].
StagedResult decodeLatents(klartraum::VulkanContext& vulkanContext, const std::filesystem::path& decoder,
                           const OnnxModelInfo& info, const HostTensor& latents, const std::string& name,
                           const HostStepLog& hostStep);

} // namespace kstudio
