#include "studio/tensor_shapes.hpp"

#include <algorithm>
#include <format>
#include <optional>
#include <string_view>

namespace kstudio {

bool isImageShape(const TensorShape& shape) {
    return shape.size() == 4 && shape[0] == 1 && (shape[1] == 1 || shape[1] == 3) && shape[2] > 0 && shape[3] > 0;
}

ShapeInference inferTensorShapes(const Graph& graph, const OnnxInfoProvider& onnxInfo) {
    ShapeInference result;
    auto report = [&](int node, std::string message) {
        result.diagnostics.push_back(Diagnostic{Severity::Error, node, std::move(message)});
    };
    auto inputShape = [&](int node, int slot = 0) -> const TensorShape* {
        const Link* link = graph.inputLink(node, slot);
        if (!link) {
            return nullptr;
        }
        auto it = result.shapes.find(link->fromNode);
        return it == result.shapes.end() ? nullptr : &it->second;
    };
    // The 1x3xHxW tensor an image converts to, if its size is known: a
    // Gaussian Splatting node renders at its Offscreen Target's size (the
    // swapchain's is only known when the graph is built), a Tensor to Image
    // node at its input tensor's, a Resample node at its own.
    auto imageShape = [&](const Node* source) -> std::optional<TensorShape> {
        if (source && source->kind == NodeKind::GaussianSplatting) {
            const Node* target = graph.inputNode(source->id, 2);
            if (target && target->kind == NodeKind::OffscreenTarget) {
                const auto& p = target->as<OffscreenTargetParams>();
                return TensorShape{1, 3, p.height, p.width};
            }
        } else if (source && source->kind == NodeKind::Resample) {
            const auto& p = source->as<ResampleParams>();
            return TensorShape{1, 3, p.height, p.width};
        } else if (source && source->kind == NodeKind::TensorToImage) {
            if (const TensorShape* in = inputShape(source->id); in && isImageShape(*in) && (*in)[1] == 3) {
                return *in;
            }
        }
        return std::nullopt;
    };

    // The model a node names; reports it if it cannot be read or uses
    // operators klartraum cannot execute.
    auto model = [&](int node, const std::string& path) -> std::shared_ptr<const OnnxModelInfo> {
        if (path.empty() || !onnxInfo) {
            return nullptr;
        }
        std::string error;
        auto info = onnxInfo(path, error);
        if (!info) {
            report(node, error.empty() ? "Cannot read the ONNX model." : error);
            return nullptr;
        }
        if (!info->unsupportedOps.empty()) {
            std::string ops;
            for (const auto& op : info->unsupportedOps) {
                ops += (ops.empty() ? "" : ", ") + op;
            }
            report(node, "klartraum cannot execute these ONNX operators: " + ops + ".");
        }
        return info;
    };
    auto modelInput = [](const OnnxModelInfo& info, std::string_view name) -> const OnnxTensorDesc* {
        for (const auto& input : info.inputs) {
            if (input.name == name) {
                return &input;
            }
        }
        return nullptr;
    };

    for (int id : graph.topologicalOrder()) {
        const Node& node = *graph.findNode(id);
        switch (node.kind) {
        case NodeKind::ImageFile: {
            const auto& p = node.as<ImageFileParams>();
            result.shapes[id] = {1, 3, p.height, p.width};
            break;
        }
        case NodeKind::ImageToTensor:
            if (auto shape = imageShape(graph.inputNode(id, 0))) {
                result.shapes[id] = *shape;
            }
            break;
        case NodeKind::TensorToImage:
            // klartraum's tensor_to_image shader reads three planes.
            if (const TensorShape* in = inputShape(id); in && !(isImageShape(*in) && (*in)[1] == 3)) {
                report(id, std::format("A {} tensor cannot be converted; expected 1x3xHxW.", shapeToString(*in)));
            }
            break;
        case NodeKind::OnnxModel: {
            const auto info = model(id, node.as<OnnxModelParams>().path);
            if (!info) {
                break;
            }
            if (info->inputs.size() != 1 || info->outputs.empty()) {
                report(id, std::format("Only models with one input are supported (this one has {} inputs, {} outputs).",
                                       info->inputs.size(), info->outputs.size()));
                break;
            }
            if (const TensorShape* in = inputShape(id); in && *in != info->inputs[0].shape) {
                report(id, std::format("The model expects a {} tensor but gets {}.",
                                       shapeToString(info->inputs[0].shape), shapeToString(*in)));
            }
            result.shapes[id] = info->outputs[0].shape;
            break;
        }
        case NodeKind::LatentNoise: {
            const auto& p = node.as<LatentNoiseParams>();
            result.shapes[id] = {1, 4, p.height / 8, p.width / 8};
            break;
        }
        case NodeKind::TextEncoder: {
            const auto info = model(id, node.as<TextEncoderParams>().path);
            if (!info) {
                break;
            }
            // The Prompt node gives 2x77 token ids and an attention mask.
            const TensorShape tokens{2, 77};
            const OnnxTensorDesc* ids = modelInput(*info, "input_ids");
            if (!ids || info->outputs.empty()) {
                report(id, "Not a CLIP text encoder: it needs an input_ids input and an output.");
                break;
            }
            for (const auto& input : info->inputs) {
                if (input.name != "input_ids" && input.name != "attention_mask") {
                    report(id, "Unexpected text encoder input '" + input.name + "'.");
                } else if (input.shape != tokens) {
                    report(id, std::format("The model expects {} {} but the Prompt gives 2x77 (negative and positive "
                                           "prompt).",
                                           shapeToString(input.shape), input.name));
                }
            }
            const auto& outputs = info->outputs;
            auto embeddings = std::find_if(outputs.begin(), outputs.end(),
                                           [](const OnnxTensorDesc& o) { return o.name == "last_hidden_state"; });
            result.shapes[id] = embeddings != outputs.end() ? embeddings->shape : outputs[0].shape;
            break;
        }
        case NodeKind::DdimSampler: {
            const auto info = model(id, node.as<DdimSamplerParams>().path);
            if (!info) {
                break;
            }
            const OnnxTensorDesc* sample = modelInput(*info, "sample");
            const OnnxTensorDesc* embeddings = modelInput(*info, "encoder_hidden_states");
            if (!sample || !embeddings || !modelInput(*info, "timestep") || info->outputs.empty()) {
                report(id, "Not an SD 1.5 UNet: it needs the inputs sample, timestep and encoder_hidden_states "
                           "(see klartraum's scripts/sd15_onnx/export_denoiser.py).");
                break;
            }
            // The UNet sees the latents twice, for the negative and the
            // positive prompt.
            TensorShape latents = sample->shape;
            if (!latents.empty()) {
                latents[0] = 1;
            }
            if (const TensorShape* in = inputShape(id, 0); in && *in != latents) {
                report(id, std::format("The UNet expects {} latents but gets {}.", shapeToString(latents),
                                       shapeToString(*in)));
            }
            if (const TensorShape* in = inputShape(id, 1); in && *in != embeddings->shape) {
                report(id, std::format("The UNet expects {} embeddings but gets {}.", shapeToString(embeddings->shape),
                                       shapeToString(*in)));
            }
            result.shapes[id] = latents;
            break;
        }
        case NodeKind::VaeDecoder: {
            const auto info = model(id, node.as<VaeDecoderParams>().path);
            if (!info) {
                break;
            }
            if (info->inputs.size() != 1 || info->outputs.empty()) {
                report(id, "A VAE decoder has one input and an output.");
                break;
            }
            if (const TensorShape* in = inputShape(id); in && *in != info->inputs[0].shape) {
                report(id, std::format("The decoder expects {} latents but gets {}.",
                                       shapeToString(info->inputs[0].shape), shapeToString(*in)));
            }
            result.shapes[id] = info->outputs[0].shape;
            break;
        }
        case NodeKind::Preview:
        case NodeKind::ImageFileWriter:
            if (const TensorShape* in = inputShape(id); in && !isImageShape(*in)) {
                report(id, std::format("A {} tensor is not an image; expected 1x1xHxW or 1x3xHxW.", shapeToString(*in)));
            }
            break;
        default:
            break;
        }
    }
    return result;
}

} // namespace kstudio
