#include "studio/tensor_shapes.hpp"

#include <algorithm>
#include <format>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string_view>

#include "klartraum/layers/layers.hpp"

namespace kstudio {

bool isImageShape(const TensorShape& shape) {
    return shape.size() == 4 && shape[0] == 1 && (shape[1] == 1 || shape[1] == 3) && shape[2] > 0 && shape[3] > 0;
}

ShapeInference inferTensorShapes(const Graph& graph, const OnnxInfoProvider& onnxInfo) {
    ShapeInference result;
    auto report = [&](int node, std::string message) {
        result.diagnostics.push_back(Diagnostic{Severity::Error, node, std::move(message)});
    };
    // The type of the tensor feeding input `slot`, if known.
    auto inputType = [&](int node, int slot = 0) -> const TensorType* {
        const Link* link = graph.inputLink(node, slot);
        if (!link) {
            return nullptr;
        }
        auto it = result.types.find({link->fromNode, link->fromSlot});
        return it == result.types.end() ? nullptr : &it->second;
    };
    auto inputShape = [&](int node, int slot = 0) -> const TensorShape* {
        const TensorType* type = inputType(node, slot);
        return type ? &type->shape : nullptr;
    };
    // Reports a float input that is not float; returns whether it is.
    auto requireFloat = [&](int node, int slot, std::string_view what) {
        const TensorType* type = inputType(node, slot);
        if (type && type->element != ElementType::Float32) {
            report(node, std::format("{} must be a float32 tensor, not {}.", what, elementTypeName(type->element)));
            return false;
        }
        return true;
    };
    // The 1x3xHxW tensor an image converts to, if its size is known: a
    // Gaussian Splatting node renders at its target's size (the swapchain's
    // is only known when the graph is built), a Clear Image, a Composite or a
    // Draw Basics at its input's,
    // a Tensor to Image node at its input tensor's, a Resample node at its own.
    std::function<std::optional<TensorShape>(const Node*)> imageShape =
        [&](const Node* source) -> std::optional<TensorShape> {
        if (!source) {
            return std::nullopt;
        }
        switch (source->kind) {
        case NodeKind::OffscreenTarget: {
            const auto& p = source->as<OffscreenTargetParams>();
            return TensorShape{1, 3, p.height, p.width};
        }
        case NodeKind::GaussianSplatting:
            return imageShape(graph.inputNode(source->id, 2));
        case NodeKind::ClearImage:
        case NodeKind::Composite:
        case NodeKind::DrawBasics:
            return imageShape(graph.inputNode(source->id, 0));
        case NodeKind::Resample: {
            const auto& p = source->as<ResampleParams>();
            return TensorShape{1, 3, p.height, p.width};
        }
        case NodeKind::TensorToImage:
            if (const TensorShape* in = inputShape(source->id); in && isImageShape(*in) && (*in)[1] == 3) {
                return *in;
            }
            return std::nullopt;
        default:
            return std::nullopt;
        }
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
        auto output = [&](int slot, TensorType type) { result.types[{id, slot}] = std::move(type); };

        if (isBinaryLayer(node.kind)) {
            const TensorShape* a = inputShape(id, 0);
            const bool floats = requireFloat(id, 0, "A") && requireFloat(id, 1, "B");
            // Without B, the node's b is a one-element tensor.
            const TensorShape one{1};
            const TensorShape* b = graph.inputLink(id, 1) ? inputShape(id, 1) : &one;
            if (a && b && floats) {
                try {
                    output(0, {klartraum::layers::broadcastShape(*a, *b), ElementType::Float32});
                } catch (const std::runtime_error&) {
                    report(id, std::format("{} and {} cannot be broadcast together.", shapeToString(*a),
                                           shapeToString(*b)));
                }
            }
            continue;
        }
        if (isUnaryLayer(node.kind)) {
            if (const TensorShape* in = inputShape(id); in && requireFloat(id, 0, "The input")) {
                output(0, {*in, ElementType::Float32});
            }
            continue;
        }

        switch (node.kind) {
        case NodeKind::ImageFile: {
            const auto& p = node.as<ImageFileParams>();
            output(0, {{1, 3, p.height, p.width}});
            break;
        }
        case NodeKind::ImageToTensor:
            if (auto shape = imageShape(graph.inputNode(id, 0))) {
                output(0, {*shape});
            }
            break;
        case NodeKind::TensorToImage:
            // klartraum's tensor_to_image shader reads three float planes.
            if (const TensorShape* in = inputShape(id); in && !(isImageShape(*in) && (*in)[1] == 3)) {
                report(id, std::format("A {} tensor cannot be converted; expected 1x3xHxW.", shapeToString(*in)));
            }
            requireFloat(id, 0, "The tensor");
            break;
        case NodeKind::OnnxModel: {
            const auto& p = node.as<OnnxModelParams>();
            const auto info = model(id, p.path);
            if (!info) {
                break;
            }
            std::vector<std::string> inputs, outputs;
            for (const auto& input : info->inputs) {
                inputs.push_back(input.name);
            }
            for (const auto& out : info->outputs) {
                outputs.push_back(out.name);
            }
            if (inputs.size() > kMaxPinsPerDirection || outputs.size() > kMaxPinsPerDirection) {
                report(id, std::format("Models with more than {} inputs or outputs are not supported.",
                                       kMaxPinsPerDirection));
                break;
            }
            if (inputs != p.inputs || outputs != p.outputs) {
                report(id, "The node's pins do not match the model's inputs and outputs.");
                break;
            }
            for (size_t i = 0; i < info->inputs.size(); ++i) {
                const TensorType* in = inputType(id, static_cast<int>(i));
                if (in && *in != info->inputs[i].type()) {
                    report(id, std::format("Input '{}' expects {} but gets {}.", info->inputs[i].name,
                                           tensorTypeToString(info->inputs[i].type()), tensorTypeToString(*in)));
                }
            }
            for (size_t i = 0; i < info->outputs.size(); ++i) {
                output(static_cast<int>(i), info->outputs[i].type());
            }
            break;
        }
        case NodeKind::Prompt:
            output(0, {{2, 77}, ElementType::Int64});
            output(1, {{2, 77}, ElementType::Int64});
            break;
        case NodeKind::LatentNoise: {
            const auto& p = node.as<LatentNoiseParams>();
            output(0, {{1, 4, p.height / 8, p.width / 8}});
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
            requireFloat(id, 0, "The latents");
            requireFloat(id, 1, "The embeddings");
            output(0, {latents});
            break;
        }
        case NodeKind::Preview:
        case NodeKind::ImageFileWriter:
            if (const TensorShape* in = inputShape(id); in && !isImageShape(*in)) {
                report(id, std::format("A {} tensor is not an image; expected 1x1xHxW or 1x3xHxW.", shapeToString(*in)));
            }
            requireFloat(id, 0, "The tensor");
            break;
        default:
            break;
        }
    }
    return result;
}

} // namespace kstudio
