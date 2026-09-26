#include "studio/tensor_shapes.hpp"

#include <format>
#include <optional>

namespace kstudio {

bool isImageShape(const TensorShape& shape) {
    return shape.size() == 4 && shape[0] == 1 && (shape[1] == 1 || shape[1] == 3) && shape[2] > 0 && shape[3] > 0;
}

ShapeInference inferTensorShapes(const Graph& graph, const OnnxInfoProvider& onnxInfo) {
    ShapeInference result;
    auto report = [&](int node, std::string message) {
        result.diagnostics.push_back(Diagnostic{Severity::Error, node, std::move(message)});
    };
    auto inputShape = [&](int node) -> const TensorShape* {
        const Link* link = graph.inputLink(node, 0);
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
            const auto& path = node.as<OnnxModelParams>().path;
            if (path.empty() || !onnxInfo) {
                break;
            }
            std::string error;
            const auto info = onnxInfo(path, error);
            if (!info) {
                report(id, error.empty() ? "Cannot read the ONNX model." : error);
                break;
            }
            if (!info->unsupportedOps.empty()) {
                std::string ops;
                for (const auto& op : info->unsupportedOps) {
                    ops += (ops.empty() ? "" : ", ") + op;
                }
                report(id, "klartraum cannot execute these ONNX operators: " + ops + ".");
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
