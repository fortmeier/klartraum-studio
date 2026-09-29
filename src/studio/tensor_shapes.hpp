#pragma once

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "studio/graph_model.hpp"
#include "studio/onnx_info.hpp"

namespace kstudio {

// Looks up an ONNX model's inputs and outputs by the path stored in an ONNX
// Model node. Returns nullptr and sets `error` if it cannot be read.
using OnnxInfoProvider =
    std::function<std::shared_ptr<const OnnxModelInfo>(const std::string& path, std::string& error)>;

// A tensor that can be shown or saved as an image: 1x1xHxW or 1x3xHxW.
bool isImageShape(const TensorShape& shape);

// An output pin: (node id, output slot).
using OutputPin = std::pair<int, int>;

struct ShapeInference {
    std::map<OutputPin, TensorType> types;  // what each tensor output carries, where known
    std::vector<Diagnostic> diagnostics;
};

// Propagates tensor types (shape and element type) through the graph and
// reports what does not fit, on the consuming node:
//  - Image File and Image to Tensor produce 1x3xHxW float tensors (unknown for
//    swapchain renderings); Latent Noise its latents; the Prompt 2x77 int64
//    token ids and attention mask;
//  - an ONNX model maps its declared inputs to its declared outputs; its
//    pins must match the model's inputs and outputs, and each input's type
//    must be the model's;
//  - layers broadcast (Add, Subtract, Multiply, Divide) or keep their input's
//    shape, and take float tensors;
//  - Stable Diffusion nodes check their inputs against their models;
//  - sinks need float images (1x1xHxW or 1x3xHxW), Tensor to Image 1x3xHxW;
//  - unreadable models and operators klartraum cannot execute.
// Unconnected inputs are skipped; Graph::validate reports those.
ShapeInference inferTensorShapes(const Graph& graph, const OnnxInfoProvider& onnxInfo);

} // namespace kstudio
