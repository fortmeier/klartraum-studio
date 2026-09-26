#pragma once

#include <functional>
#include <map>
#include <memory>
#include <string>
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

struct ShapeInference {
    std::map<int, TensorShape> shapes;  // output shape of each node producing a tensor
    std::vector<Diagnostic> diagnostics;
};

// Propagates tensor shapes through the graph: Image File and Image to Tensor
// produce 1x3xHxW, ONNX models map their declared input shape to their output
// shape. Reports mismatched shapes, unreadable or unsupported models, and
// sinks fed with tensors that are not images. Nodes with unconnected inputs
// are skipped; Graph::validate reports those.
ShapeInference inferTensorShapes(const Graph& graph, const OnnxInfoProvider& onnxInfo);

} // namespace kstudio
