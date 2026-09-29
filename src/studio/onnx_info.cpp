#include "studio/onnx_info.hpp"

#include <algorithm>
#include <fstream>
#include <set>
#include <stdexcept>

#include "onnx.pb.h"

namespace kstudio {

namespace {

TensorShape readShape(const onnx::ValueInfoProto& value) {
    TensorShape shape;
    if (!value.type().has_tensor_type() || !value.type().tensor_type().has_shape()) {
        return shape;
    }
    for (const auto& dim : value.type().tensor_type().shape().dim()) {
        shape.push_back(dim.has_dim_value() ? static_cast<uint32_t>(dim.dim_value()) : 1u);
    }
    return shape;
}

ElementType readElementType(const onnx::ValueInfoProto& value) {
    if (!value.type().has_tensor_type()) {
        return ElementType::Other;
    }
    switch (value.type().tensor_type().elem_type()) {
    case onnx::TensorProto::FLOAT: return ElementType::Float32;
    case onnx::TensorProto::INT64: return ElementType::Int64;
    default: return ElementType::Other;
    }
}

} // namespace

std::string shapeToString(const TensorShape& shape) {
    std::string text;
    for (size_t i = 0; i < shape.size(); ++i) {
        text += (i ? "x" : "") + std::to_string(shape[i]);
    }
    return text.empty() ? "scalar" : text;
}

std::string_view elementTypeName(ElementType type) {
    switch (type) {
    case ElementType::Float32: return "float32";
    case ElementType::Int64: return "int64";
    case ElementType::Other: return "other";
    }
    return "?";
}

std::string tensorTypeToString(const TensorType& type) {
    std::string text = shapeToString(type.shape);
    if (type.element != ElementType::Float32) {
        text += " " + std::string(elementTypeName(type.element));
    }
    return text;
}

bool isSupportedOnnxOp(const std::string& opType) {
    // See createTensorOperation in klartraum's onnx_operations.hpp.
    static const std::set<std::string> supported{
        "Add",     "Cast",     "Concat",  "Constant", "Conv",  "ConvTranspose", "Cos",       "Div",
        "Erf",     "Expand",   "Gather",  "Gemm",     "InstanceNormalization",   "LayerNormalization",
        "MatMul",  "Mul",      "Relu",    "Reshape",  "Resize", "Sigmoid",       "Sin",       "Slice",
        "Softmax", "Split",    "Sqrt",    "Sub",      "Transpose", "Unsqueeze"};
    return supported.contains(opType);
}

OnnxModelInfo readOnnxModelInfo(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot read " + path.string());
    }
    onnx::ModelProto model;
    if (!model.ParseFromIstream(&in)) {
        throw std::runtime_error("not a valid ONNX model: " + path.string());
    }
    const auto& graph = model.graph();

    std::set<std::string> initializers;
    for (const auto& initializer : graph.initializer()) {
        initializers.insert(initializer.name());
    }

    OnnxModelInfo info;
    for (const auto& input : graph.input()) {
        if (!initializers.contains(input.name())) {
            info.inputs.push_back({input.name(), readShape(input), readElementType(input)});
        }
    }
    for (const auto& output : graph.output()) {
        info.outputs.push_back({output.name(), readShape(output), readElementType(output)});
    }
    std::set<std::string> ops;
    for (const auto& node : graph.node()) {
        ops.insert(node.op_type());
    }
    info.opTypes.assign(ops.begin(), ops.end());
    for (const auto& op : info.opTypes) {
        if (!isSupportedOnnxOp(op)) {
            info.unsupportedOps.push_back(op);
        }
    }
    return info;
}

std::shared_ptr<const OnnxModelInfo> OnnxInfoCache::get(const std::filesystem::path& path, std::string* error) {
    std::error_code ec;
    const auto modified = std::filesystem::last_write_time(path, ec);
    if (ec) {
        if (error) {
            *error = "cannot read " + path.string();
        }
        return nullptr;
    }
    const std::string key = path.string();
    auto it = entries_.find(key);
    if (it == entries_.end() || it->second.modified != modified) {
        Entry entry;
        entry.modified = modified;
        try {
            entry.info = std::make_shared<OnnxModelInfo>(readOnnxModelInfo(path));
        } catch (const std::exception& e) {
            entry.error = e.what();
        }
        it = entries_.insert_or_assign(key, std::move(entry)).first;
    }
    if (error) {
        *error = it->second.error;
    }
    return it->second.info;
}

} // namespace kstudio
