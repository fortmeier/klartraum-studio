#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace kstudio {

// A tensor shape in ONNX order (NCHW for images). Dynamic dimensions are
// fixed to 1, as klartraum::OnnxNetwork does.
using TensorShape = std::vector<uint32_t>;

std::string shapeToString(const TensorShape& shape);

struct OnnxTensorDesc {
    std::string name;
    TensorShape shape;
};

// What the studio needs to know about an ONNX model without building it.
struct OnnxModelInfo {
    std::vector<OnnxTensorDesc> inputs;   // graph inputs that are not initializers
    std::vector<OnnxTensorDesc> outputs;
    std::vector<std::string> opTypes;     // distinct, sorted
    std::vector<std::string> unsupportedOps;
};

// The ONNX operators klartraum::OnnxNetwork can execute.
bool isSupportedOnnxOp(const std::string& opType);

// Throws std::runtime_error if the file cannot be read or parsed.
OnnxModelInfo readOnnxModelInfo(const std::filesystem::path& path);

// Caches readOnnxModelInfo by path and file modification time.
class OnnxInfoCache {
public:
    // Returns nullptr and sets `error` if the model cannot be read.
    std::shared_ptr<const OnnxModelInfo> get(const std::filesystem::path& path, std::string* error = nullptr);

private:
    struct Entry {
        std::filesystem::file_time_type modified;
        std::shared_ptr<const OnnxModelInfo> info;
        std::string error;
    };
    std::map<std::string, Entry> entries_;
};

} // namespace kstudio
