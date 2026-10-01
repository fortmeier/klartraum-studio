#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

#include "studio/graph_model.hpp"
#include "studio/onnx_info.hpp"
#include "studio/stable_diffusion.hpp"

namespace klartraum {
class SinglePathImage;
class TensorElementInterface;
class VulkanContext;
} // namespace klartraum

namespace kstudio {

// A result of a Run that the live graph reads: one tensor or one image that
// every path of the live graph reads (klartraum::TensorElementSinglePath,
// klartraum::SinglePathImage).
struct RetainedValue {
    TensorType type;  // for a tensor
    std::shared_ptr<klartraum::TensorElementInterface> tensor;
    std::shared_ptr<klartraum::SinglePathImage> image;
};

// The results of run-only nodes the live graph reads (see LivePlan), kept
// from one Run to the next. Run stores them; the live graph is built with
// them and reads them in every frame. Storing copies into the value's storage
// when its type still fits, so graphs built with it see the new result;
// otherwise new storage is made, and generation() changes: graphs built with
// the old storage must be rebuilt. Store only while no graph that reads the
// results executes.
class RetainedResults {
public:
    explicit RetainedResults(klartraum::VulkanContext& vulkanContext) : vc_(vulkanContext) {}

    // A float32 or int64 tensor, copied on the GPU.
    void storeTensor(OutputPin pin, klartraum::TensorElementInterface& tensor, const std::string& name);
    // Float values computed on the CPU, uploaded.
    void storeHost(OutputPin pin, const HostTensor& values, const std::string& name);
    // An image in the swapchain's format, left in `layout`.
    void storeImage(OutputPin pin, VkImage image, VkImageLayout layout, VkExtent2D extent, const std::string& name);

    const RetainedValue* find(OutputPin pin) const;
    // Whether every one of `pins` has a result.
    bool has(const std::vector<OutputPin>& pins) const;
    uint64_t generation() const { return generation_; }
    void clear();

private:
    klartraum::VulkanContext& vc_;
    std::map<OutputPin, RetainedValue> values_;
    uint64_t generation_ = 0;
};

} // namespace kstudio
