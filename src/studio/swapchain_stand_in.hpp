#pragma once

#include <vulkan/vulkan.h>

#include "klartraum/computegraph/computegraphelement.hpp"

namespace klartraum {
class VulkanContext;
} // namespace klartraum

namespace kstudio {

// Whether live graphs render into stand-in images instead of the swapchain's.
//
// MoltenVK backs each swapchain image with the Metal drawable of the frame
// it was acquired for, and a descriptor set holds the drawable that backed
// the image when the set was written. Graphs write their descriptor sets once,
// so a compute shader writing a swapchain image (Composite, Present's
// resample, the compute splatting backend) can write a drawable that is not
// presented. Copies and render passes find the frame's drawable. So on
// MoltenVK the Swapchain Target is an image per path, which Present copies
// into the acquired swapchain image (see SwapchainCopy).
bool needsSwapchainStandIn(klartraum::VulkanContext& vulkanContext);

// Copies the image of input 0 into the swapchain image of input 1 and leaves
// that ready for presenting. Both are ImageViewSrc elements (or elements
// passing one through a slot) of the same extent and format; input 0 is left
// in VK_IMAGE_LAYOUT_GENERAL, as targets start a frame.
class SwapchainCopy : public klartraum::ComputeGraphElement {
public:
    // `sourceLayout`: the layout input 0 is in; `presentLayout`: the one the
    // swapchain image is left in.
    SwapchainCopy(VkImageLayout sourceLayout, VkImageLayout presentLayout)
        : sourceLayout_(sourceLayout), presentLayout_(presentLayout) {}

    const char* getType() const override { return "SwapchainCopy"; }
    void checkInput(klartraum::ComputeGraphElementPtr input, int index = 0) override;
    void _record(VkCommandBuffer commandBuffer, uint32_t pathId) override;

private:
    VkImageLayout sourceLayout_;
    VkImageLayout presentLayout_;
};

} // namespace kstudio
