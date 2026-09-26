#pragma once

#include <cstdint>

#include <imgui.h>
#include <vulkan/vulkan.h>

#include "studio/image_io.hpp"

namespace klartraum {
class VulkanContext;
}

namespace kstudio {

// A GPU copy of an image that ImGui can draw (ImGui::Image). Needs the ImGui
// Vulkan backend, i.e. an ImGuiOverlay, to be initialized.
class PreviewTexture {
public:
    PreviewTexture(klartraum::VulkanContext& vulkanContext, const ImageRGBA8& image);
    ~PreviewTexture();

    PreviewTexture(const PreviewTexture&) = delete;
    PreviewTexture& operator=(const PreviewTexture&) = delete;

    ImTextureID id() const { return reinterpret_cast<ImTextureID>(descriptorSet_); }
    uint32_t width() const { return width_; }
    uint32_t height() const { return height_; }

private:
    klartraum::VulkanContext& vulkanContext_;
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    VkImage image_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    VkImageView view_ = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet_ = VK_NULL_HANDLE;
};

} // namespace kstudio
