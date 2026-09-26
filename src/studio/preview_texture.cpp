#include "studio/preview_texture.hpp"

#include <cstring>
#include <stdexcept>

#include <imgui_impl_vulkan.h>

#include "klartraum/vulkan_context.hpp"

namespace kstudio {

namespace {

void transition(VkCommandBuffer cmd, VkImage image, VkImageLayout from, VkImageLayout to, VkAccessFlags srcAccess,
                VkAccessFlags dstAccess, VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage) {
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = from;
    barrier.newLayout = to;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    barrier.srcAccessMask = srcAccess;
    barrier.dstAccessMask = dstAccess;
    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

} // namespace

PreviewTexture::PreviewTexture(klartraum::VulkanContext& vulkanContext, const ImageRGBA8& image)
    : vulkanContext_(vulkanContext), width_(image.width), height_(image.height) {
    if (image.empty()) {
        throw std::invalid_argument("cannot preview an empty image");
    }
    VkDevice device = vulkanContext_.getDevice();

    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    imageInfo.extent = {width_, height_, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(device, &imageInfo, nullptr, &image_) != VK_SUCCESS) {
        throw std::runtime_error("failed to create preview image");
    }
    VkMemoryRequirements requirements;
    vkGetImageMemoryRequirements(device, image_, &requirements);
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = requirements.size;
    allocInfo.memoryTypeIndex =
        vulkanContext_.findMemoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (vkAllocateMemory(device, &allocInfo, nullptr, &memory_) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate preview image memory");
    }
    vkBindImageMemory(device, image_, memory_, 0);

    // Upload through a staging buffer.
    VkBuffer staging;
    VkDeviceMemory stagingMemory;
    const VkDeviceSize bytes = image.pixels.size();
    vulkanContext_.createBuffer(bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging,
                                stagingMemory);
    void* mapped;
    vkMapMemory(device, stagingMemory, 0, bytes, 0, &mapped);
    std::memcpy(mapped, image.pixels.data(), bytes);
    vkUnmapMemory(device, stagingMemory);

    VkCommandBufferAllocateInfo cmdInfo{};
    cmdInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdInfo.commandPool = vulkanContext_.getCommandPool();
    cmdInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdInfo.commandBufferCount = 1;
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(device, &cmdInfo, &cmd);
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin);
    transition(cmd, image_, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
               VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {width_, height_, 1};
    vkCmdCopyBufferToImage(cmd, staging, image_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    transition(cmd, image_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
               VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
               VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
    vkEndCommandBuffer(cmd);
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    vkQueueSubmit(vulkanContext_.getGraphicsQueue(), 1, &submit, VK_NULL_HANDLE);
    vkQueueWaitIdle(vulkanContext_.getGraphicsQueue());
    vkFreeCommandBuffers(device, vulkanContext_.getCommandPool(), 1, &cmd);
    vkDestroyBuffer(device, staging, nullptr);
    vkFreeMemory(device, stagingMemory, nullptr);

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image_;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (vkCreateImageView(device, &viewInfo, nullptr, &view_) != VK_SUCCESS) {
        throw std::runtime_error("failed to create preview image view");
    }
    descriptorSet_ = ImGui_ImplVulkan_AddTexture(view_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

PreviewTexture::~PreviewTexture() {
    VkDevice device = vulkanContext_.getDevice();
    // A frame in flight may still sample the texture.
    vkDeviceWaitIdle(device);
    if (descriptorSet_ != VK_NULL_HANDLE) {
        ImGui_ImplVulkan_RemoveTexture(descriptorSet_);
    }
    vkDestroyImageView(device, view_, nullptr);
    vkDestroyImage(device, image_, nullptr);
    vkFreeMemory(device, memory_, nullptr);
}

} // namespace kstudio
