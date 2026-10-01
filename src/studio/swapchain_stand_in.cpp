#include "studio/swapchain_stand_in.hpp"

#include <stdexcept>

#include "klartraum/computegraph/imageviewsrc.hpp"
#include "klartraum/vulkan_context.hpp"

namespace kstudio {

bool needsSwapchainStandIn(klartraum::VulkanContext& vulkanContext) {
    if (!vulkanContext.hasSurface()) {
        return false;
    }
    VkPhysicalDeviceDriverProperties driver{};
    driver.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
    VkPhysicalDeviceProperties2 properties{};
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties.pNext = &driver;
    vkGetPhysicalDeviceProperties2(vulkanContext.getPhysicalDevice(), &properties);
    return driver.driverID == VK_DRIVER_ID_MOLTENVK;
}

void SwapchainCopy::checkInput(klartraum::ComputeGraphElementPtr input, int index) {
    if (index > 1) {
        throw std::runtime_error("SwapchainCopy has two inputs");
    }
    if (!std::dynamic_pointer_cast<klartraum::ImageViewSrc>(input)) {
        throw std::runtime_error("SwapchainCopy: input is not an ImageViewSrc");
    }
}

void SwapchainCopy::_record(VkCommandBuffer commandBuffer, uint32_t pathId) {
    ComputeGraphElement::_record(commandBuffer, pathId);
    auto& source = dynamic_cast<klartraum::ImageViewSrc&>(*getInputElement(0));
    auto& target = dynamic_cast<klartraum::ImageViewSrc&>(*getInputElement(1));
    const VkImage sourceImage = source.getImage(pathId);
    const VkImage targetImage = target.getImage(pathId);
    const VkExtent2D extent = target.getImageExtent(pathId);

    auto barrier = [&](VkImage image, VkImageLayout from, VkImageLayout to, VkAccessFlags srcAccess,
                       VkAccessFlags dstAccess, VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage) {
        VkImageMemoryBarrier b{};
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.oldLayout = from;
        b.newLayout = to;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = image;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        b.srcAccessMask = srcAccess;
        b.dstAccessMask = dstAccess;
        vkCmdPipelineBarrier(commandBuffer, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &b);
    };
    // Whatever wrote the stand-in, its writes are visible to the copy.
    barrier(sourceImage, sourceLayout_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_MEMORY_WRITE_BIT,
            VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    // The swapchain image's previous contents are overwritten.
    barrier(targetImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    VkImageCopy region{};
    region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.extent = {extent.width, extent.height, 1};
    vkCmdCopyImage(commandBuffer, sourceImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, targetImage,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    barrier(targetImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, presentLayout_, VK_ACCESS_TRANSFER_WRITE_BIT, 0,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
    // Back to where targets start a frame, for one that is not cleared.
    barrier(sourceImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_READ_BIT, 0,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
}

} // namespace kstudio
