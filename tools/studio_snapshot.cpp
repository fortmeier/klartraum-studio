// Renders the studio UI headlessly and writes the frame to a BMP file, so the
// UI and the rendered scene can be checked without a display.
//
//   klartraum_studio_snapshot --out shot.bmp [--backend compute|raster]
//                             [--tab authoring|compiled] [--select KIND]
//                             [--hide-buffers] [--frames N] [--then-backend B]
//                             [--open-meta] [graph.ktgraph.json]
//
// --then-backend switches the Gaussian Splatting node's backend halfway, which
// exercises recompiling while a graph is running. --example picks the start
// graph (see klartraum_studio --help); --run presses Run after the first
// frame and fails if the run fails. --open-meta opens the graph's first meta
// node after the first frame.
//
// KIND is a node kind name such as gaussian_splatting or scene. The headless
// swapchain is 512x512 pixels; the UI is laid out at kLogicalSize and scaled
// down to fit.

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <imgui.h>

#include "klartraum/headless_frontend.hpp"
#include "klartraum/imgui_overlay.hpp"
#include "klartraum/vulkan_helpers.hpp"

#include "studio/studio_app.hpp"

namespace {

constexpr float kLogicalSize = 1100.0f;

std::vector<uint8_t> readSwapchainImage(klartraum::VulkanContext& vc, uint32_t index) {
    const VkExtent2D ext = vc.getSwapChainExtent();
    const VkDeviceSize bytes = static_cast<VkDeviceSize>(ext.width) * ext.height * 4;
    VkBuffer buffer;
    VkDeviceMemory memory;
    vc.createBuffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, buffer, memory);

    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = vc.getCommandPool();
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(vc.getDevice(), &ai, &cmd);
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    VkBufferImageCopy region{};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = {ext.width, ext.height, 1};
    // Headless frames end in VK_IMAGE_LAYOUT_GENERAL.
    vkCmdCopyImageToBuffer(cmd, vc.getSwapChainImage(index), VK_IMAGE_LAYOUT_GENERAL, buffer, 1, &region);
    vkEndCommandBuffer(cmd);
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vkQueueSubmit(vc.getGraphicsQueue(), 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(vc.getGraphicsQueue());
    vkFreeCommandBuffers(vc.getDevice(), vc.getCommandPool(), 1, &cmd);

    void* data;
    vkMapMemory(vc.getDevice(), memory, 0, bytes, 0, &data);
    std::vector<uint8_t> pixels(static_cast<const uint8_t*>(data), static_cast<const uint8_t*>(data) + bytes);
    vkUnmapMemory(vc.getDevice(), memory);
    vkFreeMemory(vc.getDevice(), memory, nullptr);
    vkDestroyBuffer(vc.getDevice(), buffer, nullptr);
    return pixels;
}

// 32-bit BGRA, top-down.
bool writeBmp(const std::string& path, const std::vector<uint8_t>& bgra, uint32_t width, uint32_t height) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return false;
    }
    auto u16 = [&](uint16_t v) { out.write(reinterpret_cast<const char*>(&v), 2); };
    auto u32 = [&](uint32_t v) { out.write(reinterpret_cast<const char*>(&v), 4); };
    const uint32_t dataSize = width * height * 4;
    u16(0x4D42);
    u32(54 + dataSize);
    u32(0);
    u32(54);
    u32(40);
    u32(width);
    u32(static_cast<uint32_t>(-static_cast<int32_t>(height)));
    u16(1);
    u16(32);
    u32(0);
    u32(dataSize);
    u32(2835);
    u32(2835);
    u32(0);
    u32(0);
    std::vector<uint8_t> opaque = bgra;
    for (size_t i = 3; i < opaque.size(); i += 4) {
        opaque[i] = 255;
    }
    out.write(reinterpret_cast<const char*>(opaque.data()), static_cast<std::streamsize>(opaque.size()));
    return static_cast<bool>(out);
}

} // namespace

int main(int argc, char** argv) {
    kstudio::StudioOptions options;
    std::string outPath = "studio_snapshot.bmp";
    int tab = 0;
    std::string select;
    bool hideBuffers = false;
    int frames = 12;
    std::string thenBackend;
    bool run = false;
    bool openMeta = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--out" && i + 1 < argc) {
            outPath = argv[++i];
        } else if (arg == "--backend" && i + 1 < argc) {
            options.backend = std::string(argv[++i]) == "compute" ? kstudio::SplattingBackend::Compute
                                                                   : kstudio::SplattingBackend::Raster;
        } else if (arg == "--tab" && i + 1 < argc) {
            tab = std::string(argv[++i]) == "compiled" ? 1 : 0;
        } else if (arg == "--select" && i + 1 < argc) {
            select = argv[++i];
        } else if (arg == "--hide-buffers") {
            hideBuffers = true;
        } else if (arg == "--frames" && i + 1 < argc) {
            frames = std::stoi(argv[++i]);
        } else if (arg == "--then-backend" && i + 1 < argc) {
            thenBackend = argv[++i];
        } else if (arg == "--example" && i + 1 < argc) {
            const auto example = kstudio::exampleFromName(argv[++i]);
            if (!example) {
                std::cerr << "Unknown --example '" << argv[i] << "'\n";
                return EXIT_FAILURE;
            }
            options.example = *example;
        } else if (arg == "--open-meta") {
            openMeta = true;
        } else if (arg == "--run") {
            run = true;
        } else if (arg == "--profile") {
            options.profiling = true;
        } else if (!arg.empty() && arg[0] != '-') {
            options.graphFile = arg;
        } else {
            std::cerr << "Unknown option '" << arg << "'\n";
            return EXIT_FAILURE;
        }
    }

    klartraum::setAssetRoot(KLARTRAUM_SOURCE_DIR);

    klartraum::HeadlessFrontend frontend;
    auto& engine = frontend.getKlartraumEngine();
    auto& vc = engine.getVulkanContext();
    auto overlay = std::make_shared<klartraum::ImGuiOverlay>(vc);
    engine.setOverlay(overlay);

    const VkExtent2D extent = vc.getSwapChainExtent();
    ImGuiIO& io = ImGui::GetIO();
    const float scale = static_cast<float>(extent.width) / kLogicalSize;
    io.DisplaySize = ImVec2(kLogicalSize, static_cast<float>(extent.height) / scale);
    io.DisplayFramebufferScale = ImVec2(scale, scale);
    io.DeltaTime = 1.0f / 60.0f;

    bool ok = true;
    {
        kstudio::StudioApp app(engine, options);
        app.showTab(tab);
        app.setHideBuffers(hideBuffers);
        if (!select.empty()) {
            if (auto kind = kstudio::kindFromName(select)) {
                for (const auto& node : app.graph().nodes()) {
                    if (node.kind == *kind) {
                        app.selectNode(node.id);
                        break;
                    }
                }
            }
        }

        // One frame per swapchain image at the end, so every image holds the
        // settled UI; image 0 is then read back.
        const int total = frames + static_cast<int>(vc.getNumberOfSwapChainImages());
        for (int f = 0; f < total; ++f) {
            if (run && f == 1) {
                app.requestRun();
            }
            if (openMeta && f == 1) {
                for (const auto& node : app.graph().nodes()) {
                    if (node.kind == kstudio::NodeKind::Meta) {
                        app.openMetaNode(node.id);
                        break;
                    }
                }
            }
            if (!thenBackend.empty() && f == frames / 2) {
                for (auto& node : app.editableGraph().nodes()) {
                    if (node.kind == kstudio::NodeKind::GaussianSplatting) {
                        node.as<kstudio::SplattingParams>().backend = thenBackend == "compute"
                                                                          ? kstudio::SplattingBackend::Compute
                                                                          : kstudio::SplattingBackend::Raster;
                    }
                }
                app.editableGraph().touch();
            }
            overlay->newFrame();
            app.drawGui();
            overlay->render();
            engine.step();
            vkQueueWaitIdle(vc.getGraphicsQueue());
        }

        const bool hasPresent = std::any_of(app.graph().nodes().begin(), app.graph().nodes().end(),
                                            [](const kstudio::Node& n) { return n.kind == kstudio::NodeKind::Present; });
        if (hasPresent && !app.hasAppliedGraph()) {
            std::cerr << "No graph was compiled: " << app.lastError() << "\n";
            ok = false;
        }
        if (run && !app.lastRunSucceeded()) {
            std::cerr << "The run failed: " << app.runError() << "\n";
            ok = false;
        }
        std::cout << "compiled elements: " << app.compiledGraph().nodes.size() << "\n";
    }

    const auto pixels = readSwapchainImage(vc, 0);
    if (!writeBmp(outPath, pixels, extent.width, extent.height)) {
        std::cerr << "cannot write " << outPath << "\n";
        ok = false;
    } else {
        std::cout << "wrote " << outPath << " (" << extent.width << "x" << extent.height << ")\n";
    }

    vkDeviceWaitIdle(vc.getDevice());
    engine.setOverlay(nullptr);
    overlay.reset();
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
