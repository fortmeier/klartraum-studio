#include <algorithm>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>

#include <vulkan/vulkan.h>

#include "klartraum/imgui_frontend.hpp"
#include "klartraum/vulkan_helpers.hpp"

#include "studio/studio_app.hpp"

namespace {

constexpr int kWindowWidth = 1600;
constexpr int kWindowHeight = 960;

void printUsage(const char* program) {
    std::cout << "Usage: " << program << " [options] [graph.ktgraph.json]\n"
              << "  --example NAME             start with an example: gaussian-splatting (default),\n"
              << "                             autoencoder, splat-autoencoder, combined-scenes or\n"
              << "                             animated-scenes\n"
              << "  --spz PATH                 scene for the Gaussian-splatting examples\n"
              << "  --backend compute|raster   backend of the default graph (default: raster)\n"
              << "  --profile                  start with per-element GPU timings enabled\n"
              << "  --frames N                 exit after N frames (smoke testing)\n"
              << "  --help                     show this help\n";
}

} // namespace

int main(int argc, char** argv) {
    kstudio::StudioOptions options;
    int maxFrames = -1;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--example" && i + 1 < argc) {
            const auto example = kstudio::exampleFromName(argv[++i]);
            if (!example) {
                std::cerr << "Unknown --example '" << argv[i] << "'\n";
                return EXIT_FAILURE;
            }
            options.example = *example;
        } else if (arg == "--spz" && i + 1 < argc) {
            options.scenePath = argv[++i];
        } else if (arg == "--backend" && i + 1 < argc) {
            const std::string value = argv[++i];
            if (value == "compute") {
                options.backend = kstudio::SplattingBackend::Compute;
            } else if (value == "raster") {
                options.backend = kstudio::SplattingBackend::Raster;
            } else {
                std::cerr << "Unknown --backend '" << value << "' (expected compute or raster)\n";
                return EXIT_FAILURE;
            }
        } else if (arg == "--profile") {
            options.profiling = true;
        } else if (arg == "--frames" && i + 1 < argc) {
            maxFrames = std::stoi(argv[++i]);
        } else if (!arg.empty() && arg[0] != '-') {
            options.graphFile = arg;
        } else {
            std::cerr << "Unknown option '" << arg << "'\n";
            printUsage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    // klartraum loads its compiled shaders by paths relative to its source
    // directory; let it find them from any working directory.
    klartraum::setAssetRoot(KLARTRAUM_SOURCE_DIR);

    try {
        klartraum::ImGuiFrontend frontend;
        {
            kstudio::StudioApp app(frontend.getKlartraumEngine(), options, frontend.getGlfwWindow());
            frontend.setGui([&app]() { app.drawGui(); });

            // The resize is picked up by the first frame's event processing,
            // which rebuilds the graph for the larger swapchain.
            glfwSetWindowSize(frontend.getGlfwWindow(), kWindowWidth, kWindowHeight);

            frontend.loop(maxFrames);

            frontend.setGui(nullptr);
            const bool hasPresent =
                std::any_of(app.graph().nodes().begin(), app.graph().nodes().end(),
                            [](const kstudio::Node& n) { return n.kind == kstudio::NodeKind::Present; });
            if (maxFrames > 0 && hasPresent && !app.hasAppliedGraph()) {
                std::cerr << "No graph was compiled: " << app.lastError() << "\n";
                return EXIT_FAILURE;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "klartraum-studio: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
