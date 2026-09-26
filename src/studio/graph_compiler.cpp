#include "studio/graph_compiler.hpp"

#include <algorithm>

#include "klartraum/computegraph/imageviewsrc.hpp"
#include "klartraum/gaussian_data_standard.hpp"
#include "klartraum/gaussian_splatting_factory.hpp"
#include "klartraum/klartraum_core.hpp"

namespace kstudio {

CompilePlan planGraph(const Graph& graph) {
    CompilePlan plan;
    plan.diagnostics = graph.validate();
    const bool hasErrors = std::any_of(plan.diagnostics.begin(), plan.diagnostics.end(),
                                       [](const Diagnostic& d) { return d.severity == Severity::Error; });
    if (hasErrors) {
        return plan;
    }

    // validate() guarantees a single Present fed by a fully connected
    // Gaussian Splatting node whose target is a Swapchain Target.
    const auto present = std::find_if(graph.nodes().begin(), graph.nodes().end(),
                                      [](const Node& n) { return n.kind == NodeKind::Present; });
    const Node* splatting = graph.inputNode(present->id, 0);
    const Node* scene = graph.inputNode(splatting->id, 0);
    const Node* camera = graph.inputNode(splatting->id, 1);
    const Node* target = graph.inputNode(splatting->id, 2);

    SplattingPlan splattingPlan;
    splattingPlan.presentNode = present->id;
    splattingPlan.splattingNode = splatting->id;
    splattingPlan.sceneNode = scene->id;
    splattingPlan.cameraNode = camera->id;
    splattingPlan.targetNode = target->id;
    splattingPlan.scenePath = scene->as<SceneParams>().path;
    splattingPlan.params = splatting->as<SplattingParams>();
    plan.splatting = splattingPlan;
    return plan;
}

klartraum::GsplatConfig toGsplatConfig(const SplattingParams& params) {
    klartraum::GsplatConfig config;
    config.spreadMultiplier = params.spreadMultiplier;
    config.maxMod = params.maxMod;
    config.numSortWGsCap = params.numSortWGsCap;
    config.splatTileX = params.splatTileX;
    config.splatTileY = params.splatTileY;
    config.shDegree = params.shDegree;
    config.alphaCullThreshold = params.alphaCullThreshold;
    config.useMeshShader = params.useMeshShader;
    return config;
}

klartraum::GsplatBackend toGsplatBackend(SplattingBackend backend) {
    return backend == SplattingBackend::Raster ? klartraum::GsplatBackend::Raster : klartraum::GsplatBackend::Compute;
}

BuiltGraph buildGraph(klartraum::KlartraumEngine& engine, const SplattingPlan& plan,
                      const std::shared_ptr<klartraum::GaussianDataStandard>& model) {
    auto& vc = engine.getVulkanContext();

    // Swapchain Target: one image per swapchain image, each waiting for its
    // acquire semaphore.
    const uint32_t numImages = vc.getNumberOfSwapChainImages();
    std::vector<VkImageView> imageViews(numImages);
    std::vector<VkImage> images(numImages);
    std::vector<VkExtent2D> extents(numImages, vc.getSwapChainExtent());
    for (uint32_t i = 0; i < numImages; ++i) {
        imageViews[i] = vc.getImageView(i);
        images[i] = vc.getSwapChainImage(i);
    }
    auto imageViewSrc = std::make_shared<klartraum::ImageViewSrc>(imageViews, images, extents);
    imageViewSrc->setName("Swapchain");
    for (uint32_t i = 0; i < numImages; ++i) {
        imageViewSrc->setWaitFor(i, vc.imageAvailableSemaphoresPerImage[i]);
    }

    auto cameraUBO = std::make_shared<klartraum::CameraUboType>();
    cameraUBO->setName("CameraUBO");

    BuiltGraph built;
    built.root = klartraum::createGaussianSplatting(vc, toGsplatBackend(plan.params.backend), imageViewSrc, cameraUBO,
                                                    model, toGsplatConfig(plan.params));

    built.owners[imageViewSrc.get()] = plan.targetNode;
    built.owners[cameraUBO.get()] = plan.cameraNode;
    const auto& buffers = model->buffers();
    for (const klartraum::ComputeGraphElement* element :
         {static_cast<klartraum::ComputeGraphElement*>(buffers.pos.get()),
          static_cast<klartraum::ComputeGraphElement*>(buffers.rot.get()),
          static_cast<klartraum::ComputeGraphElement*>(buffers.scale.get()),
          static_cast<klartraum::ComputeGraphElement*>(buffers.colAlpha.get()),
          static_cast<klartraum::ComputeGraphElement*>(buffers.shR.get()),
          static_cast<klartraum::ComputeGraphElement*>(buffers.shG.get()),
          static_cast<klartraum::ComputeGraphElement*>(buffers.shB.get())}) {
        if (element) {
            built.owners[element] = plan.sceneNode;
        }
    }

    engine.add(built.root);
    engine.setCameraUBO(cameraUBO);
    return built;
}

} // namespace kstudio
