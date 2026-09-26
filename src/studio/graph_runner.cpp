#include "studio/graph_runner.hpp"

#include <chrono>
#include <stdexcept>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/generalcomputation.hpp"
#include "klartraum/computegraph/noop.hpp"
#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/draw_component.hpp"
#include "klartraum/gaussian_data_standard.hpp"
#include "klartraum/gaussian_splatting_factory.hpp"
#include "klartraum/interface_camera_orbit.hpp"
#include "klartraum/offscreen_target.hpp"
#include "klartraum/onnx/onnx_network.hpp"

namespace kstudio {

namespace {

using klartraum::ComputeGraphElementPtr;
using FloatTensor = klartraum::TensorElement<float>;

// Matches shaders/onnx/image_to_tensor.comp.
struct ImageSizePushConstants {
    uint32_t width;
    uint32_t height;
};

// A tensor as a consumer connects to it: `producer` (at `slot`, or itself
// for -1) provides `tensor`.
struct TensorRef {
    ComputeGraphElementPtr producer;
    int slot = -1;
    std::shared_ptr<FloatTensor> tensor;
};

struct CameraUpdate {
    std::shared_ptr<klartraum::CameraUboType> ubo;
    CameraParams params;
    float aspect = 1.0f;
};

class RunBuilder {
public:
    RunBuilder(klartraum::VulkanContext& vc, const Graph& graph, const RunContext& context)
        : vc_(vc), graph_(graph), context_(context) {}

    void build(const Node& node) {
        try {
            buildNode(node);
        } catch (const std::exception& e) {
            throw std::runtime_error(node.title + ": " + e.what());
        }
    }

    RunResult run(const RunPlan& plan) {
        // One root depending on every sink's tensor.
        auto root = std::make_shared<klartraum::NoOp>(vc_);
        root->setName("Run");
        for (size_t i = 0; i < plan.sinks.size(); ++i) {
            const TensorRef& ref = inputTensor(*graph_.findNode(plan.sinks[i]));
            root->setInput(ref.producer, static_cast<int>(i), ref.slot);
        }

        klartraum::ComputeGraph computeGraph(vc_, 1);
        computeGraph.enableProfiling();
        computeGraph.compileFrom(root);

        // Buffers exist once the graph is compiled.
        for (const auto& [tensor, data] : uploads_) {
            tensor->getDataBuffer(0).memcopyFrom(data);
        }
        for (const auto& update : cameraUpdates_) {
            klartraum::InterfaceCameraOrbit orbit(update.params.up == UpAxis::Y
                                                      ? klartraum::InterfaceCameraOrbit::UpDirection::Y
                                                      : klartraum::InterfaceCameraOrbit::UpDirection::Z);
            orbit.initialize(vc_);
            orbit.setAzimuth(update.params.azimuth);
            orbit.setElevation(update.params.elevation);
            orbit.setDistance(update.params.distance);
            orbit.setPosition(glm::vec3(update.params.target[0], update.params.target[1], update.params.target[2]));
            orbit.setProjectionAspectRatio(update.aspect);
            orbit.update(update.ubo->ubo);
            update.ubo->update(0);
        }

        computeGraph.submitAndWait(vc_.getGraphicsQueue(), 0);

        RunResult result;
        for (int sink : plan.sinks) {
            const Node& node = *graph_.findNode(sink);
            try {
                const TensorRef& ref = inputTensor(node);
                const auto& dims = ref.tensor->getDimensions();  // NCHW
                if (dims.size() != 4 || dims[0] != 1) {
                    throw std::runtime_error("the tensor is not an image");
                }
                std::vector<float> data(ref.tensor->getDataElementCount());
                ref.tensor->getDataBuffer(0).memcopyTo(data);
                ImageRGBA8 image = tensorToImage(data, dims[1], dims[2], dims[3]);
                if (node.kind == NodeKind::ImageFileWriter) {
                    const auto path = context_.resolveOutput(node.as<ImageFileWriterParams>().path);
                    savePng(image, path);
                    result.written.push_back(path);
                }
                result.images[sink] = std::move(image);
            } catch (const std::exception& e) {
                throw std::runtime_error(node.title + ": " + e.what());
            }
        }
        for (const auto& [label, ms] : computeGraph.getProfilingResults()) {
            result.timings.emplace(label, ms);
        }
        result.compiled = introspect(root, owners_);
        return result;
    }

private:
    const TensorRef& inputTensor(const Node& node) {
        const Link* link = graph_.inputLink(node.id, 0);
        auto it = link ? tensors_.find(link->fromNode) : tensors_.end();
        if (it == tensors_.end()) {
            throw std::runtime_error("its input tensor was not built");
        }
        return it->second;
    }

    template <typename T> std::shared_ptr<T> built(std::map<int, std::shared_ptr<T>>& map, const Node& node, int slot) {
        const Link* link = graph_.inputLink(node.id, slot);
        auto it = link ? map.find(link->fromNode) : map.end();
        if (it == map.end()) {
            throw std::runtime_error("an input was not built");
        }
        return it->second;
    }

    void buildNode(const Node& node) {
        switch (node.kind) {
        case NodeKind::Scene:
            scenes_[node.id] = context_.loadScene(node.as<SceneParams>().path);
            break;
        case NodeKind::Camera:
            // Built per Gaussian Splatting node, which knows the aspect ratio.
            break;
        case NodeKind::OffscreenTarget: {
            const auto& p = node.as<OffscreenTargetParams>();
            auto target = std::make_shared<klartraum::OffscreenTarget>(vc_, VkExtent2D{p.width, p.height}, 1u);
            target->setName(node.title);
            owners_[target.get()] = node.id;
            targets_[node.id] = target;
            targetSizes_[node.id] = {p.width, p.height};
            break;
        }
        case NodeKind::SwapchainTarget:
            throw std::runtime_error("swapchain images cannot be used by Run; use an Offscreen Target");
        case NodeKind::GaussianSplatting: {
            const auto& p = node.as<SplattingParams>();
            const Node* cameraNode = graph_.inputNode(node.id, 1);
            const Link* targetLink = graph_.inputLink(node.id, 2);
            auto target = built(targets_, node, 2);
            const VkExtent2D size = targetSizes_.at(targetLink->fromNode);

            auto ubo = std::make_shared<klartraum::CameraUboType>();
            ubo->setName("CameraUBO");
            owners_[ubo.get()] = cameraNode->id;
            cameraUpdates_.push_back(CameraUpdate{ubo, cameraNode->as<CameraParams>(),
                                                  static_cast<float>(size.width) / static_cast<float>(size.height)});

            auto model = built(scenes_, node, 0);
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
                    owners_[element] = graph_.inputNode(node.id, 0)->id;
                }
            }

            auto splatting = klartraum::createGaussianSplatting(vc_, toGsplatBackend(p.backend), target, ubo, model,
                                                                toGsplatConfig(p));
            owners_[splatting.get()] = node.id;
            renders_[node.id] = splatting;
            renderSizes_[node.id] = size;
            break;
        }
        case NodeKind::ImageToTensor: {
            const Link* source = graph_.inputLink(node.id, 0);
            auto render = built(renders_, node, 0);
            const VkExtent2D size = renderSizes_.at(source->fromNode);

            auto tensor = vc_.create<FloatTensor>(std::vector<uint32_t>{1, 3, size.height, size.width});
            tensor->setName(node.title + " tensor");
            auto convert = vc_.create<klartraum::GeneralComputation<ImageSizePushConstants>>(
                "shaders/onnx/image_to_tensor.comp.spv");
            convert->setName(node.title);
            convert->setPushConstants({{size.width, size.height}});
            convert->setGroupCount((size.width + 7) / 8, (size.height + 7) / 8, 1);
            // Slot 0 of a splatting backend is its target image.
            convert->setInput(render, 0, 0);
            convert->setInput(tensor, 1);
            // The backends leave offscreen targets ready for a transfer.
            convert->setImageLayoutTransition(0, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                                              VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                              VK_ACCESS_MEMORY_READ_BIT, VK_ACCESS_SHADER_READ_BIT);
            owners_[tensor.get()] = node.id;
            owners_[convert.get()] = node.id;
            tensors_[node.id] = TensorRef{convert, 1, tensor};
            break;
        }
        case NodeKind::ImageFile: {
            const auto& p = node.as<ImageFileParams>();
            const ImageRGBA8 image = resizeImage(loadImage(context_.resolveInput(p.path)), p.width, p.height);
            auto tensor = vc_.create<FloatTensor>(std::vector<uint32_t>{1, 3, p.height, p.width});
            tensor->setName(node.title);
            uploads_.emplace_back(tensor, imageToTensor(image));
            owners_[tensor.get()] = node.id;
            tensors_[node.id] = TensorRef{tensor, -1, tensor};
            break;
        }
        case NodeKind::OnnxModel: {
            const auto& p = node.as<OnnxModelParams>();
            std::string error;
            const auto info = context_.onnxInfo(p.path, error);
            if (!info) {
                throw std::runtime_error(error);
            }
            if (info->inputs.size() != 1 || info->outputs.empty()) {
                throw std::runtime_error("only models with one input are supported");
            }
            const TensorRef& input = inputTensor(node);
            auto network = vc_.create<klartraum::OnnxNetwork>(context_.resolveInput(p.path).string());
            network->setName(node.title);
            network->setInputTensor(info->inputs[0].name, input.producer, input.slot);
            auto output = std::dynamic_pointer_cast<FloatTensor>(network->getOutputElement(info->outputs[0].name));
            if (!output) {
                throw std::runtime_error("the model's output is not a float tensor");
            }
            owners_[network.get()] = node.id;
            // Slot i of an OnnxNetwork is its i-th output tensor.
            tensors_[node.id] = TensorRef{network, 0, output};
            break;
        }
        case NodeKind::Preview:
        case NodeKind::ImageFileWriter:
        case NodeKind::Present:
            break;
        }
    }

    klartraum::VulkanContext& vc_;
    const Graph& graph_;
    const RunContext& context_;

    std::map<int, std::shared_ptr<klartraum::GaussianDataStandard>> scenes_;
    std::map<int, std::shared_ptr<klartraum::OffscreenTarget>> targets_;
    std::map<int, VkExtent2D> targetSizes_;
    std::map<int, std::shared_ptr<klartraum::ComputeGraphElement>> renders_;
    std::map<int, VkExtent2D> renderSizes_;
    std::map<int, TensorRef> tensors_;
    std::vector<std::pair<std::shared_ptr<FloatTensor>, std::vector<float>>> uploads_;
    std::vector<CameraUpdate> cameraUpdates_;
    std::map<const klartraum::ComputeGraphElement*, int> owners_;
};

} // namespace

RunResult runGraph(klartraum::VulkanContext& vulkanContext, const Graph& graph, const RunPlan& plan,
                   const RunContext& context) {
    const auto start = std::chrono::steady_clock::now();
    RunBuilder builder(vulkanContext, graph, context);
    for (int id : plan.nodes) {
        builder.build(*graph.findNode(id));
    }
    RunResult result = builder.run(plan);
    result.milliseconds =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    return result;
}

} // namespace kstudio
