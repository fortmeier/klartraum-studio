#include "studio/graph_runner.hpp"

#include <chrono>
#include <format>
#include <set>
#include <stdexcept>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/generalcomputation.hpp"
#include "klartraum/computegraph/gaussianmerge.hpp"
#include "klartraum/computegraph/gaussiantransform.hpp"
#include "klartraum/computegraph/hostvalues.hpp"
#include "klartraum/computegraph/imageresample.hpp"
#include "klartraum/computegraph/imageviewsrc.hpp"
#include "klartraum/computegraph/noop.hpp"
#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/computegraph/transformbuffer.hpp"
#include "klartraum/draw_component.hpp"
#include "klartraum/gaussian_data_standard.hpp"
#include "klartraum/gaussian_splatting_factory.hpp"
#include "klartraum/interface_camera_orbit.hpp"
#include "klartraum/klartraum_core.hpp"
#include "klartraum/offscreen_target.hpp"
#include "klartraum/onnx/onnx_network.hpp"

#include "studio/cpu_numbers.hpp"

namespace kstudio {

namespace {

using klartraum::ComputeGraphElementPtr;
using FloatTensor = klartraum::TensorElement<float>;

// Matches shaders/onnx/image_to_tensor.comp and tensor_to_image.comp.
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

// An image as a consumer connects to it: slot `slot` of `producer` is the
// image, left in `layout`.
struct ImageRef {
    ComputeGraphElementPtr producer;
    int slot = 0;
    VkExtent2D extent{};
    VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL;
};

struct CameraUpdate {
    std::shared_ptr<klartraum::CameraUboType> ubo;
    CameraParams params;
    float aspect = 1.0f;
};

klartraum::ResampleFilter toKlartraum(ResampleFilter filter) {
    return filter == ResampleFilter::Nearest ? klartraum::ResampleFilter::Nearest
                                             : klartraum::ResampleFilter::Bilinear;
}

// Builds authoring nodes as klartraum elements, either for Run (one path, no
// swapchain) or for the window's frame loop (one path per swapchain image).
class ElementBuilder {
public:
    ElementBuilder(klartraum::VulkanContext& vc, const Graph& graph, const RunContext& context, bool live)
        : vc_(vc), graph_(graph), context_(context), live_(live),
          numPaths_(live ? vc.getNumberOfSwapChainImages() : 1u) {}

    void build(const Node& node) {
        try {
            buildNode(node);
        } catch (const std::exception& e) {
            throw std::runtime_error(node.title + ": " + e.what());
        }
    }

    // One root depending on every sink's tensor.
    ComputeGraphElementPtr runRoot(const RunPlan& plan) {
        auto root = std::make_shared<klartraum::NoOp>(vc_);
        root->setName("Run");
        inserted_.insert(root.get());
        for (size_t i = 0; i < plan.sinks.size(); ++i) {
            const TensorRef& ref = sinkTensor(*graph_.findNode(plan.sinks[i]));
            root->setInput(ref.producer, static_cast<int>(i), ref.slot);
        }
        return root;
    }

    // Brings the Present node's image into the swapchain.
    ComputeGraphElementPtr presentRoot(const Node& present) {
        try {
            return buildPresent(present);
        } catch (const std::exception& e) {
            throw std::runtime_error(present.title + ": " + e.what());
        }
    }

    // Fills uploaded tensors and, for Run, the camera buffers. Buffers exist
    // once the graph is compiled.
    void upload() {
        for (const auto& [tensor, data] : uploads_) {
            for (uint32_t path = 0; path < numPaths_; ++path) {
                tensor->getDataBuffer(path).memcopyFrom(data);
            }
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
    }

    RunResult readSinks(const RunPlan& plan) {
        RunResult result;
        for (int sink : plan.sinks) {
            const Node& node = *graph_.findNode(sink);
            try {
                const TensorRef& ref = sinkTensor(node);
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
        return result;
    }

    std::shared_ptr<klartraum::CameraUboType> cameraUbo(int cameraNode) const {
        auto it = cameraUbos_.find(cameraNode);
        return it == cameraUbos_.end() ? nullptr : it->second;
    }

    const std::map<const klartraum::ComputeGraphElement*, int>& owners() const { return owners_; }
    const std::set<const klartraum::ComputeGraphElement*>& inserted() const { return inserted_; }
    const std::vector<HostBinding>& bindings() const { return bindings_; }

private:
    const TensorRef& inputTensor(const Node& node) {
        const Link* link = graph_.inputLink(node.id, 0);
        auto it = link ? tensors_.find(link->fromNode) : tensors_.end();
        if (it == tensors_.end()) {
            throw std::runtime_error("its input tensor was not built");
        }
        return it->second;
    }

    // The node's input image, in GENERAL layout so that compute shaders can
    // read it. An image in another layout (a splatting backend leaves its
    // target ready for a transfer or for presenting) is transitioned once,
    // for all of its readers.
    const ImageRef& readableImage(const Node& node, int slot = 0) {
        const Link* link = graph_.inputLink(node.id, slot);
        auto it = link ? images_.find(link->fromNode) : images_.end();
        if (it == images_.end()) {
            throw std::runtime_error("its input image was not built");
        }
        ImageRef& image = it->second;
        if (image.layout != VK_IMAGE_LAYOUT_GENERAL) {
            auto transition = std::make_shared<klartraum::ImageViewSrcTransition>(
                image.layout, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
            transition->setName(graph_.findNode(link->fromNode)->title + " to general");
            transition->setInput(image.producer, 0, image.slot);
            owners_[transition.get()] = link->fromNode;
            inserted_.insert(transition.get());
            image = ImageRef{transition, 0, image.extent, VK_IMAGE_LAYOUT_GENERAL};
        }
        return image;
    }

    // A sink fed with an image reads the tensor it was converted to.
    const TensorRef& sinkTensor(const Node& node) {
        auto it = tensors_.find(node.id);
        return it != tensors_.end() ? it->second : inputTensor(node);
    }

    // Converts an image into a 1x3xHxW tensor owned by `node`.
    TensorRef convertImage(const ImageRef& image, const Node& node) {
        const VkExtent2D size = image.extent;
        auto tensor = vc_.create<FloatTensor>(std::vector<uint32_t>{1, 3, size.height, size.width});
        tensor->setName(node.title + " tensor");
        auto convert = vc_.create<klartraum::GeneralComputation<ImageSizePushConstants>>(
            "shaders/onnx/image_to_tensor.comp.spv");
        convert->setName(node.kind == NodeKind::ImageToTensor ? node.title : node.title + " image to tensor");
        convert->setPushConstants({{size.width, size.height}});
        convert->setGroupCount((size.width + 7) / 8, (size.height + 7) / 8, 1);
        convert->setInput(image.producer, 0, image.slot);
        convert->setInput(tensor, 1);
        convert->setImageLayoutTransition(0, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                                          VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                          VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        owners_[tensor.get()] = node.id;
        owners_[convert.get()] = node.id;
        if (node.kind != NodeKind::ImageToTensor) {
            // A sink reading an image: the conversion is the studio's.
            inserted_.insert(tensor.get());
            inserted_.insert(convert.get());
        }
        return TensorRef{convert, 1, tensor};
    }

    // The window's swapchain images, waiting for their acquisition. Created
    // once: only one element may wait for an image's acquire semaphore.
    std::shared_ptr<klartraum::ImageViewSrc> swapchain(int owner) {
        if (!live_) {
            throw std::runtime_error("swapchain images cannot be used by Run; use an Offscreen Target");
        }
        if (!swapchain_) {
            const uint32_t numImages = vc_.getNumberOfSwapChainImages();
            std::vector<VkImageView> imageViews(numImages);
            std::vector<VkImage> images(numImages);
            std::vector<VkExtent2D> extents(numImages, vc_.getSwapChainExtent());
            for (uint32_t i = 0; i < numImages; ++i) {
                imageViews[i] = vc_.getImageView(i);
                images[i] = vc_.getSwapChainImage(i);
            }
            swapchain_ = std::make_shared<klartraum::ImageViewSrc>(imageViews, images, extents);
            swapchain_->setName("Swapchain");
            for (uint32_t i = 0; i < numImages; ++i) {
                swapchain_->setWaitFor(i, vc_.imageAvailableSemaphoresPerImage[i]);
            }
            owners_[swapchain_.get()] = owner;
        }
        return swapchain_;
    }

    // The layout swapchain images are handed to the window in.
    VkImageLayout presentLayout() const {
        return vc_.hasSurface() ? VK_IMAGE_LAYOUT_PRESENT_SRC_KHR : VK_IMAGE_LAYOUT_GENERAL;
    }

    ComputeGraphElementPtr buildPresent(const Node& present) {
        const Link* link = graph_.inputLink(present.id, 0);
        const ImageRef& image = images_.at(link->fromNode);
        // A rendering into the swapchain is presented as it is.
        if (swapchain_ && image.producer->getInputElement(image.slot) == swapchain_) {
            if (image.layout == presentLayout()) {
                return image.producer;
            }
            auto transition = std::make_shared<klartraum::ImageViewSrcTransition>(
                image.layout, presentLayout(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_ACCESS_MEMORY_WRITE_BIT, 0);
            transition->setName("Present");
            transition->setInput(image.producer, 0, image.slot);
            owners_[transition.get()] = present.id;
            inserted_.insert(transition.get());
            return transition;
        }

        // Anything else is stretched to the window.
        const ImageRef& source = readableImage(present);
        auto target = swapchain(present.id);
        const VkExtent2D extent = vc_.getSwapChainExtent();
        auto resample = vc_.create<klartraum::ImageResample>(source.extent, extent);
        resample->setName(present.title + " resample");
        resample->setInput(source.producer, 0, source.slot);
        resample->setInput(target, 1);
        owners_[resample.get()] = present.id;
        inserted_.insert(resample.get());
        if (presentLayout() == VK_IMAGE_LAYOUT_GENERAL) {
            return resample;
        }
        auto transition = std::make_shared<klartraum::ImageViewSrcTransition>(
            VK_IMAGE_LAYOUT_GENERAL, presentLayout(), VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_ACCESS_SHADER_WRITE_BIT, 0);
        transition->setName(present.title);
        transition->setInput(resample, 0, 1);
        owners_[transition.get()] = present.id;
        inserted_.insert(transition.get());
        return transition;
    }

    void hostStep(const std::string& step, std::chrono::steady_clock::time_point start) const {
        if (context_.hostStep) {
            context_.hostStep(step, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                                        .count());
        }
    }

    // A value built for the node feeding input `slot`.
    template <typename T> const T& builtValue(const std::map<int, T>& map, const Node& node, int slot) {
        const Link* link = graph_.inputLink(node.id, slot);
        auto it = link ? map.find(link->fromNode) : map.end();
        if (it == map.end()) {
            throw std::runtime_error("an input was not built");
        }
        return it->second;
    }

    // Owns an element building Gaussians and the buffers it writes.
    void ownGaussianElement(const klartraum::ComputeGraphElementPtr& element,
                            const klartraum::GaussianSoABuffers& output, const Node& node) {
        element->setName(node.title);
        owners_[element.get()] = node.id;
        for (const auto* ref : output.all()) {
            owners_[ref->buffer().get()] = node.id;
        }
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
        case NodeKind::TransformGaussians:
        case NodeKind::MergeGaussians:
            // CPU work, done by the Upload Gaussians node they feed.
            break;
        case NodeKind::Number:
        case NodeKind::Time:
        case NodeKind::Sine:
            // Evaluated on the CPU for the bindings of the nodes they feed.
            break;
        case NodeKind::UploadNumber: {
            auto values = std::make_shared<klartraum::HostFloat>(vc_, 1u);
            values->setName(node.title);
            owners_[values.get()] = node.id;
            bindings_.push_back(HostBinding{values, node.id, -1});
            numbers_[node.id] = klartraum::BufferRef{values};
            break;
        }
        case NodeKind::MakeTransform: {
            std::array<klartraum::BufferRef, 7> parameters;
            for (int i = 0; i < 7; ++i) {
                if (graph_.inputLink(node.id, i)) {
                    parameters[i] = builtValue(numbers_, node, i);
                    continue;
                }
                // An unconnected input takes the node's value, set every frame.
                auto values = std::make_shared<klartraum::HostFloat>(vc_, 1u);
                values->setName(node.title + " " + std::string(kindInfo(node.kind).inputs[i].name));
                owners_[values.get()] = node.id;
                inserted_.insert(values.get());
                bindings_.push_back(HostBinding{values, node.id, i});
                parameters[i] = klartraum::BufferRef{values};
            }
            const auto transform = klartraum::createTransformBuffer(vc_, parameters);
            transform.element->setName(node.title);
            owners_[transform.element.get()] = node.id;
            owners_[transform.transform.buffer().get()] = node.id;
            transforms_[node.id] = transform.transform;
            break;
        }
        case NodeKind::TransformGaussiansGpu: {
            const auto moved = klartraum::createGaussianTransform(vc_, builtValue(gaussians_, node, 0),
                                                                  builtValue(transforms_, node, 1));
            ownGaussianElement(moved.element, moved.output, node);
            gaussians_[node.id] = moved.output;
            break;
        }
        case NodeKind::MergeGaussiansGpu: {
            const auto merged =
                klartraum::createGaussianMerge(vc_, builtValue(gaussians_, node, 0), builtValue(gaussians_, node, 1));
            ownGaussianElement(merged.element, merged.output, node);
            gaussians_[node.id] = merged.output;
            break;
        }
        case NodeKind::UploadGaussians: {
            const auto start = std::chrono::steady_clock::now();
            auto model = context_.loadGaussians(gaussianParts(graph_, node.id));
            hostStep(std::format("{}: {} Gaussians", node.title, model->count()), start);
            for (const auto* ref : model->buffers().all()) {
                owners_[ref->buffer().get()] = node.id;
            }
            gaussians_[node.id] = model->buffers();
            break;
        }
        case NodeKind::Camera:
            if (live_) {
                // Updated every frame from the window's orbit camera.
                auto ubo = std::make_shared<klartraum::CameraUboType>();
                ubo->setName("CameraUBO");
                owners_[ubo.get()] = node.id;
                cameraUbos_[node.id] = ubo;
            }
            // For Run, built per Gaussian Splatting node, which knows the
            // aspect ratio.
            break;
        case NodeKind::OffscreenTarget: {
            const auto& p = node.as<OffscreenTargetParams>();
            auto target = std::make_shared<klartraum::OffscreenTarget>(vc_, VkExtent2D{p.width, p.height}, numPaths_);
            target->setName(node.title);
            owners_[target.get()] = node.id;
            targets_[node.id] = target;
            targetSizes_[node.id] = {p.width, p.height};
            break;
        }
        case NodeKind::SwapchainTarget:
            targets_[node.id] = swapchain(node.id);
            targetSizes_[node.id] = vc_.getSwapChainExtent();
            break;
        case NodeKind::GaussianSplatting: {
            const auto& p = node.as<SplattingParams>();
            const Node* cameraNode = graph_.inputNode(node.id, 1);
            const Link* targetLink = graph_.inputLink(node.id, 2);
            auto target = built(targets_, node, 2);
            const VkExtent2D size = targetSizes_.at(targetLink->fromNode);

            std::shared_ptr<klartraum::CameraUboType> ubo;
            if (live_) {
                ubo = built(cameraUbos_, node, 1);
            } else {
                ubo = std::make_shared<klartraum::CameraUboType>();
                ubo->setName("CameraUBO");
                owners_[ubo.get()] = cameraNode->id;
                cameraUpdates_.push_back(CameraUpdate{ubo, cameraNode->as<CameraParams>(),
                                                      static_cast<float>(size.width) / static_cast<float>(size.height)});
            }

            const klartraum::GaussianSoABuffers& gaussians = builtValue(gaussians_, node, 0);

            auto splatting = klartraum::createGaussianSplatting(vc_, toGsplatBackend(p.backend), target, ubo,
                                                                gaussians, toGsplatConfig(p));
            owners_[splatting.get()] = node.id;
            // Slot 0 of a splatting backend is its target image, left for a
            // transfer (offscreen) or for presenting (swapchain).
            images_[node.id] = ImageRef{splatting, 0, size, target->getFinalLayoutOverride().value_or(presentLayout())};
            break;
        }
        case NodeKind::ImageToTensor:
            tensors_[node.id] = convertImage(readableImage(node), node);
            break;
        case NodeKind::TensorToImage: {
            const TensorRef& input = inputTensor(node);
            const auto& dims = input.tensor->getDimensions();  // NCHW
            if (dims.size() != 4 || dims[0] != 1 || dims[1] != 3) {
                throw std::runtime_error("expects a 1x3xHxW tensor");
            }
            const VkExtent2D size{dims[3], dims[2]};
            auto target = std::make_shared<klartraum::OffscreenTarget>(vc_, size, numPaths_);
            target->setName(node.title + " image");
            auto convert = vc_.create<klartraum::GeneralComputation<ImageSizePushConstants>>(
                "shaders/onnx/tensor_to_image.comp.spv");
            convert->setName(node.title);
            convert->setPushConstants({{size.width, size.height}});
            convert->setGroupCount((size.width + 7) / 8, (size.height + 7) / 8, 1);
            convert->setInput(input.producer, 0, input.slot);
            convert->setInput(target, 1);
            // Every pixel is written, so the previous contents are discarded.
            convert->setImageLayoutTransition(1, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                                              VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                              0, VK_ACCESS_SHADER_WRITE_BIT);
            owners_[target.get()] = node.id;
            owners_[convert.get()] = node.id;
            images_[node.id] = ImageRef{convert, 1, size, VK_IMAGE_LAYOUT_GENERAL};
            break;
        }
        case NodeKind::Resample: {
            const auto& p = node.as<ResampleParams>();
            const ImageRef& source = readableImage(node);
            const VkExtent2D size{p.width, p.height};
            auto target = std::make_shared<klartraum::OffscreenTarget>(vc_, size, numPaths_);
            target->setName(node.title + " image");
            auto resample = vc_.create<klartraum::ImageResample>(source.extent, size, toKlartraum(p.filter));
            resample->setName(node.title);
            resample->setInput(source.producer, 0, source.slot);
            resample->setInput(target, 1);
            owners_[target.get()] = node.id;
            owners_[resample.get()] = node.id;
            images_[node.id] = ImageRef{resample, 1, size, VK_IMAGE_LAYOUT_GENERAL};
            break;
        }
        case NodeKind::ImageFile: {
            const auto& p = node.as<ImageFileParams>();
            const auto start = std::chrono::steady_clock::now();
            const ImageRGBA8 image = resizeImage(loadImage(context_.resolveInput(p.path)), p.width, p.height);
            hostStep(std::format("{}: decoded and resized to {} x {}", node.title, p.width, p.height), start);
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
            // Shapes that depend on the window are only known here.
            const auto& dims = input.tensor->getDimensions();
            const TensorShape shape(dims.begin(), dims.end());
            if (shape != info->inputs[0].shape) {
                throw std::runtime_error(std::format("the model expects a {} tensor but gets {}",
                                                     shapeToString(info->inputs[0].shape), shapeToString(shape)));
            }
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
            if (graph_.inputType(node.id, 0) == PinType::Image) {
                tensors_[node.id] = convertImage(readableImage(node), node);
            }
            break;
        case NodeKind::Present:
            // Built last, by presentRoot().
            break;
        }
    }

    klartraum::VulkanContext& vc_;
    const Graph& graph_;
    const RunContext& context_;
    const bool live_;
    const uint32_t numPaths_;

    std::map<int, std::shared_ptr<klartraum::ImageViewSrc>> targets_;
    std::map<int, VkExtent2D> targetSizes_;
    std::shared_ptr<klartraum::ImageViewSrc> swapchain_;
    std::map<int, std::shared_ptr<klartraum::CameraUboType>> cameraUbos_;
    std::map<int, ImageRef> images_;
    std::map<int, TensorRef> tensors_;
    std::vector<std::pair<std::shared_ptr<FloatTensor>, std::vector<float>>> uploads_;
    std::vector<CameraUpdate> cameraUpdates_;
    std::map<int, klartraum::GaussianSoABuffers> gaussians_;
    std::map<int, klartraum::BufferRef> numbers_;
    std::map<int, klartraum::BufferRef> transforms_;
    std::vector<HostBinding> bindings_;
    std::map<const klartraum::ComputeGraphElement*, int> owners_;
    std::set<const klartraum::ComputeGraphElement*> inserted_;
};

} // namespace

void applyBindings(const Graph& graph, const std::vector<HostBinding>& bindings, double time) {
    for (const auto& binding : bindings) {
        const Node* node = graph.findNode(binding.node);
        if (!node) {
            continue;
        }
        if (node->kind == NodeKind::UploadNumber) {
            if (const Link* link = graph.inputLink(node->id, 0)) {
                binding.values->set(0, evaluateNumber(graph, link->fromNode, time));
            }
        } else if (node->kind == NodeKind::MakeTransform) {
            binding.values->set(0, node->as<MakeTransformParams>().component(binding.component));
        }
    }
}

RunResult runGraph(klartraum::VulkanContext& vulkanContext, const Graph& graph, const RunPlan& plan,
                   const RunContext& context) {
    const auto start = std::chrono::steady_clock::now();
    ElementBuilder builder(vulkanContext, graph, context, false);
    for (int id : plan.nodes) {
        builder.build(*graph.findNode(id));
    }
    const ComputeGraphElementPtr root = builder.runRoot(plan);

    klartraum::ComputeGraph computeGraph(vulkanContext, 1);
    computeGraph.enableProfiling();
    computeGraph.compileFrom(root);
    builder.upload();
    applyBindings(graph, builder.bindings(), context.time);
    computeGraph.submitAndWait(vulkanContext.getGraphicsQueue(), 0);

    RunResult result = builder.readSinks(plan);
    for (const auto& [label, ms] : computeGraph.getProfilingResults()) {
        result.timings.emplace(label, ms);
    }
    result.compiled = introspect(root, builder.owners(), -1, builder.inserted());
    result.milliseconds =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    return result;
}

BuiltGraph buildLiveGraph(klartraum::KlartraumEngine& engine, const Graph& graph, const LivePlan& plan,
                          const RunContext& context) {
    ElementBuilder builder(engine.getVulkanContext(), graph, context, true);
    for (int id : graph.topologicalOrder()) {
        if (plan.contains(id) && id != plan.presentNode) {
            builder.build(*graph.findNode(id));
        }
    }
    BuiltGraph built;
    built.root = builder.presentRoot(*graph.findNode(plan.presentNode));
    built.owners = builder.owners();
    built.inserted = builder.inserted();
    built.bindings = builder.bindings();

    engine.add(built.root);
    builder.upload();
    engine.setCameraUBO(builder.cameraUbo(plan.cameraNode));
    return built;
}

} // namespace kstudio
