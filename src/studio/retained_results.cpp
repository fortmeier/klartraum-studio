#include "studio/retained_results.hpp"

#include <algorithm>
#include <stdexcept>

#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/offscreen_target.hpp"

namespace kstudio {

namespace {

// Readable back and writable by copies, like the tensors it is copied from.
constexpr VkBufferUsageFlags kUsage =
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

template <typename T>
std::shared_ptr<klartraum::TensorElementInterface> makeTensor(klartraum::VulkanContext& vc, const TensorShape& shape,
                                                              const std::string& name) {
    auto tensor = vc.create<klartraum::TensorElementSinglePath<T>>(shape, kUsage);
    tensor->setName(name);
    // The buffer exists from now on, so a result can be copied in before a
    // graph that reads it is compiled; compiling it later keeps the buffer.
    tensor->_setup(vc, 1);
    return tensor;
}

// The tensor's data buffer and its size in bytes.
std::pair<VkBuffer, VkDeviceSize> dataOf(klartraum::TensorElementInterface& tensor) {
    if (auto* floats = dynamic_cast<klartraum::TensorElement<float>*>(&tensor)) {
        return {floats->getDataBuffer(0).getBuffer(), floats->getDataBufferMemSize()};
    }
    if (auto* ints = dynamic_cast<klartraum::TensorElement<int64_t>*>(&tensor)) {
        return {ints->getDataBuffer(0).getBuffer(), ints->getDataBufferMemSize()};
    }
    throw std::runtime_error("only float32 and int64 tensors can be retained");
}

} // namespace

void RetainedResults::storeTensor(OutputPin pin, klartraum::TensorElementInterface& tensor, const std::string& name) {
    const auto& dims = tensor.getDimensions();
    TensorType type{TensorShape(dims.begin(), dims.end()),
                    tensor.getElementType() == typeid(int64_t) ? ElementType::Int64 : ElementType::Float32};
    RetainedValue& value = values_[pin];
    if (!value.tensor || value.type != type) {
        value = RetainedValue{type,
                              type.element == ElementType::Int64 ? makeTensor<int64_t>(vc_, type.shape, name)
                                                                 : makeTensor<float>(vc_, type.shape, name),
                              nullptr};
        ++generation_;
    }
    const auto [source, bytes] = dataOf(tensor);
    const auto [destination, capacity] = dataOf(*value.tensor);
    if (bytes != capacity) {
        throw std::logic_error("retained tensor size mismatch");
    }
    vc_.copyBufferImmediate(source, destination, bytes);
}

void RetainedResults::storeHost(OutputPin pin, const HostTensor& values, const std::string& name) {
    const TensorType type{values.shape, ElementType::Float32};
    RetainedValue& value = values_[pin];
    if (!value.tensor || value.type != type) {
        value = RetainedValue{type, makeTensor<float>(vc_, type.shape, name), nullptr};
        ++generation_;
    }
    dynamic_cast<klartraum::TensorElement<float>&>(*value.tensor).getDataBuffer(0).memcopyFrom(values.values);
}

void RetainedResults::storeImage(OutputPin pin, VkImage image, VkImageLayout layout, VkExtent2D extent,
                                 const std::string& name) {
    RetainedValue& value = values_[pin];
    if (!value.image || value.image->extent().width != extent.width ||
        value.image->extent().height != extent.height) {
        value = RetainedValue{{}, nullptr, std::make_shared<klartraum::SinglePathImage>(vc_, extent)};
        value.image->setName(name);
        ++generation_;
    }
    value.image->copyFrom(image, layout, extent);
}

const RetainedValue* RetainedResults::find(OutputPin pin) const {
    auto it = values_.find(pin);
    return it == values_.end() ? nullptr : &it->second;
}

bool RetainedResults::has(const std::vector<OutputPin>& pins) const {
    return std::all_of(pins.begin(), pins.end(), [&](const OutputPin& pin) { return values_.contains(pin); });
}

void RetainedResults::clear() {
    if (!values_.empty()) {
        values_.clear();
        ++generation_;
    }
}

} // namespace kstudio
