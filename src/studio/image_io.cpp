#include "studio/image_io.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_resize2.h>
#include <stb_image_write.h>

namespace kstudio {

ImageRGBA8 loadImage(const std::filesystem::path& path) {
    int width = 0, height = 0, channels = 0;
    stbi_uc* data = stbi_load(path.string().c_str(), &width, &height, &channels, 4);
    if (!data) {
        throw std::runtime_error("cannot load image " + path.string() + ": " + stbi_failure_reason());
    }
    ImageRGBA8 image;
    image.width = static_cast<uint32_t>(width);
    image.height = static_cast<uint32_t>(height);
    image.pixels.assign(data, data + static_cast<size_t>(width) * height * 4);
    stbi_image_free(data);
    return image;
}

void savePng(const ImageRGBA8& image, const std::filesystem::path& path) {
    if (image.empty()) {
        throw std::runtime_error("cannot write an empty image to " + path.string());
    }
    const int stride = static_cast<int>(image.width) * 4;
    if (!stbi_write_png(path.string().c_str(), static_cast<int>(image.width), static_cast<int>(image.height), 4,
                        image.pixels.data(), stride)) {
        throw std::runtime_error("cannot write " + path.string());
    }
}

ImageRGBA8 resizeImage(const ImageRGBA8& image, uint32_t width, uint32_t height) {
    if (image.width == width && image.height == height) {
        return image;
    }
    ImageRGBA8 resized;
    resized.width = width;
    resized.height = height;
    resized.pixels.resize(static_cast<size_t>(width) * height * 4);
    if (!stbir_resize_uint8_srgb(image.pixels.data(), static_cast<int>(image.width), static_cast<int>(image.height),
                                 0, resized.pixels.data(), static_cast<int>(width), static_cast<int>(height), 0,
                                 STBIR_RGBA)) {
        throw std::runtime_error("image resize failed");
    }
    return resized;
}

std::vector<float> imageToTensor(const ImageRGBA8& image) {
    const size_t plane = static_cast<size_t>(image.width) * image.height;
    std::vector<float> data(3 * plane);
    for (size_t i = 0; i < plane; ++i) {
        for (size_t c = 0; c < 3; ++c) {
            data[c * plane + i] = static_cast<float>(image.pixels[i * 4 + c]) / 255.0f;
        }
    }
    return data;
}

ImageRGBA8 tensorToImage(const std::vector<float>& data, uint32_t channels, uint32_t height, uint32_t width) {
    if (channels != 1 && channels != 3) {
        throw std::invalid_argument("only 1- or 3-channel tensors can be shown as images, not " +
                                    std::to_string(channels));
    }
    const size_t plane = static_cast<size_t>(width) * height;
    if (data.size() != channels * plane) {
        throw std::invalid_argument("tensor data does not match its shape");
    }
    ImageRGBA8 image;
    image.width = width;
    image.height = height;
    image.pixels.resize(plane * 4);
    auto toByte = [](float v) {
        // NaN is treated as 0.
        const float clamped = std::isnan(v) ? 0.0f : std::clamp(v, 0.0f, 1.0f);
        return static_cast<uint8_t>(std::lround(clamped * 255.0f));
    };
    for (size_t i = 0; i < plane; ++i) {
        for (size_t c = 0; c < 3; ++c) {
            image.pixels[i * 4 + c] = toByte(data[(channels == 3 ? c : 0) * plane + i]);
        }
        image.pixels[i * 4 + 3] = 255;
    }
    return image;
}

} // namespace kstudio
