#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

namespace kstudio {

// An 8-bit RGBA image, rows top to bottom.
struct ImageRGBA8 {
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<uint8_t> pixels;  // width * height * 4

    bool empty() const { return pixels.empty(); }
};

// Loads PNG, JPEG, BMP, TGA, ... (everything stb_image reads). Throws
// std::runtime_error with the reason on failure.
ImageRGBA8 loadImage(const std::filesystem::path& path);

// Writes a PNG. Throws std::runtime_error on failure.
void savePng(const ImageRGBA8& image, const std::filesystem::path& path);

ImageRGBA8 resizeImage(const ImageRGBA8& image, uint32_t width, uint32_t height);

// Tensors use klartraum's image layout: NCHW floats in [0, 1], batch 1, the
// channels as separate planes (see shaders/onnx/image_to_tensor.comp).
// imageToTensor writes 3 channels (RGB).
std::vector<float> imageToTensor(const ImageRGBA8& image);

// Accepts 1 channel (grey) or 3 channels (RGB); values are clamped to [0, 1].
// Throws std::invalid_argument if the data does not match the shape.
ImageRGBA8 tensorToImage(const std::vector<float>& data, uint32_t channels, uint32_t height, uint32_t width);

} // namespace kstudio
