/**
 * TESTS:
 * - tensorRoundTrip: an RGB image survives imageToTensor/tensorToImage unchanged
 * - tensorLayoutIsPlanar: imageToTensor writes the R, G and B planes one after another
 * - tensorToImageClampsAndGrey: out-of-range and NaN values clamp; one channel becomes grey
 * - tensorToImageRejectsBadShapes: unsupported channel counts and size mismatches throw
 * - resizeChangesSize: resizing yields the requested size and keeps a uniform colour
 * - pngRoundTrip: savePng/loadImage write and read back the same pixels
 * - loadMissingFileThrows: loading a missing file throws with the path in the message
 **/

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <limits>

#include "studio/image_io.hpp"

using namespace kstudio;

namespace {

ImageRGBA8 gradient(uint32_t width, uint32_t height) {
    ImageRGBA8 image;
    image.width = width;
    image.height = height;
    image.pixels.resize(static_cast<size_t>(width) * height * 4);
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            uint8_t* p = &image.pixels[(static_cast<size_t>(y) * width + x) * 4];
            p[0] = static_cast<uint8_t>(x * 255 / std::max(1u, width - 1));
            p[1] = static_cast<uint8_t>(y * 255 / std::max(1u, height - 1));
            p[2] = static_cast<uint8_t>((x + y) % 256);
            p[3] = 255;
        }
    }
    return image;
}

} // namespace

TEST(ImageIo, tensorRoundTrip) {
    const ImageRGBA8 image = gradient(7, 5);
    const auto tensor = imageToTensor(image);
    ASSERT_EQ(tensor.size(), 3u * 7 * 5);
    const ImageRGBA8 back = tensorToImage(tensor, 3, 5, 7);
    EXPECT_EQ(back.width, 7u);
    EXPECT_EQ(back.height, 5u);
    EXPECT_EQ(back.pixels, image.pixels);
}

TEST(ImageIo, tensorLayoutIsPlanar) {
    ImageRGBA8 image;
    image.width = 2;
    image.height = 1;
    image.pixels = {255, 0, 0, 255, 0, 0, 255, 255};  // red, blue
    const auto tensor = imageToTensor(image);
    EXPECT_EQ(tensor, (std::vector<float>{1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f}));
}

TEST(ImageIo, tensorToImageClampsAndGrey) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const ImageRGBA8 grey = tensorToImage({-1.0f, 0.5f, 2.0f, nan}, 1, 2, 2);
    ASSERT_EQ(grey.pixels.size(), 16u);
    const uint8_t expected[] = {0, 128, 255, 0};
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(grey.pixels[i * 4 + 0], expected[i]);
        EXPECT_EQ(grey.pixels[i * 4 + 1], expected[i]);
        EXPECT_EQ(grey.pixels[i * 4 + 2], expected[i]);
        EXPECT_EQ(grey.pixels[i * 4 + 3], 255);
    }
}

TEST(ImageIo, tensorToImageRejectsBadShapes) {
    EXPECT_THROW(tensorToImage(std::vector<float>(8), 2, 2, 2), std::invalid_argument);
    EXPECT_THROW(tensorToImage(std::vector<float>(5), 1, 2, 2), std::invalid_argument);
}

TEST(ImageIo, resizeChangesSize) {
    ImageRGBA8 image;
    image.width = 4;
    image.height = 4;
    for (int i = 0; i < 16; ++i) {
        image.pixels.insert(image.pixels.end(), {40, 120, 200, 255});
    }
    const ImageRGBA8 resized = resizeImage(image, 3, 2);
    EXPECT_EQ(resized.width, 3u);
    EXPECT_EQ(resized.height, 2u);
    ASSERT_EQ(resized.pixels.size(), 3u * 2 * 4);
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_NEAR(resized.pixels[i * 4 + 0], 40, 1);
        EXPECT_NEAR(resized.pixels[i * 4 + 1], 120, 1);
        EXPECT_NEAR(resized.pixels[i * 4 + 2], 200, 1);
    }
}

TEST(ImageIo, pngRoundTrip) {
    const auto dir = std::filesystem::temp_directory_path() / "klartraum_studio_image_io";
    std::filesystem::create_directories(dir);
    const auto path = dir / "gradient.png";
    const ImageRGBA8 image = gradient(16, 9);
    savePng(image, path);
    const ImageRGBA8 loaded = loadImage(path);
    EXPECT_EQ(loaded.width, 16u);
    EXPECT_EQ(loaded.height, 9u);
    EXPECT_EQ(loaded.pixels, image.pixels);
    std::filesystem::remove_all(dir);
}

TEST(ImageIo, loadMissingFileThrows) {
    try {
        loadImage("does/not/exist.png");
        FAIL() << "expected an exception";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("does/not/exist.png"), std::string::npos);
    }
}
