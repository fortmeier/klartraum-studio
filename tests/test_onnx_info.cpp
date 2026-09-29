/**
 * TESTS:
 * - shapeToStringFormats: shapes print as NxCxHxW, the empty shape as scalar
 * - supportedOps: the operators klartraum executes, including Stable Diffusion's, are supported,
 *   others are not
 * - readEncoderInfo: the sample encoder has input 1x3x128x128, output 1x128x16x16, Conv and Relu
 * - readDecoderInfo: the sample decoder maps 1x128x16x16 back to 1x3x128x128
 * - readRejectsBadFiles: missing files and non-ONNX files throw
 * - cacheReportsErrors: the cache returns nullptr with an error for unreadable models and caches good ones
 **/

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

#include "studio/onnx_info.hpp"

using namespace kstudio;

namespace {

const std::filesystem::path kOnnxDir = std::filesystem::path(KLARTRAUM_SOURCE_DIR) / "data/onnx";

} // namespace

TEST(OnnxInfo, shapeToStringFormats) {
    EXPECT_EQ(shapeToString({1, 3, 128, 128}), "1x3x128x128");
    EXPECT_EQ(shapeToString({}), "scalar");
}

TEST(OnnxInfo, supportedOps) {
    // The autoencoder's operators and those Stable Diffusion 1.5 needs.
    for (const char* op : {"Conv", "ConvTranspose", "Relu", "Reshape", "Transpose", "MatMul", "Softmax", "Gemm",
                           "InstanceNormalization", "LayerNormalization", "Gather", "Resize", "Erf"}) {
        EXPECT_TRUE(isSupportedOnnxOp(op)) << op;
    }
    EXPECT_FALSE(isSupportedOnnxOp("LSTM"));
    EXPECT_FALSE(isSupportedOnnxOp("NonMaxSuppression"));
}

TEST(OnnxInfo, readEncoderInfo) {
    const OnnxModelInfo info = readOnnxModelInfo(kOnnxDir / "simple_encoder.onnx");
    ASSERT_EQ(info.inputs.size(), 1u);
    EXPECT_EQ(info.inputs[0].name, "input");
    EXPECT_EQ(info.inputs[0].shape, (TensorShape{1, 3, 128, 128}));
    ASSERT_EQ(info.outputs.size(), 1u);
    EXPECT_EQ(info.outputs[0].name, "output");
    EXPECT_EQ(info.outputs[0].shape, (TensorShape{1, 128, 16, 16}));
    EXPECT_EQ(info.opTypes, (std::vector<std::string>{"Conv", "Relu"}));
    EXPECT_TRUE(info.unsupportedOps.empty());
}

TEST(OnnxInfo, readDecoderInfo) {
    const OnnxModelInfo info = readOnnxModelInfo(kOnnxDir / "simple_decoder.onnx");
    ASSERT_EQ(info.inputs.size(), 1u);
    EXPECT_EQ(info.inputs[0].shape, (TensorShape{1, 128, 16, 16}));
    ASSERT_EQ(info.outputs.size(), 1u);
    EXPECT_EQ(info.outputs[0].shape, (TensorShape{1, 3, 128, 128}));
    EXPECT_EQ(info.opTypes, (std::vector<std::string>{"ConvTranspose", "Relu"}));
}

TEST(OnnxInfo, readRejectsBadFiles) {
    EXPECT_THROW(readOnnxModelInfo("does/not/exist.onnx"), std::runtime_error);

    const auto path = std::filesystem::temp_directory_path() / "klartraum_studio_not_onnx.onnx";
    std::ofstream(path) << "this is not a protobuf";
    EXPECT_THROW(readOnnxModelInfo(path), std::runtime_error);
    std::filesystem::remove(path);
}

TEST(OnnxInfo, cacheReportsErrors) {
    OnnxInfoCache cache;
    std::string error;
    EXPECT_EQ(cache.get("does/not/exist.onnx", &error), nullptr);
    EXPECT_FALSE(error.empty());

    const auto first = cache.get(kOnnxDir / "simple_encoder.onnx", &error);
    ASSERT_NE(first, nullptr);
    EXPECT_TRUE(error.empty());
    EXPECT_EQ(cache.get(kOnnxDir / "simple_encoder.onnx"), first);
}
