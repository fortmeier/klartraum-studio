/**
 * TESTS:
 * - onnxPinsFollowTheModel: an ONNX Model node has one tensor pin per model input and output;
 *   setOnnxPins renames them and drops links to pins that no longer exist
 * - onnxPinsAreSaved: pin names survive toJson/fromJson; files without them keep one input and one
 *   output
 * - onnxInputTypesAreChecked: a model input of another shape or element type is reported on the
 *   model node, naming the input; outputs carry the model's types
 * - layerTypes: binary layers broadcast A and B (or b) and need float32; unary layers keep the
 *   shape; shapes that do not broadcast are reported
 * - runLayers: Image File -> Multiply (b 0.5) -> Add (b 0.25) computes 0.5x + 0.25, and Multiply
 *   with B connected multiplies two tensors (GPU)
 * - runOnnxWithTwoInputsAndOutputs: a generated model with two inputs and two outputs (Add, Relu)
 *   runs with each output feeding its own Preview (GPU)
 **/

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>

#include "klartraum/headless_frontend.hpp"
#include "klartraum/klartraum_core.hpp"
#include "klartraum/vulkan_helpers.hpp"
#include "onnx.pb.h"

#include "studio/graph_compiler.hpp"
#include "studio/graph_runner.hpp"
#include "studio/graph_serialization.hpp"

using namespace kstudio;

namespace {

const std::filesystem::path kRoot = KLARTRAUM_SOURCE_DIR;
const std::string kImage = (kRoot / "data/lantern.jpg").string();

PinRef out(int node, int slot = 0) { return {node, PinDirection::Output, slot}; }
PinRef in(int node, int slot = 0) { return {node, PinDirection::Input, slot}; }

bool hasErrorOn(const std::vector<Diagnostic>& diagnostics, int node, std::string_view text = {}) {
    return std::any_of(diagnostics.begin(), diagnostics.end(), [&](const Diagnostic& d) {
        return d.node == node && d.severity == Severity::Error && d.message.find(text) != std::string::npos;
    });
}

OnnxModelInfo modelInfo(std::vector<OnnxTensorDesc> inputs, std::vector<OnnxTensorDesc> outputs) {
    OnnxModelInfo info;
    info.inputs = std::move(inputs);
    info.outputs = std::move(outputs);
    return info;
}

OnnxInfoProvider provider(const std::map<std::string, OnnxModelInfo>& models) {
    return [models](const std::string& path, std::string& error) -> std::shared_ptr<const OnnxModelInfo> {
        auto it = models.find(path);
        if (it == models.end()) {
            error = "no such model";
            return nullptr;
        }
        return std::make_shared<OnnxModelInfo>(it->second);
    };
}

void setValueInfo(onnx::ValueInfoProto* value, const std::string& name, const std::vector<int64_t>& shape) {
    value->set_name(name);
    auto* tensor = value->mutable_type()->mutable_tensor_type();
    tensor->set_elem_type(onnx::TensorProto::FLOAT);
    for (int64_t dim : shape) {
        tensor->mutable_shape()->add_dim()->set_dim_value(dim);
    }
}

// sum = a + b, rectified = relu(a); all 1x3xSxS.
std::filesystem::path writeTwoByTwoModel(const std::filesystem::path& path, int64_t size) {
    onnx::ModelProto model;
    model.set_ir_version(8);
    model.add_opset_import()->set_version(17);
    auto* graph = model.mutable_graph();
    graph->set_name("two_by_two");
    const std::vector<int64_t> shape{1, 3, size, size};
    setValueInfo(graph->add_input(), "a", shape);
    setValueInfo(graph->add_input(), "b", shape);
    setValueInfo(graph->add_output(), "sum", shape);
    setValueInfo(graph->add_output(), "rectified", shape);
    auto* add = graph->add_node();
    add->set_name("add");
    add->set_op_type("Add");
    add->add_input("a");
    add->add_input("b");
    add->add_output("sum");
    auto* relu = graph->add_node();
    relu->set_name("relu");
    relu->set_op_type("Relu");
    relu->add_input("a");
    relu->add_output("rectified");
    std::ofstream file(path, std::ios::binary);
    if (!model.SerializeToOstream(&file)) {
        throw std::runtime_error("cannot write " + path.string());
    }
    return path;
}

class TensorPinsRunTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!std::filesystem::exists(kImage)) {
            GTEST_SKIP() << "sample not found: " << kImage;
        }
        klartraum::setAssetRoot(kRoot.string());
        dir = std::filesystem::temp_directory_path() / "klartraum_studio_tensor_pins";
        std::filesystem::create_directories(dir);
        frontend = std::make_unique<klartraum::HeadlessFrontend>();
        context.resolveInput = [](const std::string& path) { return std::filesystem::path(path); };
        context.resolveOutput = [this](const std::string& path) { return dir / path; };
        context.onnxInfo = [this](const std::string& path, std::string& error) { return onnx.get(path, &error); };
    }

    void TearDown() override {
        frontend.reset();
        klartraum::setAssetRoot("");
        std::filesystem::remove_all(dir);
    }

    RunResult run(const Graph& graph) {
        const CompilePlan plan = planGraph(graph, context.onnxInfo);
        for (const auto& d : plan.diagnostics) {
            EXPECT_NE(d.severity, Severity::Error) << d.message;
        }
        if (!plan.run) {
            throw std::runtime_error("nothing to run");
        }
        return runGraph(frontend->getKlartraumEngine().getVulkanContext(), graph, *plan.run, context);
    }

    // The 8-bit image of f(x) for the lantern resized to 16x16.
    static ImageRGBA8 expected(const std::function<float(float)>& f) {
        auto values = imageToTensor(resizeImage(loadImage(kImage), 16, 16));
        for (float& v : values) {
            v = f(v);
        }
        return tensorToImage(values, 3, 16, 16);
    }

    static void expectSameImage(const ImageRGBA8& actual, const ImageRGBA8& expected) {
        ASSERT_EQ(actual.pixels.size(), expected.pixels.size());
        int maximum = 0;
        for (size_t i = 0; i < actual.pixels.size(); ++i) {
            maximum = std::max(maximum, std::abs(int(actual.pixels[i]) - int(expected.pixels[i])));
        }
        EXPECT_LE(maximum, 1);
    }

    int imageFile(Graph& graph) {
        const int id = graph.addNode(NodeKind::ImageFile);
        auto& p = graph.findNode(id)->as<ImageFileParams>();
        p.path = kImage;
        p.width = 16;
        p.height = 16;
        return id;
    }

    std::unique_ptr<klartraum::HeadlessFrontend> frontend;
    std::filesystem::path dir;
    OnnxInfoCache onnx;
    RunContext context;
};

} // namespace

TEST(TensorPins, onnxPinsFollowTheModel) {
    Graph graph;
    const int image = graph.addNode(NodeKind::ImageFile);
    const int model = graph.addNode(NodeKind::OnnxModel);
    const int preview = graph.addNode(NodeKind::Preview);
    ASSERT_EQ(graph.inputPins(*graph.findNode(model)).size(), 1u);
    ASSERT_FALSE(graph.connect(out(image), in(model)));
    ASSERT_FALSE(graph.connect(out(model), in(preview)));

    EXPECT_EQ(graph.setOnnxPins(model, {"a", "b"}, {"sum", "rectified"}), 0);
    const auto inputs = graph.inputPins(*graph.findNode(model));
    ASSERT_EQ(inputs.size(), 2u);
    EXPECT_EQ(inputs[1].name, "b");
    EXPECT_EQ(inputs[1].type, PinType::Tensor);
    EXPECT_EQ(graph.outputPins(*graph.findNode(model))[1].name, "rectified");
    EXPECT_FALSE(graph.connect(out(model, 1), in(preview)));

    // One output left: the link from the second one goes.
    const uint64_t revision = graph.revision();
    EXPECT_EQ(graph.setOnnxPins(model, {"a"}, {"sum"}), 1);
    EXPECT_GT(graph.revision(), revision);
    EXPECT_EQ(graph.inputLink(preview, 0), nullptr);
    EXPECT_NE(graph.inputLink(model, 0), nullptr);
    EXPECT_EQ(graph.setOnnxPins(model, {"a"}, {"sum"}), 0);
    EXPECT_THROW(graph.setOnnxPins(image, {}, {}), std::logic_error);
}

TEST(TensorPins, onnxPinsAreSaved) {
    Graph graph;
    const int model = graph.addNode(NodeKind::OnnxModel);
    graph.setOnnxPins(model, {"input_ids", "attention_mask"}, {"last_hidden_state", "pooler_output"});
    const Graph loaded = fromJson(toJson(graph));
    EXPECT_EQ(loaded.findNode(model)->params, graph.findNode(model)->params);

    const Graph old = fromJson(R"({"format": "klartraum-studio-graph", "version": 1,
        "nodes": [{"id": 1, "kind": "onnx_model", "params": {"path": "m.onnx"}}], "links": []})");
    const auto& p = old.findNode(1)->as<OnnxModelParams>();
    EXPECT_EQ(p.inputs, std::vector<std::string>{"input"});
    EXPECT_EQ(p.outputs, std::vector<std::string>{"output"});
}

TEST(TensorPins, onnxInputTypesAreChecked) {
    Graph graph;
    const int prompt = graph.addNode(NodeKind::Prompt);
    const int image = graph.addNode(NodeKind::ImageFile);
    const int model = graph.addNode(NodeKind::OnnxModel);
    graph.findNode(model)->as<OnnxModelParams>().path = "m";
    graph.setOnnxPins(model, {"ids", "pixels"}, {"out"});
    graph.connect(out(image), in(model, 0));   // float image into int64 ids
    graph.connect(out(prompt, 0), in(model, 1));  // int64 ids into float pixels
    const auto models = provider({{"m", modelInfo({{"ids", {2, 77}, ElementType::Int64}, {"pixels", {1, 3, 128, 128}}},
                                                  {{"out", {1, 10}}})}});
    const ShapeInference result = inferTensorShapes(graph, models);
    EXPECT_TRUE(hasErrorOn(result.diagnostics, model, "Input 'ids' expects 2x77 int64 but gets 1x3x128x128"));
    EXPECT_TRUE(hasErrorOn(result.diagnostics, model, "Input 'pixels' expects 1x3x128x128 but gets 2x77 int64"));
    EXPECT_EQ(result.types.at({model, 0}), (TensorType{{1, 10}}));

    // Pins that do not match the model are reported, too.
    graph.setOnnxPins(model, {"ids"}, {"out"});
    EXPECT_TRUE(hasErrorOn(inferTensorShapes(graph, models).diagnostics, model, "pins do not match"));
}

TEST(TensorPins, layerTypes) {
    Graph graph;
    const int image = graph.addNode(NodeKind::ImageFile);  // 1x3x128x128
    const int noise = graph.addNode(NodeKind::LatentNoise);  // 1x4x64x64
    const int scale = graph.addNode(NodeKind::Multiply);
    const int sum = graph.addNode(NodeKind::Add);
    const int bad = graph.addNode(NodeKind::Subtract);
    const int soft = graph.addNode(NodeKind::Softmax);
    graph.connect(out(image), in(scale, 0));  // B unconnected: b
    graph.connect(out(scale), in(sum, 0));
    graph.connect(out(image), in(sum, 1));
    graph.connect(out(image), in(bad, 0));
    graph.connect(out(noise), in(bad, 1));
    graph.connect(out(sum), in(soft));
    const ShapeInference result = inferTensorShapes(graph, {});
    EXPECT_EQ(result.types.at({scale, 0}), (TensorType{{1, 3, 128, 128}}));
    EXPECT_EQ(result.types.at({sum, 0}), (TensorType{{1, 3, 128, 128}}));
    EXPECT_EQ(result.types.at({soft, 0}), (TensorType{{1, 3, 128, 128}}));
    EXPECT_TRUE(hasErrorOn(result.diagnostics, bad, "cannot be broadcast"));
    EXPECT_FALSE(result.types.contains({bad, 0}));
    EXPECT_FALSE(hasErrorOn(result.diagnostics, scale));
}

TEST_F(TensorPinsRunTest, runLayers) {
    Graph graph;
    const int image = imageFile(graph);
    const int multiply = graph.addNode(NodeKind::Multiply);
    const int add = graph.addNode(NodeKind::Add);
    const int squared = graph.addNode(NodeKind::Multiply);
    const int preview = graph.addNode(NodeKind::Preview);
    const int squaredPreview = graph.addNode(NodeKind::Preview);
    graph.findNode(multiply)->as<BinaryLayerParams>().b = 0.5f;
    graph.findNode(add)->as<BinaryLayerParams>().b = 0.25f;
    graph.connect(out(image), in(multiply, 0));
    graph.connect(out(multiply), in(add, 0));
    graph.connect(out(add), in(preview));
    graph.connect(out(image), in(squared, 0));
    graph.connect(out(image), in(squared, 1));
    graph.connect(out(squared), in(squaredPreview));

    const RunResult result = run(graph);
    expectSameImage(result.images.at(preview), expected([](float x) { return 0.5f * x + 0.25f; }));
    expectSameImage(result.images.at(squaredPreview), expected([](float x) { return x * x; }));
    // The layers are elements of the run graph, owned by their nodes.
    EXPECT_TRUE(std::any_of(result.compiled.nodes.begin(), result.compiled.nodes.end(),
                            [&](const ElementNode& e) { return e.owner == add && e.name == "Add"; }));
}

TEST_F(TensorPinsRunTest, runOnnxWithTwoInputsAndOutputs) {
    const auto modelPath = writeTwoByTwoModel(dir / "two_by_two.onnx", 16);
    Graph graph;
    const int image = imageFile(graph);
    const int negate = graph.addNode(NodeKind::Multiply);
    const int model = graph.addNode(NodeKind::OnnxModel);
    const int sumPreview = graph.addNode(NodeKind::Preview);
    const int reluPreview = graph.addNode(NodeKind::Preview);
    graph.findNode(negate)->as<BinaryLayerParams>().b = -0.5f;
    graph.findNode(model)->as<OnnxModelParams>().path = modelPath.string();
    graph.setOnnxPins(model, {"a", "b"}, {"sum", "rectified"});
    graph.connect(out(image), in(negate, 0));
    graph.connect(out(image), in(model, 0));
    graph.connect(out(negate), in(model, 1));
    graph.connect(out(model, 0), in(sumPreview));
    graph.connect(out(model, 1), in(reluPreview));

    const RunResult result = run(graph);
    expectSameImage(result.images.at(sumPreview), expected([](float x) { return 0.5f * x; }));
    expectSameImage(result.images.at(reluPreview), expected([](float x) { return x; }));
}
