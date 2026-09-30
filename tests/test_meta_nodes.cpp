/**
 * TESTS:
 * - builtinDefinitions: the Text Encoder and VAE Decoder definitions exist, expose their pins and
 *   parameters, and give meta nodes tensor pins
 * - flattenReplacesMetaNodes: a flattened graph has no meta nodes; top-level nodes keep their ids,
 *   inner nodes are titled "<meta> / <inner>", belong to the meta node and are linked through its pins
 * - flattenAppliesExposedValues: a meta node's parameter values reach the inner nodes
 * - flattenNestedDefinitions: a definition containing a meta node flattens recursively, and pins
 *   resolve through both levels
 * - flattenReportsProblems: unknown definitions and definitions that contain themselves are
 *   reported on the top-level node
 * - groupAndUngroup: grouping replaces nodes by a meta node wired like them, with one exposed input
 *   per outside source (feeding every inner pin it fed) and one output per inner output; ungrouping
 *   restores equivalent nodes and links
 * - groupRejectsBadInput: empty groups, unknown nodes and taken names throw
 * - definitionsAreSaved: definitions (nested ones first) and meta node values survive
 *   toJson/fromJson; files from before meta nodes load; the former Text Encoder and VAE Decoder
 *   kinds load as meta nodes
 * - updateMetaLinksByName: when a definition's pins change, instance links follow the pin names;
 *   links to removed pins are dropped
 * - pruneInterface: exposed pins and parameters of removed inner nodes are dropped
 * - planMapsInnerDiagnostics: problems inside a meta node are reported on it with the inner node's
 *   title; tensor types of its outputs are found through flatOutput
 * - runGroupedLayers: a meta node grouping Multiply and Add computes the same as the ungrouped
 *   nodes, and its compiled elements map to it (GPU)
 **/

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>

#include "klartraum/headless_frontend.hpp"
#include "klartraum/klartraum_core.hpp"
#include "klartraum/vulkan_helpers.hpp"

#include "studio/graph_compiler.hpp"
#include "studio/graph_runner.hpp"
#include "studio/graph_serialization.hpp"
#include "studio/meta_nodes.hpp"

using namespace kstudio;

namespace {

const std::filesystem::path kRoot = KLARTRAUM_SOURCE_DIR;
const std::string kImage = (kRoot / "data/lantern.jpg").string();

PinRef out(int node, int slot = 0) { return {node, PinDirection::Output, slot}; }
PinRef in(int node, int slot = 0) { return {node, PinDirection::Input, slot}; }

bool hasError(const std::vector<Diagnostic>& diagnostics, int node, std::string_view text = {}) {
    return std::any_of(diagnostics.begin(), diagnostics.end(), [&](const Diagnostic& d) {
        return d.node == node && d.severity == Severity::Error && d.message.find(text) != std::string::npos;
    });
}

const Node* findTitled(const Graph& graph, std::string_view title) {
    for (const auto& node : graph.nodes()) {
        if (node.title == title) {
            return &node;
        }
    }
    return nullptr;
}

// A definition "scale_offset": x * b + c with exposed "Scale" and "Offset".
std::shared_ptr<MetaDefinition> scaleOffset(const std::string& name = "scale_offset") {
    auto d = std::make_shared<MetaDefinition>();
    d->name = name;
    d->title = "Scale Offset";
    const int mul = d->graph.addNode(NodeKind::Multiply);
    const int add = d->graph.addNode(NodeKind::Add);
    d->graph.findNode(mul)->title = "Scale";
    d->graph.findNode(add)->title = "Offset";
    d->graph.connect(out(mul), in(add));
    d->inputs = {{"X", {in(mul)}}};
    d->outputs = {{"Y", {out(add)}}};
    d->params = {{"Scale", mul, "b"}, {"Offset", add, "b"}};
    return d;
}

} // namespace

TEST(MetaNodes, builtinDefinitions) {
    const auto& builtins = builtinDefinitions();
    ASSERT_TRUE(builtins.contains(kTextEncoderDefinition));
    ASSERT_TRUE(builtins.contains(kVaeDecoderDefinition));
    const MetaDefinition& vae = *builtins.at(kVaeDecoderDefinition);
    EXPECT_TRUE(vae.builtin);
    EXPECT_EQ(vae.inputs.size(), 1u);
    EXPECT_EQ(vae.outputs.size(), 1u);
    EXPECT_EQ(vae.params.size(), 2u);

    Graph graph;
    const int encoder = graph.addMetaNode(kTextEncoderDefinition);
    EXPECT_EQ(graph.findNode(encoder)->title, "Text Encoder");
    const auto inputs = graph.inputPins(*graph.findNode(encoder));
    ASSERT_EQ(inputs.size(), 2u);
    EXPECT_EQ(inputs[0].name, "Ids");
    EXPECT_EQ(inputs[1].name, "Mask");
    EXPECT_EQ(inputs[0].type, PinType::Tensor);
    EXPECT_FALSE(inputs[0].optional);
    EXPECT_EQ(graph.outputPins(*graph.findNode(encoder)).at(0).name, "Embeddings");
}

TEST(MetaNodes, flattenReplacesMetaNodes) {
    Graph graph;
    const int image = graph.addNode(NodeKind::ImageFile);
    const int decoder = graph.addMetaNode(kVaeDecoderDefinition);
    const int preview = graph.addNode(NodeKind::Preview);
    graph.connect(out(image), in(decoder));
    graph.connect(out(decoder), in(preview));

    const FlatGraph flat = flatten(graph);
    EXPECT_TRUE(flat.diagnostics.empty());
    EXPECT_TRUE(std::none_of(flat.graph.nodes().begin(), flat.graph.nodes().end(),
                             [](const Node& n) { return n.kind == NodeKind::Meta; }));
    EXPECT_EQ(flat.graph.nodes().size(), 2u + 4u);
    EXPECT_EQ(flat.graph.findNode(image)->kind, NodeKind::ImageFile);
    EXPECT_EQ(flat.graph.findNode(preview)->kind, NodeKind::Preview);
    EXPECT_EQ(flat.graph.findNode(decoder), nullptr);

    const Node* unscale = findTitled(flat.graph, "VAE Decoder / Unscale latents");
    const Node* offset = findTitled(flat.graph, "VAE Decoder / Offset");
    ASSERT_NE(unscale, nullptr);
    ASSERT_NE(offset, nullptr);
    EXPECT_GT(unscale->id, preview);
    EXPECT_EQ(flat.top(unscale->id), decoder);
    EXPECT_EQ(flat.top(image), image);
    EXPECT_EQ(flat.graph.inputLink(unscale->id, 0)->fromNode, image);
    EXPECT_EQ(flat.graph.inputLink(preview, 0)->fromNode, offset->id);
    EXPECT_EQ(flat.flatOutput({decoder, 0}), (OutputPin{offset->id, 0}));
    EXPECT_EQ(flat.flatOutput({image, 0}), (OutputPin{image, 0}));
    EXPECT_TRUE(flat.covers({unscale->id}, decoder));
    EXPECT_FALSE(flat.covers({unscale->id}, image));
}

TEST(MetaNodes, flattenAppliesExposedValues) {
    Graph graph;
    const int decoder = graph.addMetaNode(kVaeDecoderDefinition);
    auto& values = graph.findNode(decoder)->as<MetaParams>().values;
    values["Model"] = "\"models/vae.onnx\"";
    values["Latent scale"] = "2.5";
    const FlatGraph flat = flatten(graph);
    EXPECT_EQ(findTitled(flat.graph, "VAE Decoder / Decoder")->as<OnnxModelParams>().path, "models/vae.onnx");
    EXPECT_FLOAT_EQ(findTitled(flat.graph, "VAE Decoder / Unscale latents")->as<BinaryLayerParams>().b, 2.5f);
    // The definition itself is unchanged.
    EXPECT_TRUE(builtinDefinitions().at(kVaeDecoderDefinition)->graph.findNode(2)->as<OnnxModelParams>().path.empty());

    values["Nope"] = "1";
    EXPECT_TRUE(hasError(flatten(graph).diagnostics, decoder, "no parameter 'Nope'"));
    values.erase("Nope");
    values["Latent scale"] = "\"text\"";
    EXPECT_TRUE(hasError(flatten(graph).diagnostics, decoder, "invalid value"));
}

TEST(MetaNodes, flattenNestedDefinitions) {
    Graph graph;
    graph.setDefinition(scaleOffset());
    // outer = scale_offset -> built-in VAE decoder
    auto outer = std::make_shared<MetaDefinition>();
    outer->name = "outer";
    outer->title = "Outer";
    {
        Graph inner;
        inner.setDefinition(graph.definitions().at("scale_offset"));
        const int so = inner.addMetaNode("scale_offset");
        const int vae = inner.addMetaNode(kVaeDecoderDefinition);
        inner.connect(out(so), in(vae));
        inner.removeDefinition("scale_offset");
        outer->graph = std::move(inner);
        outer->inputs = {{"In", {in(so)}}};
        outer->outputs = {{"Out", {out(vae)}}};
    }
    graph.setDefinition(outer);
    const int image = graph.addNode(NodeKind::ImageFile);
    const int meta = graph.addMetaNode("outer");
    const int preview = graph.addNode(NodeKind::Preview);
    ASSERT_EQ(graph.inputPins(*graph.findNode(meta)).size(), 1u);
    ASSERT_FALSE(graph.connect(out(image), in(meta)));
    ASSERT_FALSE(graph.connect(out(meta), in(preview)));

    const FlatGraph flat = flatten(graph);
    EXPECT_TRUE(flat.diagnostics.empty());
    EXPECT_EQ(flat.graph.nodes().size(), 2u + 2u + 4u);
    const Node* scale = findTitled(flat.graph, "Outer / Scale Offset / Scale");
    const Node* offset = findTitled(flat.graph, "Outer / VAE Decoder / Offset");
    ASSERT_NE(scale, nullptr);
    ASSERT_NE(offset, nullptr);
    EXPECT_EQ(flat.top(scale->id), meta);
    EXPECT_EQ(flat.top(offset->id), meta);
    EXPECT_EQ(flat.graph.inputLink(scale->id, 0)->fromNode, image);
    EXPECT_EQ(flat.graph.inputLink(preview, 0)->fromNode, offset->id);
}

TEST(MetaNodes, flattenReportsProblems) {
    Graph graph;
    const int unknown = graph.addNode(NodeKind::Meta);
    graph.findNode(unknown)->as<MetaParams>().definition = "missing";
    EXPECT_TRUE(hasError(flatten(graph).diagnostics, unknown, "Unknown meta node definition 'missing'"));
    EXPECT_TRUE(hasError(graph.validate(), unknown, "missing"));

    auto loop = std::make_shared<MetaDefinition>();
    loop->name = "loop";
    loop->title = "Loop";
    loop->graph.addNode(NodeKind::Meta);
    loop->graph.nodes()[0].as<MetaParams>().definition = "loop";
    graph.setDefinition(loop);
    const int looping = graph.addMetaNode("loop");
    EXPECT_TRUE(hasError(flatten(graph).diagnostics, looping, "contains itself"));
}

TEST(MetaNodes, groupAndUngroup) {
    Graph graph;
    const int image = graph.addNode(NodeKind::ImageFile, {0.0f, 0.0f});
    const int mul = graph.addNode(NodeKind::Multiply, {100.0f, 0.0f});
    const int add = graph.addNode(NodeKind::Add, {200.0f, 0.0f});
    const int preview = graph.addNode(NodeKind::Preview, {300.0f, 0.0f});
    const int writer = graph.addNode(NodeKind::ImageFileWriter, {300.0f, 100.0f});
    graph.findNode(mul)->as<BinaryLayerParams>().b = 0.5f;
    graph.connect(out(image), in(mul, 0));
    graph.connect(out(image), in(add, 1));  // the same source into a second inner pin
    graph.connect(out(mul), in(add, 0));
    graph.connect(out(add), in(preview));
    graph.connect(out(add), in(writer));

    const int meta = groupNodes(graph, {mul, add}, "affine", "Affine");
    EXPECT_EQ(graph.findNode(mul), nullptr);
    EXPECT_EQ(graph.findNode(meta)->title, "Affine");
    EXPECT_FLOAT_EQ(graph.findNode(meta)->position.x, 150.0f);
    const MetaDefinition& d = *graph.definitions().at("affine");
    ASSERT_EQ(d.inputs.size(), 1u);
    EXPECT_EQ(d.inputs[0].name, "A");
    EXPECT_EQ(d.inputs[0].targets, (std::vector<PinRef>{in(mul, 0), in(add, 1)}));
    ASSERT_EQ(d.outputs.size(), 1u);
    EXPECT_EQ(d.graph.findNode(mul)->position.x, 0.0f);
    EXPECT_EQ(graph.inputLink(meta, 0)->fromNode, image);
    EXPECT_EQ(graph.inputLink(preview, 0)->fromNode, meta);
    EXPECT_EQ(graph.inputLink(writer, 0)->fromNode, meta);
    EXPECT_FALSE(hasError(planGraph(graph).diagnostics, meta));

    const auto added = ungroupNode(graph, meta);
    ASSERT_EQ(added.size(), 2u);
    EXPECT_EQ(graph.findNode(meta), nullptr);
    const int newMul = graph.findNode(added[0])->kind == NodeKind::Multiply ? added[0] : added[1];
    const int newAdd = newMul == added[0] ? added[1] : added[0];
    EXPECT_FLOAT_EQ(graph.findNode(newMul)->as<BinaryLayerParams>().b, 0.5f);
    EXPECT_EQ(graph.inputLink(newMul, 0)->fromNode, image);
    EXPECT_EQ(graph.inputLink(newAdd, 1)->fromNode, image);
    EXPECT_EQ(graph.inputLink(newAdd, 0)->fromNode, newMul);
    EXPECT_EQ(graph.inputLink(preview, 0)->fromNode, newAdd);
    EXPECT_EQ(graph.inputLink(writer, 0)->fromNode, newAdd);
    EXPECT_EQ(graph.links().size(), 5u);
}

TEST(MetaNodes, groupRejectsBadInput) {
    Graph graph;
    const int node = graph.addNode(NodeKind::Relu);
    EXPECT_THROW(groupNodes(graph, {}, "g", "G"), std::invalid_argument);
    EXPECT_THROW(groupNodes(graph, {node, 99}, "g", "G"), std::invalid_argument);
    EXPECT_THROW(groupNodes(graph, {node}, kVaeDecoderDefinition, "G"), std::invalid_argument);
    EXPECT_THROW(ungroupNode(graph, node), std::invalid_argument);
    EXPECT_EQ(uniqueDefinitionName(graph, "My Group!"), "my_group_");
    graph.setDefinition(scaleOffset("group"));
    EXPECT_EQ(uniqueDefinitionName(graph, "group"), "group_2");
}

TEST(MetaNodes, definitionsAreSaved) {
    Graph graph;
    graph.setDefinition(scaleOffset());
    auto outer = std::make_shared<MetaDefinition>();
    outer->name = "a_outer";  // sorts before its dependency, which must still load first
    outer->title = "Outer";
    {
        Graph inner;
        inner.setDefinition(graph.definitions().at("scale_offset"));
        const int so = inner.addMetaNode("scale_offset");
        inner.findNode(so)->as<MetaParams>().values["Scale"] = "3.0";
        inner.removeDefinition("scale_offset");
        outer->graph = std::move(inner);
        outer->inputs = {{"In", {in(so)}}};
        outer->outputs = {{"Out", {out(so)}}};
        outer->params = {{"Title", so, "definition"}};
    }
    graph.setDefinition(outer);
    const int meta = graph.addMetaNode("a_outer");
    graph.findNode(meta)->as<MetaParams>().values["Title"] = "\"scale_offset\"";

    const std::string text = toJson(graph);
    EXPECT_NE(text.find("\"version\": 2"), std::string::npos);
    const Graph loaded = fromJson(text);
    EXPECT_EQ(toJson(loaded), text);
    ASSERT_EQ(loaded.definitions().size(), 2u);
    EXPECT_EQ(loaded.definitions().at("a_outer")->inputs, outer->inputs);
    EXPECT_EQ(loaded.definitions().at("a_outer")->params, outer->params);
    EXPECT_EQ(loaded.findNode(meta)->params, graph.findNode(meta)->params);

    // Before meta nodes, the encoder and decoder were kinds with a path.
    const Graph old = fromJson(R"({"format": "klartraum-studio-graph", "version": 1,
        "nodes": [{"id": 1, "kind": "sd_vae_decoder", "title": "VAE", "params": {"path": "vae.onnx"}},
                  {"id": 2, "kind": "sd_text_encoder", "params": {"path": "clip.onnx"}},
                  {"id": 3, "kind": "preview"}],
        "links": [{"from": [1, 0], "to": [3, 0]}]})");
    const Node& vae = *old.findNode(1);
    ASSERT_EQ(vae.kind, NodeKind::Meta);
    EXPECT_EQ(vae.as<MetaParams>().definition, kVaeDecoderDefinition);
    EXPECT_EQ(vae.as<MetaParams>().values.at("Model"), "\"vae.onnx\"");
    EXPECT_EQ(vae.title, "VAE");
    EXPECT_EQ(old.findNode(2)->as<MetaParams>().definition, kTextEncoderDefinition);
    EXPECT_EQ(old.inputLink(3, 0)->fromNode, 1);
}

TEST(MetaNodes, updateMetaLinksByName) {
    Graph graph;
    auto d = std::make_shared<MetaDefinition>();
    d->name = "two";
    d->title = "Two";
    const int a = d->graph.addNode(NodeKind::Relu);
    const int b = d->graph.addNode(NodeKind::Sigmoid);
    d->inputs = {{"A", {in(a)}}, {"B", {in(b)}}};
    d->outputs = {{"A", {out(a)}}, {"B", {out(b)}}};
    graph.setDefinition(d);
    const int image = graph.addNode(NodeKind::ImageFile);
    const int meta = graph.addMetaNode("two");
    const int preview = graph.addNode(NodeKind::Preview);
    graph.connect(out(image), in(meta, 0));
    graph.connect(out(image), in(meta, 1));
    graph.connect(out(meta, 1), in(preview));

    auto changed = std::make_shared<MetaDefinition>(*d);
    changed->inputs = {{"B", {in(b)}}};
    changed->outputs = {{"B", {out(b)}}};
    const MetaDefinition before = *d;
    graph.setDefinition(changed);
    EXPECT_EQ(updateMetaLinks(graph, "two", before), 1);  // input A is gone
    EXPECT_EQ(graph.inputLink(meta, 0)->fromNode, image);
    EXPECT_EQ(graph.inputLink(meta, 1), nullptr);
    EXPECT_EQ(graph.inputLink(preview, 0)->fromSlot, 0);
    EXPECT_EQ(updateMetaLinks(graph, "two", *changed), 0);
}

TEST(MetaNodes, pruneInterface) {
    auto d = scaleOffset();
    d->graph.removeNode(d->params[1].node);  // the Add
    pruneInterface(*d);
    EXPECT_EQ(d->inputs.size(), 1u);
    EXPECT_TRUE(d->outputs.empty());
    ASSERT_EQ(d->params.size(), 1u);
    EXPECT_EQ(d->params[0].name, "Scale");
}

TEST(MetaNodes, planMapsInnerDiagnostics) {
    Graph graph;
    graph.setDefinition(scaleOffset());
    const int noise = graph.addNode(NodeKind::LatentNoise);  // 1x4x64x64
    const int prompt = graph.addNode(NodeKind::Prompt);
    const int good = graph.addMetaNode("scale_offset");
    const int bad = graph.addMetaNode("scale_offset");
    const int preview = graph.addNode(NodeKind::Preview);
    graph.findNode(bad)->title = "Bad";
    graph.connect(out(noise), in(good));
    graph.connect(out(prompt, 0), in(bad));  // int64 into a float layer
    graph.connect(out(good), in(preview));
    const CompilePlan plan = planGraph(graph);
    EXPECT_TRUE(hasError(plan.diagnostics, bad, "Bad / Scale: A must be a float32 tensor"));
    EXPECT_FALSE(hasError(plan.diagnostics, good));
    // A 1x4 latent is not an image; the Preview reports it, the meta node's
    // output type is known through flatOutput.
    EXPECT_TRUE(hasError(plan.diagnostics, preview, "not an image"));
    EXPECT_FALSE(plan.run);
    const ShapeInference types = inferTensorShapes(plan.flat.graph, {});
    EXPECT_EQ(types.types.at(plan.flat.flatOutput({good, 0})), (TensorType{{1, 4, 64, 64}}));
}

TEST(MetaNodes, runGroupedLayers) {
    if (!std::filesystem::exists(kImage)) {
        GTEST_SKIP() << "sample not found: " << kImage;
    }
    klartraum::setAssetRoot(kRoot.string());
    auto frontend = std::make_unique<klartraum::HeadlessFrontend>();
    OnnxInfoCache onnx;
    RunContext context;
    context.resolveInput = [](const std::string& path) { return std::filesystem::path(path); };
    context.resolveOutput = [](const std::string& path) { return std::filesystem::temp_directory_path() / path; };
    context.onnxInfo = [&](const std::string& path, std::string& error) { return onnx.get(path, &error); };

    Graph graph;
    graph.setDefinition(scaleOffset());
    const int image = graph.addNode(NodeKind::ImageFile);
    auto& p = graph.findNode(image)->as<ImageFileParams>();
    p.path = kImage;
    p.width = 16;
    p.height = 16;
    const int meta = graph.addMetaNode("scale_offset");
    auto& values = graph.findNode(meta)->as<MetaParams>().values;
    values["Scale"] = "0.5";
    values["Offset"] = "0.25";
    const int preview = graph.addNode(NodeKind::Preview);
    graph.connect(out(image), in(meta));
    graph.connect(out(meta), in(preview));

    const CompilePlan plan = planGraph(graph, context.onnxInfo);
    ASSERT_TRUE(plan.run.has_value());
    const RunResult result =
        runGraph(frontend->getKlartraumEngine().getVulkanContext(), plan.flat.graph, *plan.run, context);
    auto expected = imageToTensor(resizeImage(loadImage(kImage), 16, 16));
    for (float& v : expected) {
        v = 0.5f * v + 0.25f;
    }
    const ImageRGBA8 reference = tensorToImage(expected, 3, 16, 16);
    const ImageRGBA8& actual = result.images.at(preview);
    ASSERT_EQ(actual.pixels.size(), reference.pixels.size());
    for (size_t i = 0; i < actual.pixels.size(); ++i) {
        EXPECT_LE(std::abs(int(actual.pixels[i]) - int(reference.pixels[i])), 1) << i;
    }
    EXPECT_TRUE(std::any_of(result.compiled.nodes.begin(), result.compiled.nodes.end(), [&](const ElementNode& e) {
        return e.owner >= 0 && plan.flat.top(e.owner) == meta && e.name == "Scale Offset / Offset";
    }));
    frontend.reset();
    klartraum::setAssetRoot("");
}
