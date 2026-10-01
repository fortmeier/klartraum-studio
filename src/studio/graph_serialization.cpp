#include "studio/graph_serialization.hpp"

#include <algorithm>
#include <fstream>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace kstudio {

namespace {

using nlohmann::json;

// Version 2 added meta node definitions.
constexpr int kFormatVersion = 2;

json paramsToJson(const NodeParams& params) {
    return std::visit(
        [](const auto& p) -> json {
            using T = std::decay_t<decltype(p)>;
            if constexpr (std::is_same_v<T, SceneParams>) {
                return {{"path", p.path}, {"flipY", p.flipY}};
            } else if constexpr (std::is_same_v<T, TransformGaussiansParams> || std::is_same_v<T, MakeTransformParams>) {
                return {{"translation", p.translation}, {"rotation", p.rotation}, {"scale", p.scale}};
            } else if constexpr (std::is_same_v<T, NumberParams>) {
                return {{"value", p.value}};
            } else if constexpr (std::is_same_v<T, TimeParams>) {
                return {{"speed", p.speed}};
            } else if constexpr (std::is_same_v<T, SineParams>) {
                return {{"amplitude", p.amplitude}, {"frequency", p.frequency}, {"phase", p.phase}, {"offset", p.offset}};
            } else if constexpr (std::is_same_v<T, CameraParams>) {
                return {{"azimuth", p.azimuth},
                        {"elevation", p.elevation},
                        {"distance", p.distance},
                        {"target", p.target},
                        {"up", p.up == UpAxis::Y ? "y" : "z"}};
            } else if constexpr (std::is_same_v<T, SplattingParams>) {
                return {{"backend", backendName(p.backend)},
                        {"spreadMultiplier", p.spreadMultiplier},
                        {"maxMod", p.maxMod},
                        {"numSortWGsCap", p.numSortWGsCap},
                        {"splatTileX", p.splatTileX},
                        {"splatTileY", p.splatTileY},
                        {"shDegree", p.shDegree},
                        {"alphaCullThreshold", p.alphaCullThreshold},
                        {"useMeshShader", p.useMeshShader}};
            } else if constexpr (std::is_same_v<T, OffscreenTargetParams>) {
                return {{"width", p.width}, {"height", p.height}, {"clear", p.clear}};
            } else if constexpr (std::is_same_v<T, SwapchainTargetParams>) {
                return {{"clear", p.clear}};
            } else if constexpr (std::is_same_v<T, ClearImageParams>) {
                return {{"color", p.color}};
            } else if constexpr (std::is_same_v<T, CompositeParams>) {
                return {{"mode", compositeModeName(p.mode)}, {"fit", compositeFitName(p.fit)}};
            } else if constexpr (std::is_same_v<T, DrawBasicsParams>) {
                return {{"shape", drawBasicsShapeName(p.shape)}};
            } else if constexpr (std::is_same_v<T, ImageFileParams>) {
                return {{"path", p.path}, {"width", p.width}, {"height", p.height}};
            } else if constexpr (std::is_same_v<T, ResampleParams>) {
                return {{"width", p.width}, {"height", p.height}, {"filter", filterName(p.filter)}};
            } else if constexpr (std::is_same_v<T, OnnxModelParams>) {
                return {{"path", p.path}, {"inputs", p.inputs}, {"outputs", p.outputs}};
            } else if constexpr (std::is_same_v<T, ImageFileWriterParams>) {
                return {{"path", p.path}};
            } else if constexpr (std::is_same_v<T, BinaryLayerParams>) {
                return {{"b", p.b}};
            } else if constexpr (std::is_same_v<T, MetaParams>) {
                json values = json::object();
                for (const auto& [name, value] : p.values) {
                    values[name] = json::parse(value);
                }
                return {{"definition", p.definition}, {"values", values}};
            } else if constexpr (std::is_same_v<T, PromptParams>) {
                return {{"prompt", p.prompt}, {"negativePrompt", p.negativePrompt}, {"vocabulary", p.vocabulary}};
            } else if constexpr (std::is_same_v<T, LatentNoiseParams>) {
                return {{"width", p.width}, {"height", p.height}, {"seed", p.seed}, {"path", p.path}};
            } else if constexpr (std::is_same_v<T, DdimSamplerParams>) {
                return {{"path", p.path}, {"steps", p.steps}, {"guidanceScale", p.guidanceScale}};
            } else {
                return json::object();
            }
        },
        params);
}

// Missing fields keep their defaults, so older files stay loadable.
template <typename T> void read(const json& j, const char* key, T& value) {
    if (j.contains(key)) {
        j.at(key).get_to(value);
    }
}

NodeParams paramsFromJson(NodeKind kind, const json& j) {
    NodeParams params = defaultParams(kind);
    std::visit(
        [&](auto& p) {
            using T = std::decay_t<decltype(p)>;
            if constexpr (std::is_same_v<T, SceneParams>) {
                read(j, "path", p.path);
                read(j, "flipY", p.flipY);
            } else if constexpr (std::is_same_v<T, TransformGaussiansParams> || std::is_same_v<T, MakeTransformParams>) {
                read(j, "translation", p.translation);
                read(j, "rotation", p.rotation);
                read(j, "scale", p.scale);
            } else if constexpr (std::is_same_v<T, NumberParams>) {
                read(j, "value", p.value);
            } else if constexpr (std::is_same_v<T, TimeParams>) {
                read(j, "speed", p.speed);
            } else if constexpr (std::is_same_v<T, SineParams>) {
                read(j, "amplitude", p.amplitude);
                read(j, "frequency", p.frequency);
                read(j, "phase", p.phase);
                read(j, "offset", p.offset);
            } else if constexpr (std::is_same_v<T, CameraParams>) {
                read(j, "azimuth", p.azimuth);
                read(j, "elevation", p.elevation);
                read(j, "distance", p.distance);
                read(j, "target", p.target);
                if (j.contains("up")) {
                    const auto up = j.at("up").get<std::string>();
                    if (up != "y" && up != "z") {
                        throw std::runtime_error("unknown camera up axis '" + up + "'");
                    }
                    p.up = up == "y" ? UpAxis::Y : UpAxis::Z;
                }
            } else if constexpr (std::is_same_v<T, SplattingParams>) {
                if (j.contains("backend")) {
                    const auto backend = j.at("backend").get<std::string>();
                    if (backend != "compute" && backend != "raster") {
                        throw std::runtime_error("unknown splatting backend '" + backend + "'");
                    }
                    p.backend = backend == "raster" ? SplattingBackend::Raster : SplattingBackend::Compute;
                }
                read(j, "spreadMultiplier", p.spreadMultiplier);
                read(j, "maxMod", p.maxMod);
                read(j, "numSortWGsCap", p.numSortWGsCap);
                read(j, "splatTileX", p.splatTileX);
                read(j, "splatTileY", p.splatTileY);
                read(j, "shDegree", p.shDegree);
                read(j, "alphaCullThreshold", p.alphaCullThreshold);
                read(j, "useMeshShader", p.useMeshShader);
            } else if constexpr (std::is_same_v<T, OffscreenTargetParams>) {
                read(j, "width", p.width);
                read(j, "height", p.height);
                read(j, "clear", p.clear);
            } else if constexpr (std::is_same_v<T, SwapchainTargetParams>) {
                read(j, "clear", p.clear);
            } else if constexpr (std::is_same_v<T, ClearImageParams>) {
                read(j, "color", p.color);
            } else if constexpr (std::is_same_v<T, CompositeParams>) {
                std::string mode;
                read(j, "mode", mode);
                if (mode == compositeModeName(CompositeMode::Over)) {
                    p.mode = CompositeMode::Over;
                }
                std::string fit;
                read(j, "fit", fit);
                for (auto f : {CompositeFit::Stretch, CompositeFit::Fit, CompositeFit::Fill}) {
                    if (fit == compositeFitName(f)) {
                        p.fit = f;
                    }
                }
            } else if constexpr (std::is_same_v<T, DrawBasicsParams>) {
                std::string shape;
                read(j, "shape", shape);
                for (auto s : {DrawBasicsShape::Triangle, DrawBasicsShape::Cube, DrawBasicsShape::Axes}) {
                    if (shape == drawBasicsShapeName(s)) {
                        p.shape = s;
                    }
                }
            } else if constexpr (std::is_same_v<T, ImageFileParams>) {
                read(j, "path", p.path);
                read(j, "width", p.width);
                read(j, "height", p.height);
            } else if constexpr (std::is_same_v<T, ResampleParams>) {
                read(j, "width", p.width);
                read(j, "height", p.height);
                std::string filter;
                read(j, "filter", filter);
                if (filter == filterName(ResampleFilter::Nearest)) {
                    p.filter = ResampleFilter::Nearest;
                }
            } else if constexpr (std::is_same_v<T, OnnxModelParams>) {
                read(j, "path", p.path);
                read(j, "inputs", p.inputs);
                read(j, "outputs", p.outputs);
            } else if constexpr (std::is_same_v<T, ImageFileWriterParams>) {
                read(j, "path", p.path);
            } else if constexpr (std::is_same_v<T, BinaryLayerParams>) {
                read(j, "b", p.b);
            } else if constexpr (std::is_same_v<T, MetaParams>) {
                read(j, "definition", p.definition);
                if (j.contains("values")) {
                    for (const auto& [name, value] : j.at("values").items()) {
                        p.values[name] = value.dump();
                    }
                }
            } else if constexpr (std::is_same_v<T, PromptParams>) {
                read(j, "prompt", p.prompt);
                read(j, "negativePrompt", p.negativePrompt);
                read(j, "vocabulary", p.vocabulary);
            } else if constexpr (std::is_same_v<T, LatentNoiseParams>) {
                read(j, "width", p.width);
                read(j, "height", p.height);
                read(j, "seed", p.seed);
                read(j, "path", p.path);
            } else if constexpr (std::is_same_v<T, DdimSamplerParams>) {
                read(j, "path", p.path);
                read(j, "steps", p.steps);
                read(j, "guidanceScale", p.guidanceScale);
            }
        },
        params);
    return params;
}

// Whether `from` -> `to` would link CPU Gaussians into an input that wants
// them on the GPU.
bool linksCpuToGpuGaussians(const Graph& graph, const PinRef& from, const PinRef& to) {
    const Node* source = graph.findNode(from.node);
    const Node* target = graph.findNode(to.node);
    if (!source || !target) {
        return false;
    }
    const auto& outputs = kindInfo(source->kind).outputs;
    const auto& inputs = kindInfo(target->kind).inputs;
    return from.slot >= 0 && from.slot < static_cast<int>(outputs.size()) && to.slot >= 0 &&
           to.slot < static_cast<int>(inputs.size()) && outputs[from.slot].type == PinType::GaussiansCpu &&
           inputs[to.slot].type == PinType::GaussiansGpu;
}

} // namespace

std::string paramsToString(const NodeParams& params) {
    return paramsToJson(params).dump();
}

std::vector<std::string> paramKeys(const NodeParams& params) {
    std::vector<std::string> keys;
    for (const auto& [key, value] : paramsToJson(params).items()) {
        keys.push_back(key);
    }
    return keys;
}

std::optional<std::string> paramValue(const NodeParams& params, std::string_view key) {
    const json j = paramsToJson(params);
    auto it = j.find(key);
    return it == j.end() ? std::nullopt : std::optional(it->dump());
}

void setParamValue(NodeParams& params, NodeKind kind, std::string_view key, const std::string& value) {
    json j = paramsToJson(params);
    if (!j.contains(key)) {
        throw std::runtime_error("'" + std::string(kindInfo(kind).title) + "' has no parameter '" + std::string(key) +
                                 "'");
    }
    try {
        j[std::string(key)] = json::parse(value);
        params = paramsFromJson(kind, j);
    } catch (const json::exception& e) {
        throw std::runtime_error("invalid value for '" + std::string(key) + "': " + e.what());
    }
}

namespace {

json nodesToJson(const Graph& graph) {
    json nodes = json::array();
    for (const auto& node : graph.nodes()) {
        nodes.push_back({{"id", node.id},
                         {"kind", kindInfo(node.kind).name},
                         {"title", node.title},
                         {"position", {node.position.x, node.position.y}},
                         {"params", paramsToJson(node.params)}});
    }
    return nodes;
}

json linksToJson(const Graph& graph) {
    json links = json::array();
    for (const auto& link : graph.links()) {
        links.push_back({{"from", {link.fromNode, link.fromSlot}}, {"to", {link.toNode, link.toSlot}}});
    }
    return links;
}

json pinRefToJson(const PinRef& pin) {
    return {pin.node, pin.slot};
}

PinRef pinRefFromJson(const json& j, PinDirection direction) {
    return {j.at(0).get<int>(), direction, j.at(1).get<int>()};
}

// Local definitions that `definition` uses, directly or through others.
void collectDependencies(const Graph& graph, const MetaDefinition& definition, std::vector<std::string>& order,
                         std::set<std::string>& visiting) {
    if (std::find(order.begin(), order.end(), definition.name) != order.end() ||
        !visiting.insert(definition.name).second) {
        return;  // done, or a cycle (reported when flattening)
    }
    for (const auto& node : definition.graph.nodes()) {
        if (node.kind != NodeKind::Meta) {
            continue;
        }
        auto it = graph.definitions().find(node.as<MetaParams>().definition);
        if (it != graph.definitions().end()) {
            collectDependencies(graph, *it->second, order, visiting);
        }
    }
    order.push_back(definition.name);
}

json definitionsToJson(const Graph& graph) {
    // Dependencies first, so that loading can resolve nested meta nodes.
    std::vector<std::string> order;
    std::set<std::string> visiting;
    for (const auto& [name, definition] : graph.definitions()) {
        collectDependencies(graph, *definition, order, visiting);
    }
    json definitions = json::array();
    for (const auto& name : order) {
        const MetaDefinition& d = *graph.definitions().at(name);
        auto pinsToJson = [](const std::vector<MetaPin>& pins) {
            json result = json::array();
            for (const auto& pin : pins) {
                json targets = json::array();
                for (const auto& target : pin.targets) {
                    targets.push_back(pinRefToJson(target));
                }
                result.push_back({{"name", pin.name}, {"targets", targets}});
            }
            return result;
        };
        json params = json::array();
        for (const auto& param : d.params) {
            params.push_back({{"name", param.name}, {"node", param.node}, {"key", param.key}});
        }
        definitions.push_back({{"name", d.name},
                               {"title", d.title},
                               {"group", d.group},
                               {"description", d.description},
                               {"nodes", nodesToJson(d.graph)},
                               {"links", linksToJson(d.graph)},
                               {"inputs", pinsToJson(d.inputs)},
                               {"outputs", pinsToJson(d.outputs)},
                               {"params", params}});
    }
    return definitions;
}

// Before they were meta nodes, the Stable Diffusion text encoder and VAE
// decoder were node kinds with a model path.
std::optional<std::string> formerMetaKind(const std::string& kindName) {
    if (kindName == "sd_text_encoder") return "sd15_text_encoder";
    if (kindName == "sd_vae_decoder") return "sd15_vae_decoder";
    return std::nullopt;
}

void loadNodes(Graph& graph, const json& nodes) {
    for (const auto& j : nodes) {
        const auto kindName = j.at("kind").get<std::string>();
        const auto former = formerMetaKind(kindName);
        const auto kind = former ? std::optional(NodeKind::Meta) : kindFromName(kindName);
        if (!kind) {
            throw std::runtime_error("unknown node kind '" + kindName + "'");
        }
        Node node;
        node.id = j.at("id").get<int>();
        node.kind = *kind;
        node.title = j.value("title", std::string(kindInfo(*kind).title));
        if (j.contains("position")) {
            const auto& pos = j.at("position");
            node.position = {pos.at(0).get<float>(), pos.at(1).get<float>()};
        }
        if (former) {
            MetaParams meta{*former, {}};
            const json params = j.value("params", json::object());
            if (params.contains("path")) {
                meta.values["Model"] = params.at("path").dump();
            }
            node.params = meta;
        } else {
            node.params = paramsFromJson(*kind, j.value("params", json::object()));
        }
        if (!graph.addNodeWithId(std::move(node))) {
            throw std::runtime_error("duplicate or invalid node id " + std::to_string(j.at("id").get<int>()));
        }
    }
}

void loadLinks(Graph& graph, const json& links) {
    for (const auto& j : links) {
        const PinRef from{j.at("from").at(0).get<int>(), PinDirection::Output, j.at("from").at(1).get<int>()};
        const PinRef to{j.at("to").at(0).get<int>(), PinDirection::Input, j.at("to").at(1).get<int>()};
        if (auto error = graph.connect(from, to)) {
            if (!linksCpuToGpuGaussians(graph, from, to)) {
                throw std::runtime_error("invalid link: " + *error);
            }
            // Before Upload Gaussians existed, scenes fed Gaussian
            // Splatting directly: put the upload in between.
            const Node& source = *graph.findNode(from.node);
            const Node& target = *graph.findNode(to.node);
            const int upload = graph.addNode(
                NodeKind::UploadGaussians,
                {(source.position.x + target.position.x) * 0.5f, (source.position.y + target.position.y) * 0.5f});
            graph.connect(from, {upload, PinDirection::Input, 0});
            graph.connect({upload, PinDirection::Output, 0}, to);
        }
    }
}

void loadDefinitions(Graph& graph, const json& definitions) {
    for (const auto& j : definitions) {
        auto definition = std::make_shared<MetaDefinition>();
        definition->name = j.at("name").get<std::string>();
        definition->title = j.value("title", definition->name);
        definition->group = j.value("group", std::string());
        definition->description = j.value("description", std::string());
        try {
            // The definitions loaded so far resolve nested meta nodes while
            // the links are checked; the inner graph does not keep them.
            Graph inner;
            for (const auto& [name, loaded] : graph.definitions()) {
                inner.setDefinition(loaded);
            }
            loadNodes(inner, j.at("nodes"));
            loadLinks(inner, j.at("links"));
            for (const auto& [name, loaded] : graph.definitions()) {
                inner.removeDefinition(name);
            }
            definition->graph = std::move(inner);
        } catch (const std::runtime_error& e) {
            throw std::runtime_error("definition '" + definition->name + "': " + e.what());
        }
        auto pinsFromJson = [](const json& pins, PinDirection direction) {
            std::vector<MetaPin> result;
            for (const auto& pin : pins) {
                MetaPin meta{pin.at("name").get<std::string>(), {}};
                for (const auto& target : pin.at("targets")) {
                    meta.targets.push_back(pinRefFromJson(target, direction));
                }
                result.push_back(std::move(meta));
            }
            return result;
        };
        definition->inputs = pinsFromJson(j.value("inputs", json::array()), PinDirection::Input);
        definition->outputs = pinsFromJson(j.value("outputs", json::array()), PinDirection::Output);
        for (const auto& param : j.value("params", json::array())) {
            definition->params.push_back(
                {param.at("name").get<std::string>(), param.at("node").get<int>(), param.at("key").get<std::string>()});
        }
        graph.setDefinition(std::move(definition));
    }
}

} // namespace

std::string toJson(const Graph& graph) {
    json doc = {{"format", "klartraum-studio-graph"}, {"version", kFormatVersion}, {"nodes", nodesToJson(graph)},
                {"links", linksToJson(graph)}};
    if (!graph.definitions().empty()) {
        doc["definitions"] = definitionsToJson(graph);
    }
    return doc.dump(2);
}

Graph fromJson(const std::string& text) {
    Graph graph;
    try {
        const json doc = json::parse(text);
        if (doc.value("format", "") != "klartraum-studio-graph") {
            throw std::runtime_error("not a klartraum-studio graph");
        }
        if (doc.value("version", 0) > kFormatVersion) {
            throw std::runtime_error("graph was saved by a newer version of klartraum-studio");
        }
        if (doc.contains("definitions")) {
            loadDefinitions(graph, doc.at("definitions"));
        }
        loadNodes(graph, doc.at("nodes"));
        loadLinks(graph, doc.at("links"));
    } catch (const json::exception& e) {
        throw std::runtime_error(std::string("malformed graph file: ") + e.what());
    }
    return graph;
}

void saveGraph(const Graph& graph, const std::filesystem::path& path) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("cannot write " + path.string());
    }
    out << toJson(graph) << '\n';
}

Graph loadGraph(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot read " + path.string());
    }
    std::stringstream buffer;
    buffer << in.rdbuf();
    return fromJson(buffer.str());
}

} // namespace kstudio
