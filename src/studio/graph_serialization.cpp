#include "studio/graph_serialization.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace kstudio {

namespace {

using nlohmann::json;

constexpr int kFormatVersion = 1;

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
                return {{"width", p.width}, {"height", p.height}};
            } else if constexpr (std::is_same_v<T, ImageFileParams>) {
                return {{"path", p.path}, {"width", p.width}, {"height", p.height}};
            } else if constexpr (std::is_same_v<T, ResampleParams>) {
                return {{"width", p.width}, {"height", p.height}, {"filter", filterName(p.filter)}};
            } else if constexpr (std::is_same_v<T, OnnxModelParams> || std::is_same_v<T, ImageFileWriterParams>) {
                return {{"path", p.path}};
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
            } else if constexpr (std::is_same_v<T, OnnxModelParams> || std::is_same_v<T, ImageFileWriterParams>) {
                read(j, "path", p.path);
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

std::string toJson(const Graph& graph) {
    json nodes = json::array();
    for (const auto& node : graph.nodes()) {
        nodes.push_back({{"id", node.id},
                         {"kind", kindInfo(node.kind).name},
                         {"title", node.title},
                         {"position", {node.position.x, node.position.y}},
                         {"params", paramsToJson(node.params)}});
    }
    json links = json::array();
    for (const auto& link : graph.links()) {
        links.push_back({{"from", {link.fromNode, link.fromSlot}}, {"to", {link.toNode, link.toSlot}}});
    }
    const json doc = {{"format", "klartraum-studio-graph"}, {"version", kFormatVersion}, {"nodes", nodes},
                      {"links", links}};
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
        for (const auto& j : doc.at("nodes")) {
            const auto kindName = j.at("kind").get<std::string>();
            const auto kind = kindFromName(kindName);
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
            node.params = paramsFromJson(*kind, j.value("params", json::object()));
            if (!graph.addNodeWithId(std::move(node))) {
                throw std::runtime_error("duplicate or invalid node id " + std::to_string(j.at("id").get<int>()));
            }
        }
        for (const auto& j : doc.at("links")) {
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
