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
                return {{"path", p.path}};
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
            } else if constexpr (std::is_same_v<T, OnnxModelParams> || std::is_same_v<T, ImageFileWriterParams>) {
                read(j, "path", p.path);
            }
        },
        params);
    return params;
}

} // namespace

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
                throw std::runtime_error("invalid link: " + *error);
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
