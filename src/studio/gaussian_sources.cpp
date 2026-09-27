#include "studio/gaussian_sources.hpp"

#include <format>
#include <stdexcept>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "klartraum/gaussian_transform.hpp"

namespace kstudio {

namespace {

glm::quat toQuat(const std::array<float, 4>& q) { return glm::quat(q[3], q[0], q[1], q[2]); }
std::array<float, 4> fromQuat(const glm::quat& q) { return {q.x, q.y, q.z, q.w}; }

} // namespace

Similarity toSimilarity(const TransformGaussiansParams& params) {
    const glm::quat x = glm::angleAxis(glm::radians(params.rotation[0]), glm::vec3(1.0f, 0.0f, 0.0f));
    const glm::quat y = glm::angleAxis(glm::radians(params.rotation[1]), glm::vec3(0.0f, 1.0f, 0.0f));
    const glm::quat z = glm::angleAxis(glm::radians(params.rotation[2]), glm::vec3(0.0f, 0.0f, 1.0f));
    Similarity result;
    result.translation = params.translation;
    result.rotation = fromQuat(glm::normalize(z * y * x));
    result.scale = params.scale;
    return result;
}

Similarity compose(const Similarity& outer, const Similarity& inner) {
    // outer(inner(p)) = so Ro (si Ri p + ti) + to
    const glm::quat ro = toQuat(outer.rotation);
    const glm::vec3 ti(inner.translation[0], inner.translation[1], inner.translation[2]);
    const glm::vec3 to(outer.translation[0], outer.translation[1], outer.translation[2]);
    const glm::vec3 t = outer.scale * (ro * ti) + to;
    Similarity result;
    result.translation = {t.x, t.y, t.z};
    result.rotation = fromQuat(glm::normalize(ro * toQuat(inner.rotation)));
    result.scale = outer.scale * inner.scale;
    return result;
}

std::vector<GaussianPart> gaussianParts(const Graph& graph, int nodeId) {
    const Node* node = graph.findNode(nodeId);
    if (!node) {
        throw std::logic_error("gaussianParts: unknown node");
    }
    auto input = [&](int slot) {
        const Link* link = graph.inputLink(nodeId, slot);
        if (!link) {
            throw std::runtime_error(node->title + ": input '" +
                                     std::string(kindInfo(node->kind).inputs[slot].name) + "' is not connected");
        }
        return gaussianParts(graph, link->fromNode);
    };
    switch (node->kind) {
    case NodeKind::Scene: {
        const auto& p = node->as<SceneParams>();
        return {GaussianPart{p.path, p.flipY, {}}};
    }
    case NodeKind::TransformGaussians: {
        const Similarity transform = toSimilarity(node->as<TransformGaussiansParams>());
        std::vector<GaussianPart> parts = input(0);
        for (auto& part : parts) {
            part.transform = compose(transform, part.transform);
        }
        return parts;
    }
    case NodeKind::MergeGaussians: {
        std::vector<GaussianPart> parts = input(0);
        std::vector<GaussianPart> more = input(1);
        parts.insert(parts.end(), more.begin(), more.end());
        return parts;
    }
    default:
        throw std::logic_error("gaussianParts: the node does not output Gaussians");
    }
}

std::string partsKey(const std::vector<GaussianPart>& parts) {
    std::string key;
    for (const auto& part : parts) {
        const auto& t = part.transform;
        key += std::format("{}|{}|{},{},{}|{},{},{},{}|{};", part.path, part.flipY, t.translation[0], t.translation[1],
                           t.translation[2], t.rotation[0], t.rotation[1], t.rotation[2], t.rotation[3], t.scale);
    }
    return key;
}

std::vector<klartraum::Gaussian3D> assembleGaussians(const std::vector<GaussianPart>& parts,
                                                     const GaussianLoader& load) {
    std::vector<klartraum::Gaussian3D> result;
    for (const auto& part : parts) {
        const auto scene = load(part.path, part.flipY);
        std::vector<klartraum::Gaussian3D> placed(scene->begin(), scene->end());
        if (!(part.transform == Similarity{})) {
            const auto& t = part.transform;
            klartraum::transformGaussians(placed, toQuat(t.rotation), t.scale,
                                          glm::vec3(t.translation[0], t.translation[1], t.translation[2]));
        }
        result.insert(result.end(), placed.begin(), placed.end());
    }
    return result;
}

} // namespace kstudio
