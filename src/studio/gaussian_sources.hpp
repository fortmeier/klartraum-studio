#pragma once

#include <array>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "klartraum/vulkan_gaussian_splatting_types.hpp"

#include "studio/graph_model.hpp"

namespace kstudio {

// p -> scale * R p + translation, with R the rotation quaternion (x, y, z, w).
struct Similarity {
    std::array<float, 3> translation = {0.0f, 0.0f, 0.0f};
    std::array<float, 4> rotation = {0.0f, 0.0f, 0.0f, 1.0f};
    float scale = 1.0f;
    bool operator==(const Similarity&) const = default;
};

// Rotates about X, then Y, then Z (degrees).
Similarity toSimilarity(const TransformGaussiansParams& params);
// Applies `inner` first, then `outer`.
Similarity compose(const Similarity& outer, const Similarity& inner);

// One scene file and where it is placed.
struct GaussianPart {
    std::string path;
    bool flipY = false;
    Similarity transform;
    bool operator==(const GaussianPart&) const = default;
};

// The scenes a Gaussians output carries, in merge order: Scene nodes are
// parts, Transform nodes place their input's parts, Merge nodes concatenate
// their inputs'. Expects the node's inputs to be connected.
std::vector<GaussianPart> gaussianParts(const Graph& graph, int node);

// Identifies the Gaussians the parts assemble into, e.g. for caching.
std::string partsKey(const std::vector<GaussianPart>& parts);

// A scene file's Gaussians as klartraum::loadGaussiansSpz returns them.
using GaussianLoader =
    std::function<std::shared_ptr<const std::vector<klartraum::Gaussian3D>>(const std::string& path, bool flipY)>;

// Loads, places and concatenates the parts.
std::vector<klartraum::Gaussian3D> assembleGaussians(const std::vector<GaussianPart>& parts,
                                                     const GaussianLoader& load);

} // namespace kstudio
