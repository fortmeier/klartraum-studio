#pragma once

#include <utility>
#include <vector>

#include "studio/graph_model.hpp"

namespace kstudio {

struct LayoutOptions {
    float columnSpacing = 260.0f;
    float rowSpacing = 90.0f;
    int orderingSweeps = 4;
    // Per-node heights. When given, each column is stacked top-aligned with
    // `rowGap` between nodes instead of using a fixed `rowSpacing`.
    std::vector<float> heights;
    float rowGap = 24.0f;
};

// Layered ("Sugiyama-style") layout of a DAG with nodes 0..nodeCount-1 and
// edges (from, to): nodes are placed in columns by their longest path from a
// source, then each column is ordered by the barycenter of its neighbours to
// reduce crossings. Returns one position per node.
std::vector<Vec2> layeredLayout(int nodeCount, const std::vector<std::pair<int, int>>& edges,
                                const LayoutOptions& options = {});

// The column (layer) of each node, as used by layeredLayout.
std::vector<int> longestPathLayers(int nodeCount, const std::vector<std::pair<int, int>>& edges);

} // namespace kstudio
