#include "studio/graph_layout.hpp"

#include <algorithm>
#include <numeric>
#include <queue>
#include <stdexcept>

namespace kstudio {

std::vector<int> longestPathLayers(int nodeCount, const std::vector<std::pair<int, int>>& edges) {
    std::vector<std::vector<int>> successors(nodeCount);
    std::vector<int> indegree(nodeCount, 0);
    for (const auto& [from, to] : edges) {
        successors[from].push_back(to);
        ++indegree[to];
    }

    std::vector<int> layer(nodeCount, 0);
    std::queue<int> ready;
    for (int i = 0; i < nodeCount; ++i) {
        if (indegree[i] == 0) {
            ready.push(i);
        }
    }
    int visited = 0;
    while (!ready.empty()) {
        const int node = ready.front();
        ready.pop();
        ++visited;
        for (int next : successors[node]) {
            layer[next] = std::max(layer[next], layer[node] + 1);
            if (--indegree[next] == 0) {
                ready.push(next);
            }
        }
    }
    if (visited != nodeCount) {
        throw std::invalid_argument("layeredLayout needs an acyclic graph");
    }
    return layer;
}

std::vector<Vec2> layeredLayout(int nodeCount, const std::vector<std::pair<int, int>>& edges,
                                const LayoutOptions& options) {
    const std::vector<int> layer = longestPathLayers(nodeCount, edges);
    const int layerCount = nodeCount == 0 ? 0 : *std::max_element(layer.begin(), layer.end()) + 1;

    std::vector<std::vector<int>> columns(layerCount);
    for (int i = 0; i < nodeCount; ++i) {
        columns[layer[i]].push_back(i);
    }

    std::vector<std::vector<int>> predecessors(nodeCount), successors(nodeCount);
    for (const auto& [from, to] : edges) {
        successors[from].push_back(to);
        predecessors[to].push_back(from);
    }

    std::vector<float> rank(nodeCount, 0.0f);
    auto updateRanks = [&](int column) {
        for (int i = 0; i < static_cast<int>(columns[column].size()); ++i) {
            rank[columns[column][i]] = static_cast<float>(i);
        }
    };
    for (int c = 0; c < layerCount; ++c) {
        updateRanks(c);
    }

    // Orders a column by the mean rank of each node's neighbours; nodes
    // without neighbours keep their current rank.
    auto orderColumn = [&](int column, const std::vector<std::vector<int>>& neighbours) {
        auto& nodes = columns[column];
        std::vector<float> key(nodes.size());
        for (size_t i = 0; i < nodes.size(); ++i) {
            const auto& n = neighbours[nodes[i]];
            key[i] = n.empty() ? rank[nodes[i]]
                               : std::accumulate(n.begin(), n.end(), 0.0f,
                                                 [&](float sum, int other) { return sum + rank[other]; }) /
                                     static_cast<float>(n.size());
        }
        std::vector<size_t> order(nodes.size());
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return key[a] < key[b]; });
        std::vector<int> sorted;
        for (size_t i : order) {
            sorted.push_back(nodes[i]);
        }
        nodes = std::move(sorted);
        updateRanks(column);
    };
    for (int sweep = 0; sweep < options.orderingSweeps; ++sweep) {
        for (int c = 1; c < layerCount; ++c) {
            orderColumn(c, predecessors);
        }
        for (int c = layerCount - 2; c >= 0; --c) {
            orderColumn(c, successors);
        }
    }

    std::vector<Vec2> positions(nodeCount);
    if (static_cast<int>(options.heights.size()) == nodeCount) {
        for (int c = 0; c < layerCount; ++c) {
            float y = 0.0f;
            for (int node : columns[c]) {
                positions[node] = {static_cast<float>(c) * options.columnSpacing, y};
                y += options.heights[node] + options.rowGap;
            }
        }
        return positions;
    }

    // Columns are centred vertically on the tallest one.
    size_t tallest = 0;
    for (const auto& column : columns) {
        tallest = std::max(tallest, column.size());
    }
    for (int c = 0; c < layerCount; ++c) {
        const float offset = 0.5f * static_cast<float>(tallest - columns[c].size()) * options.rowSpacing;
        for (size_t i = 0; i < columns[c].size(); ++i) {
            positions[columns[c][i]] = {static_cast<float>(c) * options.columnSpacing,
                                        offset + static_cast<float>(i) * options.rowSpacing};
        }
    }
    return positions;
}

} // namespace kstudio
