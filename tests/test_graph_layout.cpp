/**
 * TESTS:
 * - emptyGraph: laying out no nodes yields no positions
 * - layersFollowLongestPath: each node's column is its longest path from a source
 * - edgesPointRight: every edge goes from a lower to a higher column
 * - noOverlaps: nodes in the same column get distinct rows
 * - measuredHeightsStackWithoutOverlap: with node heights, a column is stacked top-aligned without overlaps
 * - orderingReducesCrossings: barycenter ordering untangles a crossed bipartite graph
 * - rejectsCycles: a cyclic graph throws
 **/

#include <gtest/gtest.h>

#include <algorithm>
#include <set>
#include <stdexcept>

#include "studio/graph_layout.hpp"

using namespace kstudio;

namespace {

int crossings(const std::vector<Vec2>& pos, const std::vector<std::pair<int, int>>& edges) {
    int count = 0;
    for (size_t i = 0; i < edges.size(); ++i) {
        for (size_t j = i + 1; j < edges.size(); ++j) {
            const auto [a0, a1] = edges[i];
            const auto [b0, b1] = edges[j];
            if (pos[a0].x != pos[b0].x || pos[a1].x != pos[b1].x) {
                continue;
            }
            if ((pos[a0].y - pos[b0].y) * (pos[a1].y - pos[b1].y) < 0) {
                ++count;
            }
        }
    }
    return count;
}

} // namespace

TEST(GraphLayout, emptyGraph) {
    EXPECT_TRUE(layeredLayout(0, {}).empty());
}

TEST(GraphLayout, layersFollowLongestPath) {
    // 0 -> 1 -> 2, and 0 -> 2 directly: 2 sits after 1.
    const auto layers = longestPathLayers(4, {{0, 1}, {1, 2}, {0, 2}});
    EXPECT_EQ(layers, (std::vector<int>{0, 1, 2, 0}));
}

TEST(GraphLayout, edgesPointRight) {
    const std::vector<std::pair<int, int>> edges{{0, 2}, {1, 2}, {2, 3}, {1, 3}, {3, 4}, {0, 4}};
    const auto pos = layeredLayout(5, edges);
    for (const auto& [from, to] : edges) {
        EXPECT_LT(pos[from].x, pos[to].x);
    }
}

TEST(GraphLayout, noOverlaps) {
    const std::vector<std::pair<int, int>> edges{{0, 3}, {1, 3}, {2, 3}, {0, 4}};
    const auto pos = layeredLayout(5, edges);
    std::set<std::pair<float, float>> seen;
    for (const auto& p : pos) {
        EXPECT_TRUE(seen.emplace(p.x, p.y).second);
    }
}

TEST(GraphLayout, measuredHeightsStackWithoutOverlap) {
    const std::vector<std::pair<int, int>> edges{{0, 3}, {1, 3}, {2, 3}};
    LayoutOptions options;
    options.heights = {200.0f, 40.0f, 80.0f, 60.0f};
    options.rowGap = 10.0f;
    const auto pos = layeredLayout(4, edges, options);

    std::vector<int> column{0, 1, 2};
    std::sort(column.begin(), column.end(), [&](int a, int b) { return pos[a].y < pos[b].y; });
    EXPECT_EQ(pos[column[0]].y, 0.0f);
    for (size_t i = 1; i < column.size(); ++i) {
        const int above = column[i - 1];
        EXPECT_FLOAT_EQ(pos[column[i]].y, pos[above].y + options.heights[above] + options.rowGap);
    }
    EXPECT_EQ(pos[3].y, 0.0f);
}

TEST(GraphLayout, orderingReducesCrossings) {
    // Sources 0,1,2 feed sinks in reverse order: 0->5, 1->4, 2->3.
    const std::vector<std::pair<int, int>> edges{{0, 5}, {1, 4}, {2, 3}};
    LayoutOptions none;
    none.orderingSweeps = 0;
    EXPECT_GT(crossings(layeredLayout(6, edges, none), edges), 0);
    EXPECT_EQ(crossings(layeredLayout(6, edges), edges), 0);
}

TEST(GraphLayout, rejectsCycles) {
    EXPECT_THROW(layeredLayout(2, {{0, 1}, {1, 0}}), std::invalid_argument);
}
