#include "studio/cpu_numbers.hpp"

#include <cmath>
#include <numbers>
#include <stdexcept>

namespace kstudio {

float evaluateNumber(const Graph& graph, int nodeId, double time) {
    const Node* node = graph.findNode(nodeId);
    if (!node) {
        throw std::logic_error("evaluateNumber: unknown node");
    }
    switch (node->kind) {
    case NodeKind::Number:
        return node->as<NumberParams>().value;
    case NodeKind::Time:
        return static_cast<float>(time * node->as<TimeParams>().speed);
    case NodeKind::Sine: {
        const Link* link = graph.inputLink(nodeId, 0);
        if (!link) {
            throw std::runtime_error(node->title + ": input 'X' is not connected");
        }
        const auto& p = node->as<SineParams>();
        const double x = evaluateNumber(graph, link->fromNode, time);
        const double angle = 2.0 * std::numbers::pi * p.frequency * x + p.phase * std::numbers::pi / 180.0;
        return static_cast<float>(p.amplitude * std::sin(angle) + p.offset);
    }
    default:
        throw std::logic_error("evaluateNumber: the node does not output a CPU number");
    }
}

} // namespace kstudio
