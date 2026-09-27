/**
 * TESTS:
 * - numberAndTime: a Number gives its value, Time the seconds times its speed
 * - sineOfTime: a Sine fed by Time swings between offset - amplitude and offset + amplitude
 *   with the given frequency and phase
 * - unconnectedSineThrows: a Sine without input reports its title
 **/

#include <gtest/gtest.h>

#include "studio/cpu_numbers.hpp"

using namespace kstudio;

namespace {

PinRef out(int node) { return {node, PinDirection::Output, 0}; }
PinRef in(int node) { return {node, PinDirection::Input, 0}; }

} // namespace

TEST(CpuNumbers, numberAndTime) {
    Graph graph;
    const int number = graph.addNode(NodeKind::Number);
    graph.findNode(number)->as<NumberParams>().value = 2.5f;
    const int time = graph.addNode(NodeKind::Time);
    graph.findNode(time)->as<TimeParams>().speed = 2.0f;
    EXPECT_FLOAT_EQ(evaluateNumber(graph, number, 10.0), 2.5f);
    EXPECT_FLOAT_EQ(evaluateNumber(graph, time, 1.5), 3.0f);
}

TEST(CpuNumbers, sineOfTime) {
    Graph graph;
    const int time = graph.addNode(NodeKind::Time);
    const int sine = graph.addNode(NodeKind::Sine);
    ASSERT_FALSE(graph.connect(out(time), in(sine)).has_value());
    auto& p = graph.findNode(sine)->as<SineParams>();
    p.amplitude = 30.0f;
    p.frequency = 0.25f;  // one swing every 4 seconds
    p.offset = 20.0f;
    EXPECT_NEAR(evaluateNumber(graph, sine, 0.0), 20.0f, 1e-4f);
    EXPECT_NEAR(evaluateNumber(graph, sine, 1.0), 50.0f, 1e-4f);
    EXPECT_NEAR(evaluateNumber(graph, sine, 3.0), -10.0f, 1e-4f);
    p.phase = 90.0f;
    EXPECT_NEAR(evaluateNumber(graph, sine, 0.0), 50.0f, 1e-4f);
}

TEST(CpuNumbers, unconnectedSineThrows) {
    Graph graph;
    const int sine = graph.addNode(NodeKind::Sine);
    graph.findNode(sine)->title = "Swing";
    try {
        evaluateNumber(graph, sine, 0.0);
        FAIL() << "expected an error";
    } catch (const std::runtime_error& e) {
        EXPECT_TRUE(std::string(e.what()).starts_with("Swing: ")) << e.what();
    }
}
