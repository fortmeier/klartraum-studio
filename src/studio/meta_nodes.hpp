#pragma once

#include <map>
#include <string>
#include <vector>

#include "studio/graph_model.hpp"

namespace kstudio {

// Meta nodes: subgraphs shown as one node (see MetaDefinition). Planning and
// execution see them flattened into their inner nodes, so the compiler, the
// runner and the live graph only ever deal with plain nodes.

// Built-in definition names.
inline constexpr const char* kTextEncoderDefinition = "sd15_text_encoder";
inline constexpr const char* kVaeDecoderDefinition = "sd15_vae_decoder";

// A graph with its meta nodes replaced by their inner nodes, recursively.
struct FlatGraph {
    Graph graph;
    // Every flat node's top-level node: itself for a node of the original
    // graph, else the outermost meta node it came from.
    std::map<int, int> topNode;
    // A top-level meta node's output pins, as the flat pins they stand for;
    // other top-level output pins keep their ids (see flatOutput).
    std::map<OutputPin, OutputPin> outputs;
    // Meta nodes that could not be expanded (unknown definitions, cycles
    // between definitions, links their inner pins reject), on top-level nodes.
    std::vector<Diagnostic> diagnostics;

    int top(int flatNode) const;
    OutputPin flatOutput(OutputPin topPin) const;
    // Whether any of `flatNodes` belongs to top-level node `topId`.
    bool covers(const std::vector<int>& flatNodes, int topId) const;
};

// Top-level nodes keep their ids; inner nodes get ids after the graph's
// largest one and are titled "<meta node title> / <inner title>". Exposed
// parameter values are applied to the inner nodes.
FlatGraph flatten(const Graph& graph);

// Replaces `nodes` of `graph` by one meta node of a new definition `name`
// stored with the graph. Links from outside into the group become exposed
// inputs (one per outside output pin, feeding every inner pin it fed), links
// out of it exposed outputs (one per inner output pin). Returns the meta
// node's id. Throws std::invalid_argument for an empty group, unknown nodes
// or a name that is taken.
int groupNodes(Graph& graph, const std::vector<int>& nodes, const std::string& name, const std::string& title);

// Replaces meta node `node` by a copy of its definition's inner nodes, with
// its exposed parameter values applied, reconnected where the meta node was.
// Returns the new nodes' ids. Throws std::invalid_argument if `node` is not a
// meta node with a known definition.
std::vector<int> ungroupNode(Graph& graph, int node);

// After definition `name` in `graph` was replaced by one whose exposed pins
// differ from `before`'s, moves the links of `graph`'s instances of it to the
// slots of the pins with the same names; links to pins that are gone are
// removed. Returns how many. (Instances inside other definitions keep their
// links by slot.)
int updateMetaLinks(Graph& graph, const std::string& name, const MetaDefinition& before);

// Removes exposed pins and parameters whose inner nodes or slots no longer
// exist in the definition's graph.
void pruneInterface(MetaDefinition& definition);

// A name for a new definition in `graph`, based on `base`, that no
// definition (local or built-in) has.
std::string uniqueDefinitionName(const Graph& graph, const std::string& base);

} // namespace kstudio
