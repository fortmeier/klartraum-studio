#include "studio/meta_nodes.hpp"

#include <algorithm>
#include <optional>
#include <set>
#include <stdexcept>

#include "studio/graph_serialization.hpp"
#include "studio/stable_diffusion.hpp"

namespace kstudio {

namespace {

PinRef inputPin(int node, int slot) { return {node, PinDirection::Input, slot}; }
PinRef outputPin(int node, int slot) { return {node, PinDirection::Output, slot}; }

std::shared_ptr<const MetaDefinition> makeTextEncoder() {
    auto d = std::make_shared<MetaDefinition>();
    d->name = kTextEncoderDefinition;
    d->title = "Text Encoder";
    d->group = "Stable Diffusion";
    d->description = "The CLIP text encoder: turns the Prompt's token ids and attention mask into 2x77x768 "
                     "embeddings. An ONNX Model inside.";
    d->builtin = true;
    const int clip = d->graph.addNode(NodeKind::OnnxModel, {0.0f, 0.0f});
    d->graph.findNode(clip)->title = "CLIP";
    d->graph.setOnnxPins(clip, {"input_ids", "attention_mask"}, {"last_hidden_state"});
    d->inputs = {{"Ids", {inputPin(clip, 0)}}, {"Mask", {inputPin(clip, 1)}}};
    d->outputs = {{"Embeddings", {outputPin(clip, 0)}}};
    d->params = {{"Model", clip, "path"}};
    return d;
}

std::shared_ptr<const MetaDefinition> makeVaeDecoder() {
    auto d = std::make_shared<MetaDefinition>();
    d->name = kVaeDecoderDefinition;
    d->title = "VAE Decoder";
    d->group = "Stable Diffusion";
    d->description = "Decodes latents into a 1x3xHxW image tensor with values in [0, 1]: undoes the latent scaling, "
                     "runs the VAE decoder model and maps its [-1, 1] output to [0, 1].";
    d->builtin = true;
    Graph& g = d->graph;
    const int unscale = g.addNode(NodeKind::Multiply, {0.0f, 0.0f});
    const int decoder = g.addNode(NodeKind::OnnxModel, {300.0f, 0.0f});
    const int half = g.addNode(NodeKind::Multiply, {600.0f, 0.0f});
    const int offset = g.addNode(NodeKind::Add, {900.0f, 0.0f});
    g.findNode(unscale)->title = "Unscale latents";
    g.findNode(unscale)->as<BinaryLayerParams>().b = 1.0f / kVaeScalingFactor;
    g.findNode(decoder)->title = "Decoder";
    g.findNode(half)->title = "Scale";
    g.findNode(half)->as<BinaryLayerParams>().b = 0.5f;
    g.findNode(offset)->title = "Offset";
    g.findNode(offset)->as<BinaryLayerParams>().b = 0.5f;
    g.connect(outputPin(unscale, 0), inputPin(decoder, 0));
    g.connect(outputPin(decoder, 0), inputPin(half, 0));
    g.connect(outputPin(half, 0), inputPin(offset, 0));
    d->inputs = {{"Latents", {inputPin(unscale, 0)}}};
    d->outputs = {{"Image", {outputPin(offset, 0)}}};
    d->params = {{"Model", decoder, "path"}, {"Latent scale", unscale, "b"}};
    return d;
}

// A definition's inner graph with an instance's exposed parameter values.
Graph instanceGraph(const MetaDefinition& definition, const MetaParams& instance) {
    Graph graph = definition.graph;
    for (const auto& [name, value] : instance.values) {
        auto param = std::find_if(definition.params.begin(), definition.params.end(),
                                  [&](const MetaParam& p) { return p.name == name; });
        if (param == definition.params.end()) {
            throw std::runtime_error("'" + definition.title + "' has no parameter '" + name + "'");
        }
        Node* node = graph.findNode(param->node);
        if (!node) {
            throw std::runtime_error("'" + definition.title + "' exposes a parameter of a missing node");
        }
        setParamValue(node->params, node->kind, param->key, value);
    }
    return graph;
}

// Where a meta node's pins lead in the flat graph.
struct Ports {
    std::vector<std::vector<PinRef>> inputs;
    std::vector<std::optional<PinRef>> outputs;
};

// How a graph's nodes appear in the flat graph.
struct Expansion {
    std::map<int, int> ids;      // plain node -> flat node
    std::map<int, Ports> metas;  // meta node -> its pins
};

class Flattener {
public:
    Flattener(const Graph& top, FlatGraph& flat) : top_(top), flat_(flat) {
        for (const auto& node : top.nodes()) {
            nextId_ = std::max(nextId_, node.id + 1);
        }
    }

    Expansion expand(const Graph& source, int owner, const std::string& prefix) {
        Expansion e;
        for (const auto& node : source.nodes()) {
            const int top = owner < 0 ? node.id : owner;
            if (node.kind != NodeKind::Meta) {
                Node copy = node;
                copy.id = owner < 0 ? node.id : nextId_++;
                copy.title = prefix + node.title;
                flat_.graph.addNodeWithId(copy);
                flat_.topNode[copy.id] = top;
                e.ids[node.id] = copy.id;
                continue;
            }
            const auto& params = node.as<MetaParams>();
            const MetaDefinition* definition = top_.findDefinition(params.definition);
            if (!definition) {
                error(top, "Unknown meta node definition '" + params.definition + "'.");
                continue;
            }
            if (std::find(stack_.begin(), stack_.end(), definition->name) != stack_.end()) {
                error(top, "The meta node definition '" + definition->name + "' contains itself.");
                continue;
            }
            Graph instance;
            try {
                instance = instanceGraph(*definition, params);
            } catch (const std::runtime_error& ex) {
                error(top, std::string(ex.what()) + ".");
                continue;
            }
            stack_.push_back(definition->name);
            const Expansion inner = expand(instance, top, prefix + node.title + " / ");
            stack_.pop_back();

            Ports ports;
            for (const auto& pin : definition->inputs) {
                std::vector<PinRef> targets;
                for (const auto& target : pin.targets) {
                    const auto resolved = resolveInput(inner, target);
                    targets.insert(targets.end(), resolved.begin(), resolved.end());
                }
                ports.inputs.push_back(std::move(targets));
            }
            for (const auto& pin : definition->outputs) {
                ports.outputs.push_back(pin.targets.empty() ? std::nullopt : resolveOutput(inner, pin.targets[0]));
            }
            e.metas[node.id] = std::move(ports);
        }

        for (const auto& link : source.links()) {
            std::optional<PinRef> from;
            if (auto meta = e.metas.find(link.fromNode); meta != e.metas.end()) {
                if (link.fromSlot < static_cast<int>(meta->second.outputs.size())) {
                    from = meta->second.outputs[link.fromSlot];
                }
            } else if (auto id = e.ids.find(link.fromNode); id != e.ids.end()) {
                from = outputPin(id->second, link.fromSlot);
            }
            std::vector<PinRef> to;
            if (auto meta = e.metas.find(link.toNode); meta != e.metas.end()) {
                if (link.toSlot < static_cast<int>(meta->second.inputs.size())) {
                    to = meta->second.inputs[link.toSlot];
                }
            } else if (auto id = e.ids.find(link.toNode); id != e.ids.end()) {
                to = {inputPin(id->second, link.toSlot)};
            }
            if (!from) {
                continue;  // a meta node that could not be expanded; reported
            }
            for (const auto& target : to) {
                if (auto problem = flat_.graph.connect(*from, target)) {
                    error(owner < 0 ? link.toNode : owner, "Inner link rejected: " + *problem);
                }
            }
        }
        return e;
    }

private:
    static std::vector<PinRef> resolveInput(const Expansion& inner, const PinRef& target) {
        if (auto meta = inner.metas.find(target.node); meta != inner.metas.end()) {
            return target.slot < static_cast<int>(meta->second.inputs.size()) ? meta->second.inputs[target.slot]
                                                                              : std::vector<PinRef>{};
        }
        if (auto id = inner.ids.find(target.node); id != inner.ids.end()) {
            return {inputPin(id->second, target.slot)};
        }
        return {};
    }

    static std::optional<PinRef> resolveOutput(const Expansion& inner, const PinRef& target) {
        if (auto meta = inner.metas.find(target.node); meta != inner.metas.end()) {
            return target.slot < static_cast<int>(meta->second.outputs.size()) ? meta->second.outputs[target.slot]
                                                                               : std::nullopt;
        }
        if (auto id = inner.ids.find(target.node); id != inner.ids.end()) {
            return outputPin(id->second, target.slot);
        }
        return std::nullopt;
    }

    void error(int node, std::string message) {
        flat_.diagnostics.push_back(Diagnostic{Severity::Error, node, std::move(message)});
    }

    const Graph& top_;
    FlatGraph& flat_;
    int nextId_ = 1;
    std::vector<std::string> stack_;
};

} // namespace

const MetaDefinitions& builtinDefinitions() {
    static const MetaDefinitions definitions = [] {
        MetaDefinitions result;
        for (auto definition : {makeTextEncoder(), makeVaeDecoder()}) {
            result[definition->name] = definition;
        }
        return result;
    }();
    return definitions;
}

int FlatGraph::top(int flatNode) const {
    auto it = topNode.find(flatNode);
    return it == topNode.end() ? flatNode : it->second;
}

OutputPin FlatGraph::flatOutput(OutputPin topPin) const {
    auto it = outputs.find(topPin);
    return it == outputs.end() ? topPin : it->second;
}

bool FlatGraph::covers(const std::vector<int>& flatNodes, int topId) const {
    return std::any_of(flatNodes.begin(), flatNodes.end(), [&](int id) { return top(id) == topId; });
}

FlatGraph flatten(const Graph& graph) {
    FlatGraph flat;
    Flattener flattener(graph, flat);
    const Expansion expansion = flattener.expand(graph, -1, "");
    for (const auto& [node, ports] : expansion.metas) {
        for (size_t slot = 0; slot < ports.outputs.size(); ++slot) {
            if (ports.outputs[slot]) {
                flat.outputs[{node, static_cast<int>(slot)}] = {ports.outputs[slot]->node, ports.outputs[slot]->slot};
            }
        }
    }
    return flat;
}

int updateMetaLinks(Graph& graph, const std::string& name, const MetaDefinition& before) {
    const MetaDefinition* after = graph.findDefinition(name);
    if (!after) {
        return 0;
    }
    auto names = [](const std::vector<MetaPin>& pins) {
        std::vector<std::string> result;
        for (const auto& pin : pins) {
            result.push_back(pin.name);
        }
        return result;
    };
    const auto beforeIn = names(before.inputs), afterIn = names(after->inputs);
    const auto beforeOut = names(before.outputs), afterOut = names(after->outputs);
    if (beforeIn == afterIn && beforeOut == afterOut) {
        return 0;
    }
    auto slotOf = [](const std::vector<std::string>& pins, const std::vector<std::string>& old, int slot) {
        if (slot < 0 || slot >= static_cast<int>(old.size())) {
            return -1;
        }
        auto it = std::find(pins.begin(), pins.end(), old[slot]);
        return it == pins.end() ? -1 : static_cast<int>(it - pins.begin());
    };
    std::set<int> instances;
    for (const auto& node : graph.nodes()) {
        if (node.kind == NodeKind::Meta && node.as<MetaParams>().definition == name) {
            instances.insert(node.id);
        }
    }
    std::vector<Link> moved;
    for (const auto& link : graph.links()) {
        if (instances.contains(link.toNode) || instances.contains(link.fromNode)) {
            moved.push_back(link);
        }
    }
    int removed = 0;
    for (const auto& link : moved) {
        graph.removeLink(link.id);
    }
    for (const auto& link : moved) {
        const int from = instances.contains(link.fromNode) ? slotOf(afterOut, beforeOut, link.fromSlot) : link.fromSlot;
        const int to = instances.contains(link.toNode) ? slotOf(afterIn, beforeIn, link.toSlot) : link.toSlot;
        if (from < 0 || to < 0 || graph.connect(outputPin(link.fromNode, from), inputPin(link.toNode, to))) {
            ++removed;
        }
    }
    return removed;
}

void pruneInterface(MetaDefinition& definition) {
    auto exists = [&](const PinRef& pin) {
        const Node* node = definition.graph.findNode(pin.node);
        if (!node) {
            return false;
        }
        const auto pins = pin.direction == PinDirection::Input ? definition.graph.inputPins(*node)
                                                               : definition.graph.outputPins(*node);
        return pin.slot >= 0 && pin.slot < static_cast<int>(pins.size());
    };
    for (auto* list : {&definition.inputs, &definition.outputs}) {
        for (auto& pin : *list) {
            std::erase_if(pin.targets, [&](const PinRef& target) { return !exists(target); });
        }
        std::erase_if(*list, [](const MetaPin& pin) { return pin.targets.empty(); });
    }
    std::erase_if(definition.params, [&](const MetaParam& param) {
        const Node* node = definition.graph.findNode(param.node);
        return !node || !paramValue(node->params, param.key);
    });
}

std::string uniqueDefinitionName(const Graph& graph, const std::string& base) {
    std::string stem;
    for (char c : base) {
        const bool plain = (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_';
        stem += plain ? c : (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : '_';
    }
    if (stem.empty()) {
        stem = "group";
    }
    std::string name = stem;
    for (int i = 2; graph.findDefinition(name); ++i) {
        name = stem + "_" + std::to_string(i);
    }
    return name;
}

int groupNodes(Graph& graph, const std::vector<int>& nodes, const std::string& name, const std::string& title) {
    if (nodes.empty()) {
        throw std::invalid_argument("nothing to group");
    }
    if (graph.findDefinition(name)) {
        throw std::invalid_argument("a definition named '" + name + "' exists already");
    }
    const std::set<int> group(nodes.begin(), nodes.end());
    Vec2 minimum{1e30f, 1e30f};
    Vec2 sum{};
    for (int id : group) {
        const Node* node = graph.findNode(id);
        if (!node) {
            throw std::invalid_argument("unknown node " + std::to_string(id));
        }
        minimum = {std::min(minimum.x, node->position.x), std::min(minimum.y, node->position.y)};
        sum = {sum.x + node->position.x, sum.y + node->position.y};
    }

    auto definition = std::make_shared<MetaDefinition>();
    definition->name = name;
    definition->title = title;
    for (int id : group) {
        Node copy = *graph.findNode(id);
        copy.position = {copy.position.x - minimum.x, copy.position.y - minimum.y};
        definition->graph.addNodeWithId(std::move(copy));
    }
    // Inside links first; links in and out become the interface.
    std::vector<std::pair<PinRef, std::vector<PinRef>>> incoming;  // outside source -> inner inputs
    std::vector<std::pair<PinRef, std::vector<PinRef>>> outgoing;  // inner output -> outside inputs
    auto add = [](auto& list, const PinRef& key, const PinRef& value) {
        auto it = std::find_if(list.begin(), list.end(), [&](const auto& entry) { return entry.first == key; });
        if (it == list.end()) {
            list.push_back({key, {value}});
        } else {
            it->second.push_back(value);
        }
    };
    // Nested meta nodes resolve through `graph` while the inner links are made.
    for (const auto& [n, d] : graph.definitions()) {
        definition->graph.setDefinition(d);
    }
    for (const auto& link : graph.links()) {
        const bool from = group.contains(link.fromNode);
        const bool to = group.contains(link.toNode);
        const PinRef source = outputPin(link.fromNode, link.fromSlot);
        const PinRef target = inputPin(link.toNode, link.toSlot);
        if (from && to) {
            definition->graph.connect(source, target);
        } else if (to) {
            add(incoming, source, target);
        } else if (from) {
            add(outgoing, source, target);
        }
    }
    for (const auto& [n, d] : graph.definitions()) {
        definition->graph.removeDefinition(n);
    }

    std::set<std::string> taken;
    auto uniquePinName = [&](const std::string& base) {
        std::string pinName = base;
        for (int i = 2; !taken.insert(pinName).second; ++i) {
            pinName = base + " " + std::to_string(i);
        }
        return pinName;
    };
    for (const auto& [source, targets] : incoming) {
        const Node& inner = *definition->graph.findNode(targets[0].node);
        definition->inputs.push_back({uniquePinName(graph.inputPins(inner).at(targets[0].slot).name), targets});
    }
    taken.clear();
    for (const auto& [source, consumers] : outgoing) {
        const Node& inner = *definition->graph.findNode(source.node);
        definition->outputs.push_back({uniquePinName(graph.outputPins(inner).at(source.slot).name), {source}});
    }

    for (int id : group) {
        graph.removeNode(id);
    }
    graph.setDefinition(definition);
    const float count = static_cast<float>(group.size());
    const int meta = graph.addMetaNode(name, {sum.x / count, sum.y / count});
    for (size_t i = 0; i < incoming.size(); ++i) {
        graph.connect(incoming[i].first, inputPin(meta, static_cast<int>(i)));
    }
    for (size_t i = 0; i < outgoing.size(); ++i) {
        for (const auto& consumer : outgoing[i].second) {
            graph.connect(outputPin(meta, static_cast<int>(i)), consumer);
        }
    }
    return meta;
}

std::vector<int> ungroupNode(Graph& graph, int nodeId) {
    const Node* node = graph.findNode(nodeId);
    if (!node || node->kind != NodeKind::Meta) {
        throw std::invalid_argument("not a meta node");
    }
    const MetaDefinition* definition = graph.findDefinition(node->as<MetaParams>().definition);
    if (!definition) {
        throw std::invalid_argument("unknown definition '" + node->as<MetaParams>().definition + "'");
    }
    const Graph inner = instanceGraph(*definition, node->as<MetaParams>());
    const Vec2 origin = node->position;
    const std::vector<MetaPin> inputs = definition->inputs;
    const std::vector<MetaPin> outputs = definition->outputs;

    // The links of the meta node, before it goes.
    std::vector<Link> external;
    for (const auto& link : graph.links()) {
        if (link.fromNode == nodeId || link.toNode == nodeId) {
            external.push_back(link);
        }
    }
    graph.removeNode(nodeId);

    std::map<int, int> ids;
    std::vector<int> added;
    for (const auto& innerNode : inner.nodes()) {
        const int id = graph.addNode(innerNode.kind, {origin.x + innerNode.position.x, origin.y + innerNode.position.y});
        Node& copy = *graph.findNode(id);
        copy.title = innerNode.title;
        copy.params = innerNode.params;
        ids[innerNode.id] = id;
        added.push_back(id);
    }
    for (const auto& link : inner.links()) {
        graph.connect(outputPin(ids.at(link.fromNode), link.fromSlot), inputPin(ids.at(link.toNode), link.toSlot));
    }
    for (const auto& link : external) {
        if (link.toNode == nodeId && link.toSlot < static_cast<int>(inputs.size())) {
            for (const auto& target : inputs[link.toSlot].targets) {
                graph.connect(outputPin(link.fromNode, link.fromSlot), inputPin(ids.at(target.node), target.slot));
            }
        } else if (link.fromNode == nodeId && link.fromSlot < static_cast<int>(outputs.size()) &&
                   !outputs[link.fromSlot].targets.empty()) {
            const PinRef& source = outputs[link.fromSlot].targets[0];
            graph.connect(outputPin(ids.at(source.node), source.slot), inputPin(link.toNode, link.toSlot));
        }
    }
    return added;
}

} // namespace kstudio
