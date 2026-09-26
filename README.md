# Klartraum Studio

An application for building, inspecting and tuning
[klartraum](https://github.com/fortmeier/klartraum) rendering compute graphs. It
opens with the Gaussian-splatting graph and renders it live.

The studio shows the graph at two levels:

- **Authoring graph** (editable). Nodes stand for klartraum's public building
  blocks: *Scene* (an `.spz` model), *Orbit Camera*, *Swapchain Target*,
  *Gaussian Splatting* (compute or raster backend, all `GsplatConfig` knobs) and
  *Present*. Pins are typed (Gaussians, Camera, Image), and the graph is
  validated as you edit it. Valid changes are compiled into a klartraum compute
  graph and swapped in without restarting. Camera parameters apply live;
  backend, scene and splatting parameters rebuild the pipelines.
- **Compiled graph** (read-only). This is the element DAG klartraum actually
  compiled, found by walking `ComputeGraphElement::getInputs()` from the
  backend's root element. That is the same traversal
  `ComputeGraph::compileFrom()` uses, so every buffer, compute pass, barrier
  and render pass inside the backend is shown. Elements are laid out in
  layers, coloured by category and linked back to the authoring node they were
  built for. With *GPU profiling* on, each element shows its mean GPU time.

Graphs can be saved to and loaded from `*.ktgraph.json` files.

Node editing uses [imgui-node-editor](https://github.com/thedmd/imgui-node-editor)
(zoomable, pannable canvas), built against the Dear ImGui that klartraum
bundles. A one-line patch (`cmake/patches/`) adapts it to ImGui ≥ 1.92.

## Building

Requirements: CMake ≥ 3.24, a C++20 compiler, the Vulkan SDK (loader, headers,
`glslc`), and git. On macOS a Vulkan driver such as MoltenVK or KosmicKrisp is
needed at runtime.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

By default klartraum is fetched from GitHub, including its submodules.
`KLARTRAUM_GIT_TAG` picks the branch or commit. It defaults to
`feature/consumable-library` because the studio needs
[klartraum#22](https://github.com/fortmeier/klartraum/pull/22) (exported include
directories, shader asset root, profiling toggle). Switch it to `develop` once
that PR is merged. To build against a local
checkout instead:

```bash
cmake -S . -B build -DFETCHCONTENT_SOURCE_DIR_KLARTRAUM=/path/to/klartraum
```

The first build takes a while because klartraum builds protobuf for its ONNX
support.

## Running

```bash
./build/klartraum_studio                       # default Gaussian-splatting graph
./build/klartraum_studio my.ktgraph.json       # open a saved graph
./build/klartraum_studio --backend compute --spz path/to/scene.spz --profile
```

The default graph uses the raster backend. The compute (tile-binned) backend
shows the classic projection → binning → sort → gather → bounds → splat
pipeline in the compiled view. On Apple Silicon with MoltenVK it runs at about
one frame per second at window resolution, and the UI is only as responsive as
the frame rate.

You can run it from any directory: klartraum's shaders load from its source
directory, found through `klartraum::setAssetRoot()`. Relative scene paths are
tried against the working directory, then the graph file's directory, then the
klartraum sources, so `3rdparty/spz/samples/racoonfamily.spz` always works.

| Where | Action |
|---|---|
| Scene | left drag orbits, mouse wheel zooms |
| Graph (both tabs) | scroll / two-finger swipe zooms, right-drag pans, F or *Fit* shows everything |
| Authoring graph | right-click the canvas to add a node, right-click a node or link for its menu |
| | drag from pin to pin to connect (invalid links are refused with a reason) |
| | Del / Backspace deletes the selection, *Arrange* re-lays out the graph |
| Inspector | edit the selected node's parameters; errors are listed there |
| Compiled graph | select an element to see its inputs, consumers and timing; *Hide buffers* declutters |
| Menu | File → New/Open/Save (Ctrl+O, Ctrl+S); Graph → auto-apply, arrange |

## Headless snapshots

`klartraum_studio_snapshot` renders the complete UI (scene, editors,
inspector) headlessly into a 512×384 BMP, with no display needed:

```bash
./build/klartraum_studio_snapshot --out authoring.bmp --select gaussian_splatting
./build/klartraum_studio_snapshot --out compiled.bmp --tab compiled --hide-buffers
./build/klartraum_studio_snapshot --out swap.bmp --then-backend compute   # recompiles mid-run
```

## Tests

```bash
ctest --test-dir build --output-on-failure
```

Most tests need no GPU: graph model and validation, serialization, planning,
layout and introspection. `GraphIntrospection.builtGraphMatchesCompiledElements`
builds both Gaussian-splatting backends headlessly. It then checks that the
introspected graph contains exactly the elements klartraum compiled.

## Layout

| Path | Contents |
|---|---|
| `src/studio/graph_model.*` | authoring graph: node kinds, typed pins, links, validation |
| `src/studio/graph_serialization.*` | JSON load/save |
| `src/studio/graph_compiler.*` | authoring graph → plan → klartraum elements |
| `src/studio/graph_introspection.*` | snapshot of a compiled klartraum element DAG |
| `src/studio/graph_layout.*` | layered DAG layout |
| `src/studio/studio_app.*` | the ImGui / imgui-node-editor user interface |
| `src/main.cpp` | command line and frontend setup |
| `tools/studio_snapshot.cpp` | headless UI snapshot tool |

## Limitations

The authoring node set covers what klartraum's public Gaussian-splatting API
can express. A splatting backend renders straight into swapchain images, so
chaining two splatting passes, or compositing one onto another's output, is
reported as a validation error rather than compiled.
