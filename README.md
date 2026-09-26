# Klartraum Studio

An application for building, inspecting and tuning
[klartraum](https://github.com/fortmeier/klartraum) compute graphs. It opens
with the Gaussian-splatting graph and renders it live.

A graph has two kinds of output, and one graph can use both:

- **Live:** everything feeding the *Present* node renders into the window
  every frame. Valid edits are compiled and swapped in without restarting.
  Camera parameters apply immediately; backend, scene and splatting parameters
  rebuild the pipelines.
- **Run:** everything feeding a *Preview* or *Image File Writer* is compiled
  into its own klartraum compute graph and executed once each time you press
  **Run** (F5), or on every change with *Run on every change*. Previews show
  the result inside the node and in the Inspector; writers save it as a PNG.

Nodes stand for klartraum's public building blocks:

| Group | Nodes |
|---|---|
| Sources | *Scene* (`.spz` Gaussians), *Image File* (PNG, JPEG, … resized to W×H, as a 1×3×H×W tensor) |
| Rendering | *Orbit Camera*, *Swapchain Target*, *Offscreen Target* (W×H), *Gaussian Splatting* (compute or raster backend, all `GsplatConfig` settings) |
| Compute | *Image to Tensor* (offscreen image → 1×3×H×W tensor), *ONNX Model* (`klartraum::OnnxNetwork`; Conv, ConvTranspose, Relu, Reshape, Transpose) |
| Outputs | *Present* (live), *Preview* and *Image File Writer* (run) |

Pins are typed (Gaussians, Camera, Image, Tensor), and the graph is validated
as you edit it. The checks cover missing inputs and files, a swapchain image
where an offscreen one is needed, tensor shapes propagated through ONNX models
(the model's declared input vs. what it gets), unsupported ONNX operators, and
tensors that are not images feeding a Preview.

The **compiled graph** view is read-only. It shows the element DAG klartraum
actually compiled, for the live graph or the last run, found by walking
`ComputeGraphElement::getInputs()`. That is the same traversal
`ComputeGraph::compileFrom()` uses, so every buffer, compute pass, barrier,
render pass and ONNX layer is shown. Elements are laid out in layers, coloured
by category and linked back to the node they were built for. With *GPU
profiling* on, each element shows its mean GPU time.

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
`KLARTRAUM_GIT_TAG` picks the branch or commit (default: `develop`). To build against a local checkout instead:

```bash
cmake -S . -B build -DFETCHCONTENT_SOURCE_DIR_KLARTRAUM=/path/to/klartraum
```

The first build takes a while because klartraum builds protobuf for its ONNX
support.

## Running

```bash
./build/klartraum_studio                                # live Gaussian-splatting graph
./build/klartraum_studio --example autoencoder          # image file -> encoder -> decoder -> preview + PNG
./build/klartraum_studio --example splat-autoencoder    # offscreen splatting -> encoder -> decoder -> preview
./build/klartraum_studio my.ktgraph.json                # open a saved graph
./build/klartraum_studio --backend compute --spz path/to/scene.spz --profile
```

The same examples are in *File → New from Example*. The autoencoder uses
klartraum's sample models (`data/onnx/simple_encoder.onnx`,
`simple_decoder.onnx`, 1×3×128×128 → 1×128×16×16 → 1×3×128×128) and
`data/lantern.jpg`. Press **Run** (F5) to compute it. The preview shows the
reconstructed image, and `autoencoded.png` is written next to the graph file
(or into the working directory for an unsaved graph).

The default graph uses the raster backend. The compute (tile-binned) backend
shows the classic projection → binning → sort → gather → bounds → splat
pipeline in the compiled view. On Apple Silicon with MoltenVK it runs at about
one frame per second at window resolution, and the UI is only as responsive as
the frame rate.

You can run it from any directory: klartraum's shaders load from its source
directory, found through `klartraum::setAssetRoot()`. Relative input paths
(scenes, images, models) are tried against the working directory, then the
graph file's directory, then the klartraum sources, so the bundled samples
always resolve. Relative output paths are relative to the graph file, or to
the working directory for an unsaved graph.

| Where | Action |
|---|---|
| Scene | left drag orbits, mouse wheel zooms |
| Graph (both tabs) | scroll / two-finger swipe zooms, right-drag pans, F or *Fit* shows everything |
| Authoring graph | right-click the canvas to add a node, right-click a node or link for its menu |
| | drag from pin to pin to connect (invalid links are refused with a reason) |
| | Del / Backspace deletes the selection, *Arrange* re-lays out the graph |
| Inspector | edit the selected node's parameters; errors are listed there |
| Compiled graph | select an element to see its inputs, consumers and timing; *Hide buffers* declutters |
| Run | F5, the *Run* buttons, or *Graph → Run on every change* |
| Menu | File → New from Example / Open / Save (Ctrl+O, Ctrl+S); Graph → run, auto-apply, arrange |

## Headless snapshots

`klartraum_studio_snapshot` renders the complete UI (scene, editors,
inspector) headlessly into a 512×384 BMP, with no display needed:

```bash
./build/klartraum_studio_snapshot --out authoring.bmp --select gaussian_splatting
./build/klartraum_studio_snapshot --out compiled.bmp --tab compiled --hide-buffers
./build/klartraum_studio_snapshot --out swap.bmp --then-backend compute   # recompiles mid-run
./build/klartraum_studio_snapshot --out run.bmp --example autoencoder --run --select preview
```

With `--run` the tool presses Run and fails if the run fails.

## Tests

```bash
ctest --test-dir build --output-on-failure
```

Most tests need no GPU: graph model and validation, serialization, planning,
tensor shapes, ONNX model info, image I/O, layout and introspection. The GPU
tests run headlessly:
- `GraphIntrospection.builtGraphMatchesCompiledElements` checks that the
  introspected graph contains exactly the elements klartraum compiled, for both
  splatting backends.
- `GraphRunnerTest` executes the example run graphs. The image autoencoder's
  output has to resemble its input, and the offscreen splatting chain has to
  produce a non-black reconstruction.

## Layout

| Path | Contents |
|---|---|
| `src/studio/graph_model.*` | authoring graph: node kinds, typed pins, links, validation |
| `src/studio/graph_serialization.*` | JSON load/save |
| `src/studio/graph_compiler.*` | authoring graph → live and run plans; the live klartraum graph |
| `src/studio/tensor_shapes.*` | tensor shape propagation and checks |
| `src/studio/onnx_info.*` | reads ONNX model inputs, outputs and operators |
| `src/studio/graph_runner.*` | builds and executes the run part once, reads back the sinks |
| `src/studio/image_io.*` | image files (stb) and image ↔ tensor conversion |
| `src/studio/preview_texture.*` | uploads run results as ImGui textures |
| `src/studio/graph_introspection.*` | snapshot of a compiled klartraum element DAG |
| `src/studio/graph_layout.*` | layered DAG layout |
| `src/studio/studio_app.*` | the ImGui / imgui-node-editor user interface |
| `src/main.cpp` | command line and frontend setup |
| `tools/studio_snapshot.cpp` | headless UI snapshot tool |

## Limitations

- A splatting backend renders straight into a target image. Chaining two
  splatting passes, or compositing one onto another's output, is reported as a
  validation error rather than compiled.
- *Present* shows swapchain renderings only. Run results are shown by
  *Preview* nodes, not in the window's background.
- ONNX models need a single input. Only the operators klartraum implements are
  supported, and the editor lists any others.
