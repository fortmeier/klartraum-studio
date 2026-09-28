# Klartraum Studio

An application for building, inspecting and tuning
[klartraum](https://github.com/fortmeier/klartraum) compute graphs. It opens
with the Gaussian-splatting graph and renders it live.

**Documentation:** [fortmeier.github.io/klartraum-studio](https://fortmeier.github.io/klartraum-studio/) · **Website:** [klartraum.ai](https://klartraum.ai)

A graph has two kinds of output, and one graph can use both:

- **Live:** everything feeding the *Present* node runs every frame and shows
  in the window: a Gaussian splatting as it is, or processed further, for
  example resampled, run through ONNX models and turned back into an image.
  Valid edits are compiled and swapped in without restarting. Camera
  parameters apply immediately; any other change rebuilds the pipelines.
- **Run:** everything feeding a *Preview* or *Image File Writer* is compiled
  into its own klartraum compute graph and executed once each time you press
  **Run** (F5), or on every change with *Run on every change*. Previews show
  the result inside the node and in the Inspector; writers save it as a PNG.

Nodes stand for klartraum's public building blocks:

| Group | Nodes |
|---|---|
| Sources | *Scene* (`.spz` Gaussians into CPU memory, optionally mirrored across Y for Nerfstudio exports), *Image File* (PNG, JPEG, … resized to W×H, as a 1×3×H×W tensor) |
| Gaussians (CPU) | *Transform* (scale, rotate about X/Y/Z, translate; `klartraum::transformGaussians`, which turns orientations and view-dependent colour along), *Merge* (two sets of Gaussians into one, rendered and sorted together), *Upload Gaussians* (into the GPU buffers of a `klartraum::GaussianDataStandard`) |
| Gaussians (GPU) | *Make Transform* (a transform buffer from x, y, z, pitch, yaw, roll and scale; unconnected inputs take the node's values, which apply without rebuilding; `klartraum::TransformBuffer`), *Transform (GPU)* (`klartraum::GaussianTransform`), *Merge (GPU)* (`klartraum::GaussianMerge`), all running every frame |
| Numbers (CPU) | *Number*, *Time* (seconds since start), *Sine* (amplitude · sin(frequency · 2π · x + phase) + offset), evaluated by the studio every frame; *Upload Number* (`klartraum::HostFloat`, copied into a GPU buffer before every frame) |
| Rendering | *Orbit Camera*, *Swapchain Target*, *Offscreen Target* (W×H), *Gaussian Splatting* (compute or raster backend, all `GsplatConfig` settings) |
| Compute | *Image to Tensor* (image → 1×3×H×W tensor), *Tensor to Image* (1×3×H×W tensor → H×W image), *Resample* (image → W×H image, nearest or bilinear; `klartraum::ImageResample`), *ONNX Model* (`klartraum::OnnxNetwork`; Conv, ConvTranspose, Relu, Reshape, Transpose) |
| Outputs | *Present* (live; images other than a swapchain rendering are stretched to the window), *Preview* and *Image File Writer* (run; they take an image tensor or an image) |

Pins are typed (Gaussians on the CPU, Gaussians on the GPU, Camera, Image,
Tensor), and the graph is validated
as you edit it. The checks cover missing inputs and files, Run reading the
window's swapchain images (render into an *Offscreen Target* for it), tensor
shapes propagated through ONNX models
(the model's declared input vs. what it gets), unsupported ONNX operators, and
tensors that are not images feeding a Preview.

### Where things run

Not everything in a graph runs on the GPU or belongs to klartraum's compute
graph, and the editor shows which is which:

- Under each node's title, a badge says **where** it runs (`GPU`, `CPU`,
  `CPU->GPU` for uploads, `GPU->CPU` for readbacks) and **what it is made
  of**: elements of the *klartraum graph*, a *klartraum function* the studio
  calls on the CPU (e.g. `loadGaussiansSpz`, `transformGaussians`), or
  *studio* code. The inspector's *Runs as* section adds when it runs and the
  exact class or function; the add-node menu shows the same.
- A **blue ring** marks nodes that become elements of the live klartraum
  compute graph, a **green ring** those in the Run graph. Nodes without a ring
  prepare data on the CPU or read results back.
- Gaussians stay CPU data (grey links) until an *Upload Gaussians* node puts
  them into GPU buffers (orange links); Gaussian Splatting only accepts
  uploaded Gaussians. Graph files from before the Upload node existed get one
  inserted when they are loaded.
- In the compiled graph, elements the studio adds on its own (layout
  transitions, Present's resample into the swapchain, conversions for
  Preview/Writer inputs, the Run root) are outlined and labelled *added by
  studio*.
- The overview lists the CPU steps of the last live build and the last run
  with their durations: decoding scene files, assembling, uploading, decoding
  images.

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
./build/klartraum_studio --example combined-scenes      # raccoon scene + transformed lantern, merged and rendered live
./build/klartraum_studio --example animated-scenes      # the same on the GPU, the lantern swinging over time
./build/klartraum_studio my.ktgraph.json                # open a saved graph
./build/klartraum_studio --backend compute --spz path/to/scene.spz --profile
```

The same examples are in *File → New from Example*. The autoencoder uses
klartraum's sample models (`data/onnx/simple_encoder.onnx`,
`simple_decoder.onnx`, 1×3×128×128 → 1×128×16×16 → 1×3×128×128) and
`data/lantern.jpg`. Press **Run** (F5) to compute it. The preview shows the
reconstructed image, and `autoencoded.png` is written next to the graph file
(or into the working directory for an unsaved graph).

The combined-scenes example merges the raccoon stump with klartraum's
`data/lantern.spz`, loaded with *Flip Y* and placed on the lawn beside the
stump by a *Transform* node. Transform and Merge work on the Gaussians before
they are uploaded: each Gaussian Splatting input is assembled on the CPU from
its scene files and uploaded once. Scene files stay loaded while the live
graph uses them, so moving a scene only re-assembles and re-uploads.

The animated-scenes example does the placing on the GPU instead: both scenes
are uploaded as they are, a *Make Transform* holds the lantern's placement,
and its yaw comes from *Time* → *Sine* → *Upload Number*. Every frame the
studio evaluates the CPU numbers and sets the uploaded values; klartraum
copies them into the frame's buffers right before submitting it, and the
*Transform (GPU)* and *Merge (GPU)* passes write the Gaussians the splatting
renders. Numbers, Time, Sine and Make Transform values apply without
rebuilding the graph.

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
- Run results are shown by *Preview* nodes, not in the window's background.
- A swapchain rendering has the window's size, which changes with it. A tensor
  made from it only fits a model with fixed input dimensions after a
  *Resample*; a mismatch is reported when the live graph is built.
- ONNX models need a single input. Only the operators klartraum implements are
  supported, and the editor lists any others.
- *Transform* scales uniformly; Gaussians cannot be stretched along one axis.
  Changing a transform rebuilds the live graph (on the CPU, once you let go of
  the control), so it is not an animation tool.
