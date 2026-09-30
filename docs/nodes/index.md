# Node reference

Nodes are grouped as in the add-node menu. *Runs* says when a node does its
work; *Implemented by* names the Klartraum class or function behind it.

## Sources

| Node | Description | Implemented by | Runs |
|---|---|---|---|
| Scene | Loads a 3D Gaussian model from an `.spz` file into CPU memory, optionally mirrored across Y for Nerfstudio exports. | `klartraum::loadGaussiansSpz` | once, when the graph is built |
| Image File | Loads an image file (PNG, JPEG, …), resized to W×H, as a 1×3×H×W tensor with values in [0, 1]. | {cpp:class}`klartraum::TensorElement` | once, when the graph is built |

## Numbers (CPU)

| Node | Description | Implemented by | Runs |
|---|---|---|---|
| Number | A constant number. | studio | every frame (live), once per run |
| Time | Seconds since the studio started, times a speed. | studio | every frame (live), once per run |
| Sine | amplitude · sin(frequency · 2π · x + phase) + offset, e.g. to swing something back and forth. | studio | every frame (live), once per run |
| Upload Number | Copies a CPU number into a GPU buffer before every frame, without rebuilding the graph. | `klartraum::HostFloat` | every frame (live), once per run |

## Gaussians (CPU)

| Node | Description | Implemented by | Runs |
|---|---|---|---|
| Transform (CPU) | Scales, rotates and moves Gaussians in CPU memory; orientations and view-dependent colour turn along. | `klartraum::transformGaussians` | once, when the graph is built |
| Merge (CPU) | Combines two sets of Gaussians in CPU memory into one, rendered and sorted together. | studio | once, when the graph is built |
| Upload Gaussians | Uploads Gaussians into the GPU buffers a Gaussian Splatting reads. | {cpp:class}`klartraum::GaussianDataStandard` | once, when the graph is built |

## Gaussians (GPU)

| Node | Description | Implemented by | Runs |
|---|---|---|---|
| Make Transform | A GPU transform from translation, rotation (degrees about X, then Y, then Z) and scale. Inputs that are not connected take the node's values, which apply without rebuilding. | {cpp:class}`klartraum::TransformBuffer` | every frame (live) or every run |
| Transform (GPU) | Moves Gaussians on the GPU by a transform, every frame. | {cpp:class}`klartraum::GaussianTransform` | every frame (live) or every run |
| Merge (GPU) | Combines two sets of Gaussians on the GPU, every frame. | {cpp:class}`klartraum::GaussianMerge` | every frame (live) or every run |

## Rendering

| Node | Description | Implemented by | Runs |
|---|---|---|---|
| Orbit Camera | Camera uniform buffer driven by an orbit camera. | {cpp:class}`klartraum::InterfaceCameraOrbit` | every frame (live), once per run |
| Swapchain Target | The window's swapchain images, rendered into directly. | {cpp:class}`klartraum::ImageViewSrc` | every frame (live only) |
| Offscreen Target | An image of fixed size (W×H) to render into for further processing. | `klartraum::OffscreenTarget` | every frame (live) or every run |
| Gaussian Splatting | Renders the Gaussians into the target image, with the compute or raster backend and all {cpp:struct}`klartraum::GsplatConfig` settings. | `klartraum::createGaussianSplatting` | every frame (live) or every run |

## Compute

| Node | Description | Implemented by | Runs |
|---|---|---|---|
| Image to Tensor | Converts a rendered image into a 1×3×H×W tensor. | {cpp:class}`klartraum::GeneralComputation` | every frame (live) or every run |
| Tensor to Image | Converts a 1×3×H×W tensor (values in [0, 1]) into an H×W image. | {cpp:class}`klartraum::GeneralComputation` | every frame (live) or every run |
| Resample | Resamples an image to W×H, nearest or bilinear. | {cpp:class}`klartraum::ImageResample` | every frame (live) or every run |
| ONNX Model | Runs an ONNX network. It has one tensor pin per model input and output, named like them; they follow the model file, and links to pins a new model no longer has are removed. The operators Klartraum executes include Conv, ConvTranspose, MatMul, Gemm, Softmax, the normalizations, elementwise arithmetic, Reshape, Transpose, Slice, Concat and Resize. | {cpp:class}`klartraum::OnnxNetwork` | every frame (live) or every run |

Tensor pins carry any tensor. Its shape and element type (float32 or int64)
are checked where it is consumed: an ONNX model's inputs must match the
model's, layers and sinks take float32, and the node shows each output's
shape (and element type, unless float32).

## Layers

Single network layers on float32 tensors, built with the same
`klartraum::layers` factories the ONNX models use.

| Node | Description | Implemented by | Runs |
|---|---|---|---|
| Add, Subtract, Multiply, Divide | A op B elementwise; the shapes broadcast (NumPy rules). If B is not connected, the node's *b* is used as a one-element tensor. | `klartraum::layers::binary` | every frame (live) or every run |
| ReLU, Sigmoid, Sqrt | Elementwise; the output has the input's shape. | `klartraum::layers::relu`, `unary` | every frame (live) or every run |
| Softmax | Softmax over the last axis. | `klartraum::layers::softmax` | every frame (live) or every run |

## Stable Diffusion

Stable Diffusion 1.5 text-to-image generation, with the models that Klartraum's
`scripts/sd15_onnx/export_denoiser.py` exports. The models are fixed-size, so
one export serves one image size. They are not part of the Klartraum
repository. Export them with `uv run python export_denoiser.py --size 256
--onnx-dir ../../data/onnx/sd15_denoiser_256` (see the script's README).
The *Stable Diffusion 1.5* example expects them in that directory.

| Node | Description | Implemented by | Runs |
|---|---|---|---|
| Prompt | A prompt and a negative prompt, tokenized with CLIP's byte-pair encoding into two 2×77 int64 tensors, *Ids* (negative prompt first) and *Mask* (the attention mask). The *Vocabulary* is the export's `vocab.json`, with `merges.txt` next to it. | {cpp:class}`klartraum::ClipTokenizer`, uploaded into two {cpp:class}`klartraum::TensorElement` | once, when the graph is built |
| Text Encoder | Built-in meta node: an ONNX Model (the CLIP text encoder) that turns the Prompt's *Ids* and *Mask* into 2×77×768 embeddings. Exposes *Model*. | {cpp:class}`klartraum::OnnxNetwork` | every run |
| Latent Noise | Gaussian noise of the latent size for a W×H image (1×4×H/8×W/8), from a seed. Optionally it reads raw float32 latents from a file instead, e.g. the export's `initial_latents_f32.bin` to reproduce its reference image. | studio | every run |
| DDIM Sampler | Denoises the latents in *Steps* DDIM steps with SD 1.5's scheduler. Each step runs the UNet once on the negative and the positive prompt, blends the two noise estimates with the *Guidance* scale and takes the DDIM step on the CPU. | {cpp:class}`klartraum::OnnxNetwork`, submitted once per step | every run, once per denoising step |
| VAE Decoder | Built-in meta node: Multiply (undoes the latent scaling) → ONNX Model (the VAE decoder) → Multiply and Add (maps [-1, 1] to [0, 1]). Outputs a 1×3×H×W image tensor for a Preview or an Image File Writer. Exposes *Model* and *Latent scale*. | `klartraum::layers`, {cpp:class}`klartraum::OnnxNetwork` | every run |

The DDIM Sampler is a *staged* node and only runs with **Run**. Run executes
the Klartraum graph of everything before it, reads the tensors it needs back
and runs the UNet once per step. It then uploads the result for the nodes that
follow, whose graph it executes next. The compiled graph of a run shows every
stage's graph together with the UNet. Run waits for all denoising steps; the
overview lists each step with its duration.

## Meta nodes

A meta node stands for a subgraph, its *definition*. It has the pins and
parameters the definition exposes; each stands for a pin or a parameter of an
inner node. Definitions can contain meta nodes, to any depth.

- **Built-in definitions** come with the studio (Text Encoder, VAE Decoder)
  and appear in the add-node menu in their group. They are read-only; *Duplicate
  into this graph* in the opened definition makes an editable copy that the node
  then uses.
- **Group** (Ctrl+G, or the node's context menu) turns the selected nodes into
  a meta node with a new definition stored in the graph file. Links into the
  group become exposed inputs, one per outside output feeding every inner pin
  it fed; links out of it become exposed outputs. *Ungroup* replaces a meta node
  by its inner nodes.
- **Open** (double-click, the context menu or the inspector) shows the
  definition's inner graph; the path back is shown above the editor. Changes
  apply to every node using the definition. In the inspector of an inner node,
  *Meta node interface* exposes or hides its pins and parameters; links of
  instances follow exposed pins by name.
- A meta node's parameters are edited in its inspector; *Reset* returns one to
  the definition's value.

Meta nodes are expanded into their inner nodes before the graph is checked and
compiled, so they run exactly like the nodes they contain. Problems inside
one are reported on the meta node, prefixed with the inner node's title
("VAE Decoder / Decoder: ..."), and its compiled elements belong to it.

## Outputs

| Node | Description | Implemented by | Runs |
|---|---|---|---|
| Present | Shows the image in the window every frame, stretched to the window's size. | {cpp:class}`klartraum::ImageResample`, added by the studio | every frame (live only) |
| Preview | Shows a 1- or 3-channel image tensor, or an image, when the graph is run. | studio (read back, shown as an ImGui texture) | every run |
| Image File Writer | Writes a 1- or 3-channel image tensor, or an image, to a PNG file when the graph is run. | studio (read back, written with stb_image_write) | every run |
