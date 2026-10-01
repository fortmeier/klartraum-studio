# Getting started

## Building

Requirements: CMake ≥ 3.24, a C++20 compiler, the
[Vulkan SDK](https://vulkan.lunarg.com/) (loader, headers, `glslc`) and git.
On macOS, a Vulkan driver such as MoltenVK or KosmicKrisp is needed at
runtime.

```bash
git clone https://github.com/fortmeier/klartraum-studio.git
cd klartraum-studio
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Klartraum itself is fetched from GitHub, including its submodules.
`KLARTRAUM_GIT_TAG` picks the branch or commit (default: `develop`). To build
against a local checkout instead:

```bash
cmake -S . -B build -DFETCHCONTENT_SOURCE_DIR_KLARTRAUM=/path/to/klartraum
```

The first build takes a while, because Klartraum builds protobuf for its ONNX
support.

## Running

```bash
./build/klartraum_studio                                # live Gaussian splatting graph
./build/klartraum_studio --example autoencoder          # image file -> encoder -> decoder -> preview + PNG
./build/klartraum_studio --example splat-autoencoder    # offscreen splatting -> encoder -> decoder -> preview
./build/klartraum_studio --example combined-scenes      # raccoon scene + transformed lantern, merged and rendered live
./build/klartraum_studio --example animated-scenes      # the same on the GPU, the lantern swinging over time
./build/klartraum_studio --example stable-diffusion     # prompt -> Stable Diffusion 1.5 -> preview + PNG
./build/klartraum_studio --example stable-diffusion-background  # lantern splatted live over an SD image
./build/klartraum_studio my.ktgraph.json                # open a saved graph
./build/klartraum_studio --backend compute --spz path/to/scene.spz --profile
```

The same examples are in *File → New from Example*.

The studio can be started from any directory. Relative input paths (scenes,
images, models) are tried against the working directory, then the graph
file's directory, then the Klartraum sources, so the bundled samples always
resolve. Relative output paths are relative to the graph file, or to the
working directory for an unsaved graph.

## The examples

autoencoder
: Loads `data/lantern.jpg`, runs Klartraum's sample encoder and decoder
  (`data/onnx/simple_encoder.onnx`, `simple_decoder.onnx`,
  1×3×128×128 → 1×128×16×16 → 1×3×128×128) and shows the reconstruction.
  Press **Run** (F5); `autoencoded.png` is written next to the graph file.

splat-autoencoder
: The same autoencoder, fed by a Gaussian splatting rendered into an
  offscreen target.

combined-scenes
: Merges the raccoon scene with `data/lantern.spz`, placed on the lawn by a
  *Transform (CPU)* node. The scenes are combined on the CPU and uploaded
  once; moving a scene only re-assembles and re-uploads.

animated-scenes
: Places the lantern on the GPU instead. Its yaw comes from
  *Time* → *Sine* → *Upload Number*, and *Transform (GPU)* and *Merge (GPU)*
  update the Gaussians every frame without rebuilding the graph.

stable-diffusion
: Generates a 256×256 image from a prompt with Stable Diffusion 1.5:
  *Prompt* → *Text Encoder*, *Latent Noise* → *DDIM Sampler* (20 steps) →
  *VAE Decoder* → Preview and `stable_diffusion.png`. It needs the models
  exported into `data/onnx/sd15_denoiser_256` of the Klartraum sources (see
  the {doc}`nodes/index`) and a Klartraum build with Stable Diffusion support.
  Press **Run** (F5). The run takes a while; each denoising step appears in the
  overview.

stable-diffusion-background
: Renders the lantern live over a Stable Diffusion image: a *Composite* draws
  the decoded image (a Run result) onto the swapchain, filling the window, and
  a *Gaussian Splatting* that the orbit camera moves renders over it every
  frame. The first Run starts on its own; press **Run** (F5) for a
  new image, e.g. after changing the prompt.
