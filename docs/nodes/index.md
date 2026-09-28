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
| ONNX Model | Runs an ONNX network on a tensor (Conv, ConvTranspose, Relu, Reshape, Transpose). | {cpp:class}`klartraum::OnnxNetwork` | every frame (live) or every run |

## Outputs

| Node | Description | Implemented by | Runs |
|---|---|---|---|
| Present | Shows the image in the window every frame, stretched to the window's size. | {cpp:class}`klartraum::ImageResample`, added by the studio | every frame (live only) |
| Preview | Shows a 1- or 3-channel image tensor, or an image, when the graph is run. | studio (read back, shown as an ImGui texture) | every run |
| Image File Writer | Writes a 1- or 3-channel image tensor, or an image, to a PNG file when the graph is run. | studio (read back, written with stb_image_write) | every run |
