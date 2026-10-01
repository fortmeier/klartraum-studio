# 00003 Review and land the render-over and retained-results work

## State

All of this is uncommitted and waiting for review.

- **klartraum**, uncommitted on `feature/sd15-onnx-inference` (originally meant for
  `feature/render-over-targets`):
  - Swapchain and offscreen targets clear to black by default (can be disabled).
  - `ClearImage`, `ImageViewForward`, `ImageComposite` with `shaders/image/composite.comp`.
  - Splatting renders over its target (raster `setLoadExisting`, compute shader blends
    over the background).
  - `SinglePathImage`, `VulkanContext::submitImmediate`.
  - Docs and `tests/test_render_over_targets.cpp`.
- **studio**, on `feature/retained-results`:
  - Clear Image and Composite nodes; renderers draw only into targets.
  - A tensor linked into an image input gets a Tensor to Image node inserted.
  - Run-only nodes and `RetainedResults`: the live graph reads Run results, with one
    automatic Run per graph revision.
  - Example "Lantern over a Stable Diffusion image (live)", tests and docs.

## To do

- Review by Dirk.
- Decide the klartraum branch for these changes, then commit and open the klartraum PR.
- Point the studio's `KLARTRAUM_GIT_TAG` (CMakeLists.txt) at the merged klartraum branch.
- Commit the studio branch. No PR unless asked.
