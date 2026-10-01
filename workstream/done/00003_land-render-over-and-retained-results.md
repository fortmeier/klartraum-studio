# 00003 Review and land the render-over and retained-results work

## State

Status: done. klartraum PR #38 is merged into `feature/sd15-onnx-inference`, and the studio branch is merged into `main`.

- **klartraum**, PR #38 (`feature/render-over-targets`):
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

## Outcome

- klartraum #38 is merged; the local klartraum copy is cleaned up (merged branches and stashes removed).
- The studio's `KLARTRAUM_GIT_TAG` tracks `feature/sd15-onnx-inference` again.
- `feature/retained-results` is merged into the studio's `main`.
