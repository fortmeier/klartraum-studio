# 00003 Review and land the render-over and retained-results work

## State

Both parts are committed and wait for review: klartraum in PR #38, the studio on its branch.

- **klartraum**, on `feature/render-over-targets` (PR #38). Your local checkout still has the
  same changes uncommitted on `feature/sd15-onnx-inference`, which the studio builds against:
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

- Review by Dirk:
  - klartraum PR #38 (`feature/render-over-targets` → `feature/sd15-onnx-inference`).
  - The studio branch `feature/retained-results` (no PR).
- After klartraum #38 is merged, point the studio's `KLARTRAUM_GIT_TAG` back at the merged branch
  (it names `feature/render-over-targets` for now).
- Merge the studio branch into `main`.
