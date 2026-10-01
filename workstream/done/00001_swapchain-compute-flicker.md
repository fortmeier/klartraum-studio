# 00001 Swapchain flicker with compute writes

Status: done. Worked around in the studio on MoltenVK with a swapchain stand-in.

## Problem

In the Stable Diffusion background example (lantern over the SD image), black frames
appear continuously after resizing the window or switching to fullscreen. In those frames
the lantern and the UI render, but the SD background is missing.

## Findings

- A real synchronisation bug was found with sync validation and fixed: the render pass
  that loads its target ([klartraum] `renderpass.hpp`, `setLoadExisting`) lacked
  `COLOR_ATTACHMENT_READ` in its subpass dependency. That was not the resize flicker.
- Tests that narrowed it down (all on MoltenVK):
  - Plain splatting: no flicker (its background is black anyway, so it proves little).
  - Red Clear Image → splatting: no flicker. Clears and render passes are fine.
  - Offscreen splat → Present (compute resample into the swapchain): **flickers**. This
    path is older than the render-over work.
- Cause: on MoltenVK, a descriptor set that points at a swapchain image captures that
  frame's Metal drawable when the set is written. Our graphs write descriptor sets once,
  at build time, so compute shaders writing the swapchain (Composite, Present's resample,
  the compute splatting backend) sometimes write into a drawable that isn't presented.
  - With `MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS=0`, it crashes in
    `setTexture:atIndex:` instead.
  - Rebuilding the graph once does not help; it only captures a new set of drawables.
- On **KosmicKrisp** (`libkosmickrisp_icd.json` in the Vulkan SDK) there is no flicker,
  including the full SD background example. After the klartraum fixes on
  `feature/sd15-onnx-inference` (single command buffer per path, attention threadgroup
  memory), the SD Run there takes about 70 s, against about 30 s on MoltenVK with the
  older build.

## Decision

klartraum's core stays as it is. On MoltenVK, the studio's live graphs render into a stand-in:

- `src/studio/swapchain_stand_in.{hpp,cpp}`: `needsSwapchainStandIn` (MoltenVK driver ID, windowed
  only) and `SwapchainCopy`, which copies the stand-in into the acquired swapchain image and
  leaves it ready for presenting.
- `graph_runner.cpp`: the Swapchain Target becomes an `OffscreenTarget` per path (named
  "Swapchain stand-in"), and Present ends with the copy, including after its resample.
- The cost is one full-window image copy per frame.
- The resample test graph is stable through resizing and fullscreen on MoltenVK.

## Outcome

- The resample test graph and the SD background example are stable through resizing and
  fullscreen on MoltenVK.
- The Draw Basics node, added for this investigation, stays at the end of the SD example.
- The driver stays the default (MoltenVK); KosmicKrisp works too and needs no copy.
- Also fixed: the Composite's mode and fit were not saved in graph files.

## Notes

- Sync validation (the layer prints to **stdout** without a debug messenger, so capture
  both streams):
  ```bash
  SDK=~/VulkanSDK/1.4.357.1/macOS VK_LAYER_PATH=$SDK/share/vulkan/explicit_layer.d \
  VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation VK_KHRONOS_VALIDATION_VALIDATE_SYNC=1 \
  VK_KHRONOS_VALIDATION_ENABLE_MESSAGE_LIMIT=false DYLD_LIBRARY_PATH=$SDK/lib \
  ./build-sd/klartraum_studio --example stable-diffusion-background > validation.log 2>&1
  ```
- KosmicKrisp:
  ```bash
  SDK=~/VulkanSDK/1.4.357.1/macOS VK_DRIVER_FILES=$SDK/share/vulkan/icd.d/libkosmickrisp_icd.json \
  DYLD_LIBRARY_PATH=$SDK/lib ./build-sd/klartraum_studio --example stable-diffusion-background
  ```
- The headless snapshot tool waits for the GPU after every frame, so frames never overlap
  there and it cannot show this kind of bug.
