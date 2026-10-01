# Work to do

## Current work

- None at the moment.

## Next up

- [Run in the background](00002_background-run.md)
  - The window is usable while a Run works.
  - Progress and cancel; results handed over between frames.
- [Review and land the render-over and retained-results work](00003_land-render-over-and-retained-results.md)
  - Uncommitted in klartraum (on `feature/sd15-onnx-inference`) and in the studio.
  - Choose the branch, then commit; the klartraum PR; update `KLARTRAUM_GIT_TAG`.
- [Remaining validation findings in klartraum](00004_validation-findings.md)
  - Write-after-write hazards in the raster scratch zeroing.
  - `rgba8` storage images on BGRA views.
  - The failing mesh shader test.
- [Requirements for tensors, images and graphs](00005_requirements.md)
  - Deferred until asked for.
