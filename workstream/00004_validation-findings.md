# 00004 Remaining validation findings in klartraum

These were found with sync validation (see 00001). They are older than the render-over
work and are not known to cause visible errors.

## Write-after-write hazards in the raster scratch zeroing

`vkCmdFillBuffer` writes scratch buffers that an earlier `vkCmdFillBuffer` wrote, with no
barrier between them (sizes 64, 1284 and 20480 bytes). It shows in the plain splatting
example too. Both writes are zeros, but they are unsynchronised. Add a transfer → transfer
barrier, or zero each buffer once per frame.

## Storage image format mismatch

`Undefined-Value-StorageImage-FormatMismatch-ImageView`: shaders declare `rgba8`, but the
swapchain views are `B8G8R8A8`. It affects the resample, compute splatting and composite
shaders. Use views in a matching format, or format-less storage images
(`shaderStorageImageWriteWithoutFormat`).

## Failing mesh shader test

`GaussianSplattingRaster.meshShaderPathMatchesVertexPath` fails, also without the
render-over changes. Find out why.
