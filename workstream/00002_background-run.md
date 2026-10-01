# 00002 Run in the background

## Problem

The automatic Run at startup (and every Run) blocks the UI. The window doesn't appear
until the Run is done: about 30 s on MoltenVK and about 70 s on KosmicKrisp for the SD
example.

## Goal

The studio stays usable while a Run works. The live graph shows what it can and picks up
the Run's results when they're ready.

## To do

- Run on a worker thread, on its own queue if the device has one; otherwise interleave
  with frames.
- Hand the results over between frames: `RetainedResults` storage must only change while
  no live graph reads it, and a change of `generation()` triggers the live rebuild.
- Show progress (stage, sampler step), with a way to cancel.
- Define what happens when the graph is edited during a Run: cancel, or discard the
  result.
- Until results exist, render the live graph without the run-only parts, or as it is now
  (black).
