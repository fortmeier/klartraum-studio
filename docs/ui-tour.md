# A tour of the user interface

The window shows the rendered scene with three panels on top: the
**Overview**, the **Graph** editor and the **Inspector**.

## Graph editor

The graph editor has two tabs.

Authoring graph
: The graph you edit. Right-click the canvas to add a node, right-click a node
  or link for its menu, and drag from pin to pin to connect. Pins are typed
  (Gaussians on the CPU, Gaussians on the GPU, Camera, Image, Tensor) and
  invalid links are refused with a reason.

Compiled graph
: A read-only view of the element graph Klartraum actually compiled, for the
  live graph or the last run. It is found by walking
  `ComputeGraphElement::getInputs()`, the same traversal
  `ComputeGraph::compileFrom()` uses, so every buffer, compute pass, barrier,
  render pass and ONNX layer is shown. Elements are laid out in layers,
  coloured by category and linked back to the node they were built for. With
  *GPU profiling* on, each element shows its mean GPU time.

## Where things run

Not everything in a graph runs on the GPU or belongs to Klartraum's compute
graph, and the editor shows which is which:

- A badge under each node's title says **where** it runs (`GPU`, `CPU`,
  `CPU->GPU` for uploads, `GPU->CPU` for readbacks) and **what it is made
  of**: elements of the Klartraum graph, a Klartraum function the studio
  calls on the CPU, or studio code.
- A **blue ring** marks nodes that become elements of the live Klartraum
  compute graph, a **green ring** those in the Run graph. Nodes without a ring
  prepare data on the CPU or read results back.
- Gaussians stay CPU data (grey links) until an *Upload Gaussians* node puts
  them into GPU buffers (orange links).
- In the compiled graph, elements the studio adds on its own (layout
  transitions, Present's resample into the swapchain, conversions for
  Preview and Writer inputs, the Run root) are outlined and labelled
  *added by studio*.

## Validation

The graph is validated as you edit it. The checks cover missing inputs and
files, Run reading the window's swapchain images (render into an
*Offscreen Target* for that), tensor shapes propagated through ONNX models,
unsupported ONNX operators, and tensors that are not images feeding a
Preview. Errors are listed in the Inspector.

## Applying changes

Camera parameters apply immediately. Numbers, *Time*, *Sine* and
*Make Transform* values apply without rebuilding the graph. Any other change
rebuilds the pipelines.

## Controls

| Where | Action |
|---|---|
| Scene | left drag orbits, mouse wheel zooms |
| Graph (both tabs) | scroll or two-finger swipe zooms, right-drag pans, F or *Fit* shows everything |
| Authoring graph | right-click the canvas to add a node, right-click a node or link for its menu |
| | drag from pin to pin to connect |
| | Del / Backspace deletes the selection, *Arrange* re-lays out the graph |
| Inspector | edit the selected node's parameters; errors are listed there |
| Compiled graph | select an element to see its inputs, consumers and timing; *Hide buffers* declutters |
| Run | F5, the *Run* buttons, or *Graph → Run on every change* |
| Menu | File → New from Example / Open / Save (Ctrl+O, Ctrl+S); Graph → run, auto-apply, arrange |
