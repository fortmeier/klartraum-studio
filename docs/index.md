# Klartraum Studio

Klartraum Studio is an application for building, inspecting and tuning
[Klartraum](https://fortmeier.github.io/klartraum/docs/) compute graphs. It
opens with a Gaussian splatting graph and renders it live.

![Klartraum Studio showing a stone lantern that turns once around its axis, driven by a Time node in the node graph below it](_static/studio.gif)

## Goals

- **Build and inspect compute graphs visually.** Nodes stand for Klartraum's
  public building blocks, and pins are typed, so invalid links are refused.
- **See results live while editing.** Valid edits are compiled and swapped in
  without restarting.
- **Run one-off graphs.** Parts of a graph that feed a preview or an image
  file are computed on demand.
- **Show what really runs.** The compiled graph view shows exactly the
  elements Klartraum compiled, with their GPU timings.

## Two kinds of output

A graph has two kinds of output, and one graph can use both:

Live
: Everything feeding the *Present* node runs every frame and is shown in the
  window: a Gaussian splatting as it is, or processed further, for example
  resampled, run through ONNX models and turned back into an image.

Run
: Everything feeding a *Preview* or *Image File Writer* is compiled into its
  own Klartraum compute graph and executed once each time you press **Run**
  (F5), or on every change with *Run on every change*.

```{toctree}
:hidden:
:caption: Getting started

self
getting-started
ui-tour
```

```{toctree}
:hidden:
:caption: Reference

nodes/index
graph-format
```

```{toctree}
:hidden:
:caption: Development

development
requirements/index
```
