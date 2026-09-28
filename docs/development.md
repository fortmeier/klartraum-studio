# Development

## Tests

```bash
ctest --test-dir build --output-on-failure
```

Most tests need no GPU: graph model and validation, serialization, planning,
tensor shapes, ONNX model info, image I/O, layout and introspection. The GPU
tests run headlessly:

- `GraphIntrospection.builtGraphMatchesCompiledElements` checks that the
  introspected graph contains exactly the elements Klartraum compiled, for
  both splatting backends.
- `GraphRunnerTest` executes the example run graphs. The image autoencoder's
  output has to resemble its input, and the offscreen splatting chain has to
  produce a non-black reconstruction.

## Headless snapshots

`klartraum_studio_snapshot` renders the complete UI (scene, editors,
inspector) headlessly into a 512×384 BMP, with no display needed:

```bash
./build/klartraum_studio_snapshot --out authoring.bmp --select gaussian_splatting
./build/klartraum_studio_snapshot --out compiled.bmp --tab compiled --hide-buffers
./build/klartraum_studio_snapshot --out swap.bmp --then-backend compute   # recompiles mid-run
./build/klartraum_studio_snapshot --out run.bmp --example autoencoder --run --select preview
```

With `--run`, the tool presses Run and fails if the run fails.

## Recording the animation on the start page

`tools/record_studio_animation.sh` (macOS) opens
`examples/lantern-turntable.ktgraph.json`, in which a *Time* node turns the
lantern once every 3 seconds, and records the studio window with its title
bar. It then cuts exactly one turn into seamlessly looping animations: an
MP4 video with a poster image (used on klartraum.ai) and a GIF (used here).

```bash
tools/record_studio_animation.sh           # writes docs/_static/studio.mp4, studio-poster.jpg, studio.gif
```

The terminal needs the Screen Recording permission. The window is placed with
`klartraum_studio --window-position X Y` so that it lies fully on the main
display, and it must stay uncovered for the few seconds of the recording.

## Source layout

| Path | Contents |
|---|---|
| `src/studio/graph_model.*` | authoring graph: node kinds, typed pins, links, validation |
| `src/studio/graph_serialization.*` | JSON load and save |
| `src/studio/graph_compiler.*` | authoring graph → live and run plans; the live Klartraum graph |
| `src/studio/tensor_shapes.*` | tensor shape propagation and checks |
| `src/studio/onnx_info.*` | reads ONNX model inputs, outputs and operators |
| `src/studio/graph_runner.*` | builds and executes the run part once, reads back the sinks |
| `src/studio/image_io.*` | image files (stb) and image ↔ tensor conversion |

## Building the documentation

```bash
python3 -m venv docs/.venv
docs/.venv/bin/pip install -r docs/requirements.txt
docs/.venv/bin/sphinx-build -W --keep-going docs docs/_build/html
```

Links into the Klartraum Engine API reference come from the published engine
docs. To build against a local engine docs build instead, set
`KLARTRAUM_INVENTORY=/path/to/klartraum/docs/_build/html/objects.inv`; set it
to `off` to build without engine links.
