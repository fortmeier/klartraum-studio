# Graph file format

Graphs are saved as JSON files with the extension `.ktgraph.json`.

```json
{
  "format": "klartraum-studio-graph",
  "version": 1,
  "nodes": [
    {
      "id": 1,
      "kind": "scene",
      "title": "Scene",
      "position": [0.0, 0.0],
      "params": { "path": "3rdparty/spz/samples/racoonfamily.spz", "flipY": false }
    },
    {
      "id": 2,
      "kind": "camera",
      "title": "Orbit Camera",
      "position": [0.0, 150.0],
      "params": { "azimuth": 0.9, "elevation": -0.5, "distance": 1.0, "target": [-0.5, 0.0, 0.5], "up": "y" }
    }
  ],
  "links": [
    { "from": [1, 0], "to": [4, 0] }
  ]
}
```

## Top level

| Key | Meaning |
|---|---|
| `format` | Always `"klartraum-studio-graph"`; other files are rejected. |
| `version` | Format version, currently `1`. Files from a newer version are rejected. |
| `nodes` | The nodes of the authoring graph. |
| `links` | The connections between node pins. |

## Nodes

| Key | Meaning |
|---|---|
| `id` | Unique integer ID within the file. |
| `kind` | The node kind (see below). |
| `title` | The title shown in the editor. |
| `position` | Position on the canvas, `[x, y]`. |
| `params` | The node's parameters; which keys exist depends on the kind. Missing keys take their defaults. |

Node kinds: `scene`, `image_file`, `number`, `time`, `sine`, `upload_number`,
`transform_gaussians`, `merge_gaussians`, `upload_gaussians`,
`make_transform`, `transform_gaussians_gpu`, `merge_gaussians_gpu`, `camera`,
`swapchain_target`, `offscreen_target`, `gaussian_splatting`,
`image_to_tensor`, `tensor_to_image`, `resample`, `onnx_model`, `sd_prompt`,
`sd_text_encoder`, `sd_latent_noise`, `sd_ddim_sampler`, `sd_vae_decoder`,
`present`, `preview`, `image_file_writer`. See the {doc}`nodes/index` for what
they do.

## Links

Each link connects an output pin to an input pin, written as
`[node id, pin index]`:

```json
{ "from": [5, 0], "to": [6, 0] }
```

Graph files from before the *Upload Gaussians* node existed get one inserted
when they are loaded.
