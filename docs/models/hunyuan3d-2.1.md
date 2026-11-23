# Hunyuan3D 2.1 (shape)

**A picture in, a printable mesh out.**

Hunyuan3D 2.1 is the model that answers the **mesh** medium: it reconstructs the surface in
a reference photo and writes it as `.glb`, `.obj` or `.stl`. A picture is the whole request.

- **One checkpoint** does the whole job (`--dit`): transformer, image conditioner and shape
  codec in a single 6.9 GB file
- Built-in defaults: **30 steps, guidance 5.0**, euler, a 384³ surface cut
- One picture per shape; a second is refused rather than quietly ignored
- Geometry only
- Meshes come back closed and facing outward, so a slicer can take them as they are

> Commands below assume a dev checkout (`./thenoise.sh`, `.venv/bin/python`).
> On a [portable bundle](../setup.md) use `./bin/thenoise` and `./bin/python3` instead.

## Specs

| Spec | Data |
|---|---|
| Architecture | image-conditioned DiT, 21 blocks over unstructured latent tokens |
| Download size | ~6.9 GB fp16, in one file |
| Modes | picture-to-shape only |
| Conditioning | the reference photo, at its whole 518² patch grid |
| Sizes | the shape fills the volume it is sampled from; `--scale` sets the units |
| Text encoder | — |
| Editing | — |
| Output | `.glb`, `.obj` or `.stl`, no material |
| Default settings | 4096 tokens, 30 steps, guidance 5.0, euler, 384³ |

## Checkpoints

| Checkpoint | File |
|---|---|
| `--dit` *(the only one there is)* | `hunyuan_3d_v2.1.safetensors` |


## Download

```bash
# Comfy-Org's single-file repackage, into ./models/hunyuan3d-2.1/
.venv/bin/python scripts/download.py --model hunyuan3d21
```

## Usage

```bash
./thenoise.sh mesh \
  --dit ./models/hunyuan3d-2.1/hunyuan_3d_v2.1.safetensors \
  --image chair.png --out chair.glb
```

Serve over HTTP with the web UI (open <http://localhost:8000/>):

```bash
./thenoise.sh serve --dit ./models/hunyuan3d-2.1/hunyuan_3d_v2.1.safetensors
```

In the web UI a shape is the Generate tab: a dropzone for the picture, the density and
the seed. Over HTTP the picture travels base64-encoded in the `image`
field — `POST /mesh` answers with the file itself, `POST /jobs/mesh` queues it and
`GET /jobs/{id}/content` reads it back:

```bash
curl -s -X POST localhost:8000/mesh \
  -H 'Content-Type: application/json' \
  -d "{\"image\": \"$(base64 -w0 chair.png)\", \"output_format\": \"stl\"}" \
  --output chair.stl
```

## Density, bounds, units

- **`--octree-resolution` is what a shape costs.** 96, 192, 384, 768 — each step doubles
  the grid, which is eight times the cells, a finer surface and a bigger file. 768 is the
  ceiling, and it is the slowest thing here by a distance: start at 192.
- **`--bounds`** is the half-width of the volume the surface is cut from, and the default
  1.01 leaves a margin. An object that touches the edge gets clipped there, and nothing
  downstream can tell you it happened.
- **`--scale`** multiplies the coordinates on the way out, because no mesh format says what
  a unit is: an unscaled shape is about two units wide, which is neither centimetres nor
  inches until you decide.
- **`--mc-level`** picks which surface of the field to cut. The default is the reference
  pipeline's own cut and rarely wants touching.
- `.stl` gives every triangle its own three corners and its own normal, which is exactly
  what the cut produces; a slicer welds them to its tolerance. A few numerically flat
  triangles are kept rather than dropped, because dropping one opens a hole where its
  neighbour's edge used to be shared.
