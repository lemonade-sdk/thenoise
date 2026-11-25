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

