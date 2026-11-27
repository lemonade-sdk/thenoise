# Hunyuan3D 2.1 (shape)

**A picture in, a printable mesh out.**

Hunyuan3D 2.1 is the model that answers the **mesh** medium: it reconstructs the surface in
a reference photo and writes it as `.glb`, `.obj` or `.stl`. A picture is the whole request.

- **One checkpoint** does the whole job (`--dit`): transformer, image conditioner and shape
  codec in a single 6.9 GB file
- Built-in defaults: **30 steps, guidance 5.0**, euler, a 384³ surface cut
- One picture per shape
- Geometry only - positions and shading directions, no texture, no UV layout, no material.
- Meshes come back closed and facing outward, so a slicer can take them as they are
- Meshes shading directions travel in the file, so a shape opens smooth rather than faceted
- **Cleaned up after the cut** - clean the model, reduce triangles, smoothing.

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
| Output | `.glb`, `.obj` or `.stl`, shading directions included, no material |
| Post-processing | clean-up, level of detail, smoothing — each off until asked for |
| Default settings | 4096 tokens, 30 steps, guidance 5.0, euler, 384³ |

## Checkpoints

| Checkpoint | File |
|---|---|
| `--dit` *(the only one there is)* | `hunyuan_3d_v2.1.safetensors` |

## Performance (Strix Halo)

BF16, 768x1024:

- Generate - 4096 tokens, 30 steps, guidance 5.0, euler, 384³: **~55.7 s**

## Examples

*Generated:*

| Ref Image | Result |
|---|---|
| <img width="1024" alt="thenoise_edit_2673176517" src="https://github.com/user-attachments/assets/3a7f4fc2-753e-481e-a46d-6c77dcb9ae29" /> | <img width="1024" alt="thenoise_edit_3754200251" src="https://github.com/user-attachments/assets/9fdc5bd6-68e8-412f-bbbe-7332f48eec86" /> |


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

## After the cut

Nothing is trimmed on the way out: the default 384³ cut of one real subject left 4.3 million
triangles and a 102 MB `.glb`. Three settings — clean-up, level of detail and smoothing — let
you change that afterwards.

```bash
./thenoise.sh mesh \
  --dit ./models/hunyuan3d-2.1/hunyuan_3d_v2.1.safetensors \
  --image chair.png --out chair.glb --floaters 0.05 --triangles 0.25 --smooth 0.4
```

- `--floaters 0.05` throws away every piece spreading less than 5 % as wide as the biggest
  one: the specks and shells a cut sometimes leaves hanging on. `0` — the default — keeps
  everything, which is what you want while judging the shape itself.
- `--triangles 0.25` keeps a quarter of the triangles, and roughly a quarter of the file: 25 MB
  rather than 102. Enough for a preview, and usually enough for a print.
- `--smooth 0.4` eases out the bumpiness a cut leaves. Corners are held, so a hard edge stays
  hard instead of rounding off.

A face too thin to be drawn is left out of the file rather than written invisible, so what a
viewer shows is what a slicer gets.

For every flag (views, grid, level, scale, clean-up, LoRAs, containers), see the
[CLI reference](../cli.md) and the [HTTP API reference](../api.md).
