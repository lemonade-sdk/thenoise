# Hunyuan3D 2.1 (shape)

One checkpoint answers the **mesh** medium: a picture goes in, a triangle mesh comes
out. It is image-conditioned only — there is no text encoder in the file and no prompt
takes a shape anywhere it would otherwise not go.

| | Hunyuan3D 2.1 shape |
|---|---|
| Modes | image-to-shape (`i2d`) only; text-to-shape does not exist |
| Checkpoint | **one file** (`--dit`): transformer + DINOv2 image conditioner + shape codec |
| Conditioning | the reference picture's DINOv2-L/14 patches, whole and aspect-preserved |
| Latents | 4096 unstructured tokens of 64 channels, no positional embedding |
| Output | `.glb` (binary glTF), `.obj` or `.stl` — positions and triangles, no material |
| Defaults | 4096 tokens, 30 steps, CFG 5.0, Euler, 384³ octree, seed random |

Download it with `scripts/download.py --model hunyuan3d` (Comfy-Org's single-file
repackage, `hunyuan_3d_v2.1.safetensors`, 6.9 GB fp16). Point `--dit` at that one file and
leave `--vae` / `--text-encoder` alone: there is nothing else to load.

> **The weights are not free to use everywhere.** They are released under the
> [Tencent Hunyuan 3D community licence](https://github.com/Tencent/Hunyuan3D-2/blob/main/LICENSE),
> which excludes the EU, the UK and South Korea, and is non-commercial. The code here is
> Apache-2.0 and independent of Tencent's; the weights are yours to be entitled to.

## What to send

```bash
.venv/bin/python -m thenoise mesh \
  --dit models/hunyuan3d/hunyuan_3d_v2.1.safetensors \
  --image chair.png --out chair.glb
```

`serve` answers the same request on `POST /mesh` (the file comes back in the response) and
`POST /jobs/mesh` (queued; read it back from `/jobs/{id}/content`). The picture is sent
base64-encoded in the `image` field and is verified before the job is accepted.

```bash
curl -s -X POST localhost:8000/mesh \
  -H 'Content-Type: application/json' \
  -d "{\"image\": \"$(base64 -w0 chair.png)\", \"output_format\": \"obj\"}" \
  --output chair.obj
```

A picture is the whole request: a request without one is refused rather than answered with
a blob, since nothing else conditions this model.

## The knobs

| Flag | Field | Default | What it does |
|---|---|---|---|
| `--image` | `image` | — | the picture to reconstruct; the only conditioning |
| `--latents` | `latents` | 4096 | latent tokens to denoise; the model was trained at its default |
| `--steps` | `steps` | 30 | denoise steps, over a uniform sigma grid |
| `--guidance-scale` | `guidance_scale` | 5.0 | classifier-free guidance against a zeroed context |
| `--sampler` | `sampler` | `euler` | solver |
| `--octree-resolution` | `octree_resolution` | 384 | finest voxel grid the surface is cut from |
| `--bounds` | `bounds` | 1.01 | half-width of the sampled volume in latent units |
| `--mc-level` | `mc_level` | 0.0 | the SDF value the surface is cut at |
| `--output-format` | `output_format` | `glb` | `glb`, `obj` or `stl`; `--out`'s extension decides if unset |

STL is the one format that stores no shared vertices: every triangle carries its own three
corners and its own normal, which is exactly what comes out of the cut — a slicer welds
them to its own tolerance, so nothing is committed here that it would have to undo.

`--octree-resolution` is the one that costs both memory and time: sampling is a narrow
band, not a dense grid, so only cells near the surface are ever queried — 384 on a
centred object asks for around 2.5 M points where a dense grid would ask for 57 M. The
band is built by halving down from the resolution you asked for until 48, so 384 walks
48 → 96 → 192 → 384 and 300 walks 75 → 150 → 300; anything at or below 48 is a single
level.

`--bounds` slightly above 1 matters: an object that touches the edge of the sampled volume
gets its surface clipped there, and nothing downstream can tell you it happened.

## What it is

The latents are tokens, not an image: the DiT denoises 4096 of them, and the codec turns
them into a signed-distance field. Three components live in the one file:

* a `HunYuanDiTPlain`-style transformer — 21 blocks, 16 heads, a time token prepended to
  the sequence and dropped again by the output layer, LIFO skips from the first ten blocks
  to the last ten, and a mixture of experts (8 experts, top 2, plus a shared one) in the
  last six blocks;
* a DINOv2-L/14 @518 image encoder, whose whole patch grid every block cross-attends to.
  A `null` context is zeros, not an empty text sequence;
* a shape codec whose 16-layer transformer refines the tokens once, and whose geometry
  decoder then answers SDF queries against them.

The SDF is sampled in a **narrow band**: an octree is refined only where the field could
cross the cut level, so the queries that would land in the middle of solid matter or in
empty space are never made. The surface is cut out of that band with tetrahedral marching
cubes, which needs no invented sign outside the sampled cells and produces a watertight,
consistently wound mesh. Nothing is welded, decimated or smoothed — what comes out is the
raw cut, at the resolution you asked for.

## Not supported

* **Text.** `prompt` is accepted so a request travels between media unchanged, and is
  otherwise ignored. `negative_prompt` too, though CFG still runs against a zeroed context.
* **Image media other than the reference.** No inpainting a shape, no combining views:
  one picture, one shape.
* **Other checkpoints in the family.** The 2.0 releases (`hunyuan3d-dit-v2*`) are a
  different transformer — Flux-style, guidance-distilled — and are not detected. Only the
  2.1 shape file is.

Nothing here has been compared against a reference implementation's output numerically;
the kernels are covered by weight-free tests, and the first real GPU run is the check on
the schedule's shape and the defaults above.
