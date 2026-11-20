# Hunyuan3D 2.1 (shape)

One checkpoint answers the **mesh** medium: a picture goes in, a triangle mesh comes
out. It is image-conditioned only — there is no text encoder in the file and no prompt
takes a shape anywhere it would otherwise not go.

| | Hunyuan3D 2.1 shape |
|---|---|
| Modes | image-to-shape (`i2d`); views-to-shape (`mv2d`) when several pictures are sent; text-to-shape does not exist |
| Checkpoint | **one file** (`--dit`): transformer + DINOv2 image conditioner + shape codec |
| Conditioning | the reference reframed to a 518² square, as its whole DINOv2-L/14 patch grid; one grid per picture, concatenated into one context |
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
a blob, since nothing else conditions this model. In the web UI the Generate tab becomes
the shape form — a dropzone and the knobs below, and no prompt field, because there is
nothing to type.

Before the tower sees the picture it is framed the way the tower's own preprocessing
frames it: the short side is scaled to 518 and the overhang is cropped off the long sides,
so nothing is stretched and the canvas is always 37×37 patches — the 1370 conditioning
tokens the checkpoint's config declares. Transparency is composited onto white, the
surface the cutouts it was trained on sat on.

Several pictures are views of **one** object, not several objects: each 1370-token grid is
marked with the sincos offset of its slot — front, left, back, right, in the order sent —
and the marked grids are concatenated into a single conditioning, which is what both the
release's `DinoImageEncoderMV` and ComfyUI's multi-view node do. Four is the ceiling
because four is how many slots are numbered. A single picture never takes an offset, so
the verified single-view path is untouched; and note what that leaves unproven — the
offsets mean something to a model trained with them, and the 2.1 shape file is configured
as a single-view model. Views on it run, and what they produce is yours to judge.

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
| `--mc-level` | `mc_level` | 1/512 | the field value the surface is cut at; the reference pipeline's cut, in this field's sign |
| `--output-format` | `output_format` | `glb` | `glb`, `obj` or `stl`; `--out`'s extension decides if unset |

STL is the one format that stores no shared vertices: every triangle carries its own three
corners and its own normal, which is exactly what comes out of the cut — a slicer welds
them to its own tolerance, so nothing is committed here that it would have to undo.

`--octree-resolution` is the one that costs both memory and time: sampling is a narrow
band, not a dense grid, so only cells near the surface are ever queried — a centred object
at 384 asks for a few million points where a dense grid would ask for 57 M. The band is
built by halving down from the resolution you asked for until 63, so 384 walks
96 → 192 → 384 and 300 walks 75 → 150 → 300; anything at or below 63 is a single level.

`--bounds` slightly above 1 matters: an object that touches the edge of the sampled volume
gets its surface clipped there, and nothing downstream can tell you it happened.

## What it is

The latents are tokens, not an image: the DiT denoises 4096 of them, and the codec turns
them into the field the surface is cut from. Three components live in the one file:

* a `HunYuanDiTPlain`-style transformer — 21 blocks, 16 heads, a time token prepended to
  the sequence and dropped again by the output layer, LIFO skips from the first ten blocks
  to the last ten, and a mixture of experts (8 experts, top 2, plus a shared one) in the
  last six blocks;
* a DINOv2-L/14 @518 image encoder, whose whole patch grid every block cross-attends to.
  A `null` context is zeros, not an empty text sequence;
* a shape codec whose 16-layer transformer refines the tokens once, and whose geometry
  decoder then answers field queries against them.

The DiT counts the other way round: its clock runs up from noise
(`x_t = t · latents + (1 - t) · noise`) and it predicts `latents - noise`, so the adapter
hands it `1 - sigma` and negates what comes back, which is the flow `noise - latents` the
solver integrates.

Its attention also reads heads the way the release does, which is not how a textbook split
does: self-attention concatenates the three projections and takes `q`, `k`, `v` three
blocks to a head — head 0's key is `to_q`'s second block of features — while cross-attention
splits its query plainly and interleaves only the fused key-value pair. Both give tensors of
exactly the usual shapes; the trained weights answer to these readings and nothing else.

The field is sampled in a **narrow band**: the octree starts at the coarsest level of the
reference's 63-halving ladder (96 for a 384 target) and refines the cells holding a sign
change, the one-cell shell around them, and any cell whose corners come within 0.95 of the
cut. That last term is the reference decoder's own, and it only earns its keep because the
codec answers in logits that run out at about ±1: a corner that close to the level is worth
subdividing even when no corner crossed it, whereas on a plain distance field the same rule
would keep the whole volume. Tetrahedral marching cubes cut the surface out of the band,
needing no invented sign outside the sampled cells and giving a watertight, consistently
wound mesh. The codec's logit is positive inside, so the adapter negates it before the cut:
triangles face outward, and an inside-out glb is not something this engine hands back.
Nothing is welded, decimated or smoothed — the raw cut, at the resolution you asked for.

## Not supported

* **Text.** `prompt` is accepted so a request travels between media unchanged, and is
  otherwise ignored. `negative_prompt` too, though CFG still runs against a zeroed context.
* **Image media other than the reference.** No inpainting a shape and no half-built
  views: a slot is either a picture or absent, and every picture sent is one view of the
  one shape being built.
* **Other checkpoints in the family.** The 2.0 releases (`hunyuan3d-dit-v2*`) are a
  different transformer — Flux-style, guidance-distilled — and are not detected. Only the
  2.1 shape file is.

Every stage of the port has been diffed against Tencent's own implementation with the real
checkpoint on CPU: same weights, same inputs, per-block outputs equal to within bf16
rounding. The weight-free tests hold that shape in place; none of them could have caught
the conventions that were wrong at first — the Fourier `include_pi`, the clock and flow
sign, the head interleave — so each is pinned by a test that fails in the other reading.
