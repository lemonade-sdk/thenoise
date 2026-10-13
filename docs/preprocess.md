# Preprocess: control-map extraction

`thenoise` can extract **control maps** from a plain image — the conditioning inputs
used by ControlNet-style pipelines. It is a standalone pixel-domain tool, exactly like
`/upscale`: image in, map out, **no diffusion model loaded**. A map is always the size
of the image it was measured on.

Two kinds ship today: **canny** (weights-free, pure `torch`) and **depth**
(Depth-Anything-V2). **Pose** is the next kind planned.

## Downloading checkpoints

Only depth needs weights; canny runs out of the box.

```bash
python scripts/download.py --model preprocess                # Depth-Anything-V2 Small
python scripts/download.py --model preprocess --variant base # 390 MB, crisper
python scripts/download.py --model preprocess --variant large
```

Each variant lands in `./models/preprocessors/depth_anything_v2_<variant>`, a
directory with `config.json`, `model.safetensors` and `preprocessor_config.json`.
Point `--preprocessor-dir` (CLI) at that pool and the engine discovers the depth kind
by reading each subdirectory's `config.json` `model_type` — it does not trust file
names.

```bash
./thenoise.sh preprocess --input photo.jpg --type depth \
  --preprocessor-dir ./models/preprocessors --out depth.png
```

## The depth map

Depth-Anything-V2 emits **relative** depth (larger = closer), measured at the model's
own resolution and interpolated back to the input's. The field is normalised per image
in fp32 (min–max, never in bf16) and painted as 8-bit grey: bright = near, dark = far
unless `invert`. Relative depth is the ControlNet convention; metric checkpoints exist
for real-world scale but are not part of this feature.

| Option | Default | Meaning |
|--------|---------|---------|
| `invert` | `false` | Swap near/far (dark = near) |
| `colormap` | `gray` | The map's palette (only `gray` today; `turbo` is a later addition) |

## The canny map

Canny needs no weights: luminance (Rec. 709), a separable 5×5 Gaussian (σ=1.2), Sobel
gradients, non-maximum suppression, and hysteresis grown from strong seeds — all in
`torch`, no `cv2`. White edges on black; `low`/`high` are 8-bit Sobel-magnitude
thresholds, `thickness` dilates the line.

## The `preprocess_data` chunk

Every map is saved with a PNG `preprocess_data` text chunk recording `type`, `model`
and the options that differed from their defaults, alongside whatever metadata the
input image already carried. The map is therefore self-describing on disk and in the
UI's info modal.

## The two kinds on offer

| Kind | Needs weights | Checkpoint (default) | Size |
|------|---------------|----------------------|------|
| `canny` | no | — | 0 |
| `depth` | yes | `depth-anything/Depth-Anything-V2-Small-hf` | 99 MB |
