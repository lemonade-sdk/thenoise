# Preprocess: control-map extraction

`thenoise` can extract **control maps** from a plain image — the conditioning inputs
used by ControlNet-style pipelines. It is a standalone pixel-domain tool, exactly like
`/upscale`: image in, map out, **no diffusion model loaded**. A map is always the size
of the image it was measured on.

Five kinds ship today: **canny**, **lineart** and **softedge** (all weights-free, pure
`torch`), **depth** (Depth-Anything-V2) and **pose** (RT-DETR + ViTPose).

| Input | Canny Edge | Depth map | Pose | Lineart | Soft-edged map | 
|--------|---------|---------|---------|---------|---------|
|<img width="1664" height="1248" alt="thenoise_edit_3564891533 (1)" src="https://github.com/user-attachments/assets/50ce8ae9-3adf-457c-8822-d916a00219ca" /> | <img width="1664" height="1248" alt="canny" src="https://github.com/user-attachments/assets/f210210d-b23e-481a-b373-bccd2efd1f8c" /> | <img width="1664" height="1248" alt="depth" src="https://github.com/user-attachments/assets/62485b42-6863-43b7-8173-966e93408e82" /> | <img width="1664" height="1248" alt="pose (3)" src="https://github.com/user-attachments/assets/6a8aa67e-796d-45c2-8089-0830665c2991" /> | <img width="1664" height="1248" alt="lineart" src="https://github.com/user-attachments/assets/d134594c-7a5c-4599-a132-3997fc84745b" /> | <img width="1664" height="1248" alt="softedge" src="https://github.com/user-attachments/assets/a4157140-07f9-47e9-b2a7-f9cc0372998d" />|

## Installing and downloading what each kind needs

Canny, lineart and softedge run out of the box. Depth and pose need checkpoints:

```bash
python scripts/download.py --model preprocess                 # Depth-Anything-V2 Small
python scripts/download.py --model preprocess --variant base  # 390 MB, crisper
python scripts/download.py --model preprocess --variant pose  # ViTPose-B + RT-DETR r18vd
python scripts/download.py --model preprocess --variant pose-r50vd  # the better detector
```

Everything lands in `./models/preprocessors`, one directory per checkpoint. Point
`--preprocessor-dir` (CLI) at that pool and the engine discovers each kind by reading
the subdirectories' `config.json` `model_type` — it does not trust file names.

```bash
./thenoise.sh preprocess --input photo.jpg --type depth \
  --preprocessor-dir ./models/preprocessors --out depth.png
./thenoise.sh preprocess --input photo.jpg --type pose \
  --preprocessor-dir ./models/preprocessors --out pose.png
```

A pose run is **two** checkpoints: ViTPose estimates the joints of one person crop at a
time, so RT-DETR has to say where the people are first. With an estimator but no
detector in the pool, `pose` is not offered — and `--list` says what is missing rather
than staying quiet about it.

## The depth map

Depth-Anything-V2 emits **relative** depth (larger = closer), measured at the model's
own resolution and interpolated back to the input's. The field is normalised per image
in fp32 (min–max, never in bf16) and painted as 8-bit grey: bright = near, dark = far
unless `invert`. Relative depth is the ControlNet convention; metric checkpoints exist
for real-world scale but are not part of this feature.

| Option | Default | Meaning |
|--------|---------|---------|
| `invert` | `false` | Swap near/far (dark = near) |
| `colormap` | `gray` | The map's palette: `gray` or `turbo` (a smooth, perceptually-ordered rainbow) |

`turbo` is a ~1 KB vendored lookup table (dark blue → green → dark red), sampled from
the Google `turbo` colormap's anchor colours, so a depth map reads as a vivid rainbow
instead of grey — handy when the map is for display rather than for conditioning.

## The pose map

RT-DETR finds the people, ViTPose measures the 17 COCO joints of each, and the skeleton
is *drawn* — from joint positions normalised into the image — rather than stretched, so
a limb keeps its angle whatever the map's size. It is painted as **COCO-18**: the neck
is derived at the shoulder midpoint, and the shoulder line and the head are rewired
through it, which is the skeleton the openpose ControlNets were trained on. A joint the
model was not sure about is dropped along with the limbs that hang from it, so a
half-detected person loses that limb instead of gaining a random line.

Bodies only: hands and face are not in COCO-17, and no Apache-licensed ViTPose
wholebody checkpoint exists in transformers format, so `openpose_full`-style maps are
not on offer.

| Option | Default | Meaning |
|--------|---------|---------|
| `detector` | first found | Which RT-DETR checkpoint to crop people with |
| `threshold` | `0.5` | Minimum detector score for a person to be in the map at all |
| `kp_threshold` | `0.3` | Minimum confidence for a joint to be drawn |
| `line_width` | `4` | Bone thickness, in pixels |
| `radius` | `4` | Radius of the joint dots |
| `background` | `black` | `black` or `green` — the canvas SD's original openpose preprocessor used |

The keypoint model runs in BF16 like everything else; the detector runs in fp32
(deformable attention and the box topk are less happy reduced), and the crop warps and
joint refinement run in fp32 on CPU, which is where a pose run spends a visible slice of
its time.

## The canny map

Canny needs no weights: luminance (Rec. 709), a separable 5×5 Gaussian (σ=1.2), Sobel
gradients, non-maximum suppression, and hysteresis grown from strong seeds — all in
`torch`, no `cv2`. White edges on black; `low`/`high` are 8-bit Sobel-magnitude
thresholds, `thickness` dilates the line.

## The lineart map

Line-art is the same Sobel + non-maximum-suppression pipeline as Canny, but with a
**single threshold** (no hysteresis) and the conventional inverted palette: thin, clean
dark lines on a white canvas. It is a weights-free approximation of the model-based
ControlNet `lineart` annotator — good for clean, unambiguous line drawings.

| Option | Default | Meaning |
|--------|---------|---------|
| `threshold` | `0.5` | Line intensity threshold, 0-1, on the per-image normalised edge strength |
| `thickness` | `1` | Line thickness, in pixels |

## The softedge map

Soft-edge is the **continuous** edge field rather than a binarised one: the Sobel
magnitude, passed through a smooth threshold so a soft edge keeps its grey gradient,
then optionally blurred. Dark, anti-aliased edges on a white canvas — a weights-free
approximation of the HED-style `softedge` annotator.

| Option | Default | Meaning |
|--------|---------|---------|
| `threshold` | `0.2` | Soft edge cutoff, 0-1: where the gradient starts to count |
| `blur` | `1.5` | Gaussian smoothing sigma (pixels); 0 = no blur, so an edge stays crisp |


## The `preprocess_data` chunk

Every map is saved with a PNG `preprocess_data` text chunk recording `type`, `model`
and the options that differed from their defaults, alongside whatever metadata the
input image already carried. The map is therefore self-describing on disk and in the
UI's info modal.

## The kinds on offer

| Kind | Needs weights | Checkpoint (default) | Size |
|------|---------------|----------------------|------|
| `canny` | no | — | 0 |
| `lineart` | no | — | 0 |
| `softedge` | no | — | 0 |
| `depth` | yes | `depth-anything/Depth-Anything-V2-Small-hf` | 99 MB |
| `pose` | yes | `usyd-community/vitpose-base-simple` + `PekingU/rtdetr_r18vd_coco_o365` | 425 MB |

Larger variants of each are one flag away: `--variant base` / `--variant large` for
depth, `--variant pose-r50vd` for the detector that handles crowds and occlusion better.
Missing a person is a hard failure (they are simply absent from the map) while a false
positive is a soft one (one stray skeleton), so the bigger detector is usually the
better place to spend 90 MB.
