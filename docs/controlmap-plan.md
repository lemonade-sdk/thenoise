# Plan: control-map extraction (pose / depth / canny)

Status: **proposal — needs decisions from the maintainer** (see [Open questions](#open-questions)).

Add a tool that takes an image and returns a **control map**: an OpenPose-style pose
skeleton, a Depth-Anything depth map, or a Canny edge map. Exposed as a CLI
subcommand, an HTTP endpoint and a web-UI tab, in all three cases **without a
diffusion model loaded** — exactly like today's pixel upscaler.

---

## 1. What this is, and what it is not

`thenoise` has no ControlNet/control-adapter support: nothing in the engine consumes a
control map today. So this feature is a **standalone pixel-domain tool** — an
"auxiliary/controlmap" utility of the same kind as `/upscale`: image in, image out,
no DiT, no text encoder.

```
Phase 1 (this plan)   extract:  image → pose map | depth map | canny map   (standalone)
Phase 2 (out of scope) consume:  control map → generation (control adapters)
```

Phase 1 is useful on its own (produce maps for any external tool; feed a map to the
existing `edit` endpoint as a reference image for rough control), and it is the
prerequisite for Phase 2. If Phase 2 is the real goal, say so now: it changes a few
conventions (output sizing, background colour, whether we return a latent-aligned
size) which are otherwise free decisions — see [Open questions](#open-questions).

Naming used below: **controlmap** (the tool), **control map** (its output).
Decided: the feature is called **`controlmap`** (subcommand, endpoint, tab).

### 1.1 What "future ControlNet conditioning" would actually change

If Phase 2 (control conditioning inside `thenoise`) becomes the goal, these are the
concrete consequences — most of them are cheap to prepare for now, one or two are not:

| # | Consequence | Cheap now? |
|---|---|---|
| a | **Shared vocabulary.** The control-map *kind* becomes an enum used by both endpoints: `/text2image` grows `control: {type, from, strength, start/end}`. Put the kind enum in `controlmap/kinds.py` and reference it from `models/config.py:GenerateRequest` so the two can never drift. | ✅ do now |
| b | **Detect and render must be separate steps.** A raster can be resized; a *skeleton* must be re-drawn (stretching a pose map changes limb angles). So the pipeline must be `detect(image) → ControlSignature` (keypoints in **normalized** coords, depth as an fp32 field, edges as a mask) + `render(signature)`. | ✅ do now (`render` boundary) |
| c | **Raw outputs, not just PNG.** The control branch may need its own normalisation/inversion, so the manager must be able to hand back the raw fp32 depth field and unthresholded edge/heatmap tensors, not only a uint8 map. | ✅ do now (internal API only) |
| d | **Output conventions stop being free.** Background colour (SD1.5 openpose ControlNets were trained on the green `(36,255,12)` canvas, FLUX/Qwen union ones on black), depth polarity, canny thresholds and line thickness all become **per-target-model defaults** — effectively a `CONTROL_PROFILES` dict on the adapter, next to `CAPABILITIES`. | ⚠️ Phase 2 |
| e | **Memory ordering.** A ControlNet encoder is another 1–6 GB resident next to the DiT; the controlmap must load → run → free *inside* the generation boundary rather than lazily on demand, and interact with `--offload-device` and the pipeline cache. | ⚠️ Phase 2 |
| f | **One-shot request shape.** "Extract + generate" in a single call (no double base64 round-trip) ⇒ the typed option dataclasses must be embeddable in a generation request. This is exactly why the request model uses nested typed option groups instead of flat fields. | ✅ do now |
| g | **Quality bar.** Standalone, a slightly different Canny is fine; as conditioning, thresholds and skeleton thickness materially change the output, and body-only maps leave hands unconstrained. | ⚠️ Phase 2 |
| h | **The adapters themselves.** `thenoise` has no control-branch implementation at all; Phase 2 means porting/wiring a control encoder per family (and obtaining control checkpoints). That is a large, separate project — the maps are genuinely useful without it (external tools, and feeding a map to `/edit` as a reference image). | ⚠️ Phase 2 |

**Decision (D1): Phase 2 is not committed**, so only (a), (b), (c), (f) are prepared —
and **none of them is dead code**, because each one is the ordinary way to write the
feature, not a scaffold for it:

* (a) ⇒ the kind enum lives in `controlmap/kinds.py` and *is* what `type` validation and
  `/controlmaps` key off. It is not a field added to the generation request.
* (b) ⇒ `detect(image) → signature` + `render(signature)` is the live execution path:
  the map is the size of the image it was measured on (§4.4), but a skeleton still has
  to be drawn from its normalized keypoints rather than stretched as pixels, which is
  exactly what the two steps keep apart. It is also what makes the renderer
  unit-testable with hand-built heatmaps, no weights.
* (c) ⇒ free consequence of (b): depth is carried as an fp32 field because min–max
  normalisation must not run in bf16. The public API still only ever returns a PNG —
  there is no `raw` response mode nothing consumes.
* (f) ⇒ `CannyOptions` / `DepthOptions` / `PoseOptions` are named pydantic models in one
  module because the UI sends them; they are *not* added to `Text2ImageRequest`.

Hard rule for the implementation: **no unused parameters, no unused return paths, no
"for future controlnet" branches.** If Phase 2 never happens the diff reads simply as
"detection and painting are separate steps, map kinds are an enum".

### 1.2 What Phase 2 actually buys you (and what it costs)

Today a map is an **end product**: a PNG you hand to an external tool, or drop into the
existing `/edit` endpoint as a reference image for loose, prompt-driven control. Phase 2
is the step where `thenoise` itself treats the map as **conditioning during generation**
— the ControlNet-style "do what the map says, in this style" workflow.

Concretely, with Phase 2 a request could carry a `control` block and the engine would
hold the map's layout while following the prompt:

```
generate --prompt "a woman in a red coat" \
  --control pose=person.png,strength=0.8 \
  --control depth=scene.png,strength=0.5
```

The map becomes a constraint rather than a suggestion:

| Map | What it constrains |
|-----|-------------------|
| **pose** | where the people are and their posture — "same pose, different face / outfit / background" is the point |
| **depth** | the 3D structure and scene layout — objects stay put, foreground/background separation survives a style change |
| **canny / lineart / softedge** | the boundaries — outlines and edges are respected while colours and shading are regenerated |
| **normal** | surface orientation and lighting direction, if it is ever added |

That is how you get "same composition, new style", "same person, new outfit",
"background stays, subject moves" and the rest of the conditioning family of tricks.

**What it costs, and why it is a project rather than a flag:**

* **A control encoder per model family.** There is no universal control adapter: the
  FLUX / Qwen-union adapters and the SD1.5 ControlNets are different weights with
different conventions, and `thenoise` has no control-branch at all today. Each family
needs its adapter ported and its control checkpoints downloaded.
* **Memory.** A control encoder is another ~1–6 GB resident next to the DiT. On the Strix
  Halo's unified memory that is affordable but not free, and it must interact with
  `--offload-device` and the pipeline cache so it loads → runs → frees inside the
generation boundary rather than lazily on demand.
* **Per-target-model defaults.** Background colour (green vs black), depth polarity,
canny thresholds and skeleton thickness are all things each target model was trained
with. A `CONTROL_PROFILES` dict on the adapter — next to `CAPABILITIES` — decides them,
so "just use defaults" works and a user can override one value without reading a profile.
* **The quality bar goes up.** Standalone, a slightly different canny is fine; as
  conditioning, thresholds and skeleton thickness materially change the output, and
body-only maps leave hands unconstrained (which is why wholebody is worth a separate
decision).

The good news is the maps are already shaped to be consumed this way. The `detect →
signature → render` split, the kind enum, the fp32 depth field and the nested option
groups in the request model were all chosen so Phase 2 needs no redesign of what you see
today — it only needs the control-branch itself.

---

## 2. Library analysis

### 2.1 The two hard constraints that decide everything

1. **The venv is CPython 3.14 with a hand-installed ROCm `torch`.**
   Verified against PyPI: `opencv-python-headless` publishes **no cp314 wheels at all**
   (any version), and `mediapipe` has none either. `scipy` (≥1.18) and `scikit-image`
   (≥0.26) *do* have cp314 wheels, `onnxruntime` (≥1.31) too.
2. **Nothing may resolve `torch`/`torchvision` from PyPI** (AGENTS.md §1). Any
   dependency whose metadata requires `torch` is a landmine: `pip`/`uv` is free to
   "fix" the ROCm build with a CUDA wheel.

### 2.2 Candidate stacks

| Stack | Verdict | Why |
|---|---|---|
| **OpenPose (CMU)** | ❌ reject | Caffe/C++ reference build, no maintained Python path; `.caffemodel` weights; its own license is **non-commercial research only**; skeleton is CPU-heavy Python post-processing (NMS over peak maps). Its only practical Python entry point is `controlnet_aux`. |
| **`controlnet_aux`** (lllyasviel) | ❌ reject | Its metadata requires **`torch`, `torchvision`, `opencv-python-headless`, `timm`, `scipy`, `scikit-image`** — constraint (2) directly violated (it can replace the ROCm torch), and constraint (1) violated (no cp314 OpenCV wheel ⇒ it cannot even be installed in this venv). Also assumes `cv2` everywhere. |
| **ONNX route** (`onnxruntime` + DWPose/OpenPose `.onnx`) | ⚠️ possible | `onnxruntime` does ship cp314 wheels and is torch-agnostic. But: no ROCm execution provider in practice ⇒ CPU-only inference (≈0.5–2 s/person); DWPose weights carry **CC-BY-NC** provenance (the HF repo claims Apache-2.0, the paper repo says non-commercial — ambiguous); ONNX weights are a second, non-safetensors artifact pipeline; a native-extension dep for one feature. |
| **MediaPipe Pose/Holistic** | ❌ reject | Nice API and body+hand+face in one, but **no cp314 wheels**, ~100 MB of native code, and its own model-download channel (not HF). |
| **MMPose / RT-DETR+RTMPose ONNX** | ❌ reject (v1) | MMPose drags the whole OpenMMLab stack (mmcv compile, opencv). RTMPose-as-ONNX is the ONNX route again. |
| **✅ `transformers` (already a dependency) — Depth-Anything-V2 + ViTPose + RT-DETR** | ✅ **recommended** | Zero new required dependencies, all pure `torch` (runs on ROCm/BF16 like the rest of the engine), all checkpoints are **safetensors on the Hub** (same `huggingface_hub` path as `scripts/download.py`), all **Apache-2.0**. |

### 2.3 The recommended stack, checkpoint by checkpoint

All three model types already exist in `transformers`; the project pins
`transformers>=4.57.6`, and all three have been in the library far longer
(RT-DETR 4.29, DepthAnything 4.35, ViTPose 4.40).

| Task | Class | Checkpoint (default) | Size | License | Notes |
|---|---|---|---|---|---|
| depth | `AutoModelForDepthEstimation` (`DepthAnythingForDepthEstimation`) | `depth-anything/Depth-Anything-V2-Small-hf` | 99 MB | Apache-2.0 | DINOv2-S + DPT head, 518² input, *relative* depth (larger = closer, which is the ControlNet depth convention). `-Base-hf` (390 MB) / `-Large-hf` (1.34 GB) as options; `-Metric-{Indoor,Outdoor}-*-hf` for metric depth later. |
| pose: person boxes | `AutoModelForObjectDetection` (`RTDetr*`) | `PekingU/rtdetr_r18vd_coco_o365` | 81 MB | Apache-2.0 | Alternatives: `rtdetr_r50vd` (172 MB, more accurate), `Comfy-Org/RT-DETR` repackaging. Keep the person class only. |
| pose: keypoints | `VitPoseForPoseEstimation` (4.x: `VitPoseForKeypointDetection`) | `usyd-community/vitpose-base-simple` | 344 MB | Apache-2.0 | COCO-17, 256×192 crops, top-down (needs boxes). `usyd-community/vitpose-plus-base` (502 MB) is the multi-dataset variant if we ever want more than body. |
| canny | — | none | 0 | — | ~80 lines of `torch` (§4.1). |

Why a detector at all: ViTPose is **top-down**, it estimates keypoints for one
person-crop at a time. RT-DETR is the Apache-licensed person detector that HF's own
ViTPose examples use. (OpenPose is bottom-up and needs no detector — that is its one
advantage, and it is not available to us.)

**Verified in the installed transformers (v5.18):**

* `RTDetrImageProcessorPil.post_process_object_detection` is **pure torch** (no
  `torchvision`, no NMS op) ⇒ RT-DETR needs nothing beyond `torch` + `PIL`.
* `VitPoseImageProcessorPil` (PIL backend) needs `numpy` + **`scipy`** (affine warp +
  the DARK gaussian), and bounces every crop through CPU numpy.
* v5 selects processors with `backend="pil"` (v4: `use_fast=False`); v5 renamed
  `VitPoseForKeypointDetection` → `VitPoseForPoseEstimation` and
  `post_process_keypoints` → `post_process_pose_estimation`, with a different box
  convention (see §4.2 ⚠️).

That drift is the one real cost of riding `transformers`, and it is what decision
**D2** in [Open questions](#open-questions) is about.

### 2.4 New dependencies: the whole list

| Extra | Packages | Needed for |
|---|---|---|
| `[controlmap]` | *optional* `scipy` only (D2: only if we use HF's ViTPose pre/post) | pose crop/refinement |

scipy's real cost, measured: **35 MB wheel download, 112 MB installed** (81 MB of
shared objects), and its only dependency is `numpy>=2.0,<2.8` — the venv has numpy
2.5.3, so nothing else moves. It is pure math: unlike `controlnet_aux` it **cannot
pull a PyPI torch**, so it does not threaten the ROCm build. Trivial next to the
multi-GB torch install, but it is a second native stack for one feature — which is why
D2 exists.

Nothing for canny, depth or — with D2 = "vendored" — pose. No `torch`, no
`torchvision`, no `opencv`, no `timm`, no `onnxruntime`. Canny/depth work with the
current install; pose with D2 = "vendored" also works with the current install.

---

## 3. Architecture

Mirrors the pixel-upscaler subsystem one-for-one, which keeps the review surface
small and makes the feature obviously "the same kind of thing".

```
thenoise/controlmap/            NEW subpackage (pixel-domain, model-free)
  base.py          ControlMap ABC: name, kind, DTYPE, load(device), run(image, opts)
  registry.py      discovery by config.json model_type + CONTROLMAPS table + errors
  manager.py       ControlMapManager: dir, lazy load, 1-resident LRU, device/dtype
  canny.py         torch Canny (weights-free)
  depth.py         Depth-Anything-V2 runner
  pose.py          RT-DETR + ViTPose runner (two stages)
  render.py        keypoint → OpenPose-style skeleton (PIL ImageDraw), palettes, colormaps
  options.py       typed option dataclasses (CannyOptions / DepthOptions / PoseOptions)
  _hf.py           the ONLY place transformers names/conventions are referenced (D2 shim)
thenoise/controlmap_controller.py  NEW: lock + inference() + validate + PNG info
                                   (the twin of upscale_controller.py)
thenoise/cli.py         + controlmap subcommand, + --controlmap-dir on serve
thenoise/__main__.py    + dispatch to run_controlmap
thenoise/controlmap_cli.py  NEW: one-shot CLI (twin of upscale_cli.py)
thenoise/api.py         + GET /controlmaps, + POST /controlmap
thenoise/runtime.py     + Settings.controlmap_dir, + Runtime.controlmaps/.controlmap
thenoise/utils/png.py   + build_controlmap_pnginfo() → "controlmap_data" iTXt
thenoise/ui/            + "ControlMap" tab (index.html, js/controlmap.js, style.css)
scripts/download.py     + --model controlmap
docs/                   + controlmap.md, sections in api.md / cli.md / README / development.md
tests/test_controlmap_*.py
```

Wiring follows existing conventions:

* **Inference policy** (`thenoise/inference.py`): the docstring names the boundaries
  where `inference()` is opened. `ControlMapController.run()` becomes a **fourth**
  boundary (it can be called with no model loaded, like `PixelUpscaleController`),
  taking the same process-global `inference_lock` so control-map extraction never races a
  generation. Update that docstring.
* **Pixels**: reuse `utils/image_tensor.py` (`load_image`, `pil_to_pixels`,
  `pixels_to_pil`, `flatten_alpha`, `resize_to_target`, `align_down`). Control models
  are RGB: flatten alpha to white (`ALPHA_BACKGROUND`) rather than silently dropping it.
* **Residency**: `ControlMapManager` keeps **one** resident controlmap
  (last-used, like `PixelUpscalerManager.switch`), `load()` on first use, `del` +
  `gc.collect()` + `clean_memory_on_device()` on swap — the same dance as
  `Runtime._unload`. Depth+pose weights are 100–600 MB, so on a small dGPU this
  matters: control-map extraction between generations must not eat the DiT's headroom.
* **dtype**: engine-wide BF16, but per-controlmap opt-out as a class attribute
  `DTYPE`: depth `bfloat16`, pose estimator `bfloat16`, **detector `float32`**
  (deformable attention + topk are less happy in reduced precision) — to be confirmed
  by the maintainer's runs (§8).
* **Device**: `--device` as everywhere (`cuda` → hip under ROCm); processors run on
  GPU, and only the pose post-processing touches CPU.

### 3.1 Interfaces

```python
# thenoise/controlmap/base.py
class ControlMap(ABC):
    name: ClassVar[str]                  # "depth-anything-v2-small", "vitpose-base", "canny"
    kind: ClassVar[str]                  # "depth" | "pose" | "canny"
    DTYPE: ClassVar[torch.dtype] = torch.bfloat16
    WEIGHTS: ClassVar[bool] = True       # canny: False

    def load(self, device: str | torch.device, path: str | None) -> None: ...
    def unload(self) -> None: ...
    def run(self, image: Image.Image, options) -> Image.Image:
        """RGB/RGBA PIL in → single-channel (depth/canny) or RGB (pose) PIL out,
        at the size of ``image``."""
```

`kind` is what a request asks for; `name` is which checkpoint implements it — the same
"capability vs implementation" split the engine already uses with
`CAPABILITIES`/`pixel_upscaler`.

```python
# thenoise/controlmap/manager.py
class ControlMapManager:
    def __init__(self, controlmap_dir: str, device: str): ...
    def list(self) -> dict[str, list[str]]        # {"depth": [...], "pose": [...], "canny": ["canny"]}
    def resolve(self, kind: str, name: str = "") -> ControlMap   # "" → default; ValueError with a hint
    def run(self, image, kind, name, options) -> Image.Image       # lock held by caller
```

Discovery is by **reading `config.json`** in each subdirectory of
`--controlmap-dir` and mapping `model_type` → kind
(`depth_anything`→depth, `vitpose`→pose estimator, `rt_detr`/`deformable_detr`→person
detector). This is the same "inspect the checkpoint, don't trust the filename"
philosophy as `thenoise/models/__init__.py:resolve()`. A pose run therefore needs an
estimator *and* a detector present, and reports a precise error when one is missing.

```
models/controlmaps/
├── depth_anything_v2_small/   config.json  model.safetensors  preprocessor_config.json
├── vitpose_base_simple/       …
└── rtdetr_r18vd/              …
```

---

## 4. The three controlmaps

### 4.1 Canny (no weights, pure `torch`)

No `cv2` exists for this Python, so implement it. Pipeline:

1. RGB (alpha composited onto white by `pil_to_pixels`) → Rec. 709 luminance,
2. 5×5 gaussian blur, σ=1.2 — two separable `conv2d` passes, replicate padding, each
   pass padding only the axis it walks,
3. Sobel 3×3 (`conv2d`, separable) → `gx, gy`; magnitude `hypot` in 8-bit grey levels
   (so the conventional 100/200 defaults land on real edges),
4. angle quantised to 4 orientations, then non-maximum suppression: keep the local
   maxima along the gradient, which is what makes a step edge one pixel wide,
5. double threshold (`low`, `high`) into strong/weak,
6. **hysteresis without `cv2`**: iterate `strong ← max_pool2d(strong ∪ weak, 3)` to a
   fixed point (early exit when unchanged, cap 2048 passes as a bound on pathological
   chains). 8-connected, exact semantics, ~10 lines, and trivially unit-testable.
7. `thickness`: 1 = the NMS result, n>1 = n−1 3×3 max-pool passes (dilation).
8. output: black background, white edges, the size of the image it measured (§4.4).

For control maps the criterion is "same edges, similar density", which the unit tests
assert structurally (§7). Cost at 1024²: tens of milliseconds on GPU.

### 4.2 Depth — Depth-Anything-V2

```python
proc = AutoImageProcessor.from_pretrained(path, backend="pil")          # v5 (use_fast=False on v4)
model = AutoModelForDepthEstimation.from_pretrained(path, dtype=DTYPE).to(device).eval()
out = model(**proc(images=image, return_tensors="pt").to(device, DTYPE)).predicted_depth   # [B,1,h,w]
depth = F.interpolate(out, size=image.size[::-1], mode="bilinear", align_corners=False)[0, 0]
```

Then, in `render.py`: min–max normalise (fp32 — do **not** normalise in bf16), map to
uint8, optional `invert`, optional `colormap` (`gray` default, `turbo` as a ~1 KB
vendored LUT). Relative depth, so `normalize: per-image` is the only sane default;
document that metric checkpoints exist for real-world scale. Input is 518²,
aspect-preserved to a multiple of 14 (`keep_aspect_ratio`, `ensure_multiple_of: 14` in
the checkpoint's own `preprocessor_config.json`) — no padding, no tiling needed. The
field is interpolated back to the input size before it is painted. At 1024² this is
sub-second on a Strix Halo in BF16 (§8).

Options: `invert: bool`, `colormap: "gray"|"turbo"`.

### 4.3 Pose — RT-DETR (person) → ViTPose (COCO-17) → skeleton render

```python
det   = RTDetrForObjectDetection.from_pretrained(detector_dir, dtype=torch.float32)
pose  = <VitPose class resolved via _hf.py>.from_pretrained(estimator_dir, dtype=torch.bfloat16)

boxes = detect(image, threshold=0.5)                     # keep COCO class "person" only
crops = [affine_crop(image, b, (192, 256)) for b in boxes]     # batched, on device
heat  = pose(**pixel_values).heatmaps                    # [P,17,64,48]
kps   = heatmaps_to_keypoints(heat, centers, scales)     # argmax + DARK refine + back-project
canvas = draw_skeleton(image.size, kps, palette=COCO_POSE_PALETTE, ...)
```

Two stages, batched: one detector forward, one estimator forward of `P` crops.

⚠️ **The box convention is the single riskiest detail.** ViTPose's crop wants
`[x, y, w, h]` **absolute COCO** boxes and the mmpose `box_to_center_and_scale`
(padding ×1.25, aspect-matched); v5's `post_process_pose_estimation` multiplies the
boxes it is handed by the target size, i.e. it expects something else again, and v4's
`post_process_keypoints` differs from both. This must be pinned down with a
golden/structural test rather than guessed, and the answer confined to
`controlmap/_hf.py`.

⚠️ **v4-vs-v5 API drift** (class name, post-processing method name, `dtype=` vs
`torch_dtype=`, `backend=` vs `use_fast=`): resolved by decision **D2**.

**Rendering** (`render.py`, `PIL.ImageDraw`, no `cv2`): black background (option
`background: "black"|"green"` — SD's original openpose controlmap used the green
`(36,255,12)` canvas, ComfyUI's DWPose uses black); the model's own `config.json`
`edges` list as the skeleton; per-limb COCO palette; keypoints as filled circles
(`radius`, `line_width` options); low-confidence keypoints dropped
(`kp_threshold`, default 0.3). COCO-17 → **COCO-18** by deriving the neck as the
shoulder midpoint and rewiring the torso edges, which is what every
ControlNet-openpose canvas uses.

Options: `detector` (name), `threshold` (person score), `kp_threshold`, `line_width`,
`radius`, `background`.

**Hands and face are not in v1** (COCO-17 body only) — and with the transformers stack
they are **not available at all**: verified against the checkpoints, both
`usyd-community/vitpose-base-simple` and `usyd-community/vitpose-plus-base` ship
exactly 17 COCO labels, and the only ViTPose *wholebody* (133-keypoint) artifacts on
the Hub are community conversions (1.23 GB ONNX, or CC-BY-NC mmpose checkpoints), i.e.
not transformers-format and not Apache-2.0. Getting "openpose_full" (17 body + 68 face
+ 2×21 hands) therefore means the ONNX/DWPose route with its CPU cost and licence
ambiguity, or hand-built configs plus key remapping for a community wholebody
checkpoint. Worth a separate decision later.

### 4.4 Output geometry

**Always the input resolution.** A precomputed map is paired with the image it was
measured on, and whoever conditions on it does its own fitting — so there is nothing
for the extractor to guess, and no `--width/--height/--fit` to get wrong. Every stage of
every controlmap therefore keeps the geometry (`detect` measures the input raster,
`render` paints a canvas of that same size), which is also what keeps a map
pixel-aligned with its image for Phase 2.

Keeping `detect`/`render` apart still matters, just not for sizing: a skeleton is drawn
from normalized keypoints at whatever size the canvas is, and a depth field picks its own
normalisation and palette — neither is a resize of a raster.

---

## 5. HTTP API

```
GET  /controlmaps   →  {"controlmaps": {"depth": ["depth-anything-v2-small"],
                                            "pose":   ["vitpose-base-simple"],
                                            "canny":  ["canny"]},
                          "detectors": ["rtdetr-r18vd"],
                          "defaults":  {"depth": "depth-anything-v2-small", ...},
                          "pose_ready": true}
POST /controlmap      →  image/png  (or {"b64_json": …} with "out":"json")
```

Both **model-free**, like `/upscalers` / `/upscale`: they answer with an empty list /
HTTP 400-with-hint when `--controlmap-dir` is unset or the weights are missing, and
the message names the fix (`scripts/download.py --model controlmap`).

`ControlMapRequest` (pydantic, explicit typed fields — no free-form dict):

| Field | Type | Default | Notes |
|---|---|---|---|
| `image_b64` | `str` | required | PNG/JPEG bytes, base64 (same convention as `/upscale`) |
| `type` | `"depth"\|"pose"\|"canny"\|"lineart"\|"softedge"` | required | the *kind* |
| `model` | `str\|null` | `null` = default for the kind | checkpoint name from the dir |
| `depth` | `DepthOptions\|null` | | `invert`, `colormap` |
| `pose` | `PoseOptions\|null` | | `detector`, `threshold`, `kp_threshold`, `line_width`, `radius`, `background` |
| `canny` | `CannyOptions\|null` | | `low`, `high`, `thickness` |
| `lineart` | `LineartOptions\|null` | | `threshold`, `thickness` |
| `softedge` | `SoftedgeOptions\|null` | | `threshold`, `blur` |
| `out` | `"png"\|"json"` | `"png"` | |

There is no size in the request: the answer is the size of `image_b64` (§4.4).

Errors: 400 unknown type / bad option / missing weights (message carries the remedy),
400 undecodable image, 500 with logging otherwise — the identical mapping the `/upscale`
route uses, shared with it (and with `/text2image`, `/edit`) in one helper rather than
repeated per route. Response carries a `controlmap_data` PNG chunk (`utils/png.py`, next
to `build_upscale_pnginfo` and built by the same `_provenance` helper) with kind, model
and the options that differed from their defaults, so the map is self-describing on disk
and in the UI info modal.

---

## 6. CLI and UI

### 6.1 CLI

```bash
# one-shot, no diffusion model, no server
.venv/bin/python -m thenoise controlmap \
    --input photo.jpg --type depth --out control_depth.png \
    --controlmap-dir ./models/controlmaps [--device cuda]

.venv/bin/python -m thenoise controlmap --input photo.jpg --type canny \
    --canny-low 100 --canny-high 200 --out control_canny.png
.venv/bin/python -m thenoise controlmap --input photo.jpg --type lineart \
    --lineart-threshold 0.5 --out control_lineart.png
.venv/bin/python -m thenoise controlmap --input photo.jpg --type softedge \
    --softedge-threshold 0.2 --softedge-blur 1.5 --out control_softedge.png
.venv/bin/python -m thenoise controlmap --input photo.jpg --type pose \
    --pose-kp-threshold 0.3 --pose-background black --out control_pose.png

.venv/bin/python -m thenoise controlmap --list --controlmap-dir ./models/controlmaps
```

* `--type` repeatable (or `--type all`) → `out.png`, `out_depth.png`, … ; one load
  pass per run. `--input` may also be repeated (§6.1 batch mode is M4).
* `--controlmap-dir` also goes on `serve` (server config, like `--upscaler-dir`).
* `--list` prints kinds, available checkpoints and what is missing, then exits.
* `controlmap_cli.py` is the twin of `upscale_cli.py`: build `Settings`, construct
  `Runtime` **without** `load()`, call `runtime.controlmap.run(...)`, save with
  `utils/png.py:save_output` — the same helper `generate`, `edit` and `upscale` use, so
  the output extension and the provenance chunk are handled once for all four.

### 6.2 Web UI

A fourth tab **"ControlMap"**, structured like Upscale (sidebar + stage + timer), fully
usable with no model loaded:

* dropzone (reuse `bindDropzone`, incl. paste),
* **plain-language labels**: the wire keeps the technical names (`type: "canny"`,
  `canny.low`), the form says *Detect* (filled as *Edges*, *Depth*, *Pose*), *Method*
  for the implementation select, and *Edge cutoff* / *Strong edge cutoff* / *Line width*
  for the canny options, each with a hint. The info modal reuses the same words, so a
  map's metadata reads like the form that made it.
* `type` select `depth | pose | canny`; option blocks shown/hidden per type,
* model select filled from `/controlmaps`; disabled state + the same
  "not available — start the server with `--controlmap-dir`" panel as the upscaler
  tab (`no-model` block, driven from `applyModelState()`),
* no size controls: a map is the size of the image it was extracted from (§4.4), so the
  tab has no width/height/fit fields and no `validateDims` call,
* Run → spinner → output image, **side-by-side input/output** (reuse the `.single` /
  `.result` stage CSS from the upscale tab), Download, and an info modal fed by
  `decodeMeta(blob, 'controlmap_data')`,
* new `js/controlmap.js` (keeps `app.js` from growing past ~600 lines), small CSS
  additions; `helpers.js` gains only an optional label function on `fillSelect`, so a
  select can show words while submitting ids.

---

## 7. Tests (`tests/`, no weights, no GPU, no network)

| File | Covers |
|---|---|
| `test_controlmap_canny.py` | flat image → all black; synthetic square/line → edges exactly on the boundary ±1px; monotonicity in `low`/`high`/`thickness`; hysteresis connects a weak chain to a strong seed and rejects an isolated weak blob; alpha input flattened to white; every stage keeps the geometry, so the mask is exactly the input size |
| `test_controlmap_depth.py` | runner behind a `FakeDepthModel` (`nn.Module` returning a fixed `predicted_depth`): normalisation endpoints 0/255, `invert`, `turbo` LUT bounds, the field interpolated back to the input size, dtype/device args, missing-weights error message names the download command |
| `test_controlmap_pose.py` | renderer: COCO-18 neck derivation, palette per limb, `line_width`/`radius`, `kp_threshold` culling, `background` option, keypoints→canvas coordinates (affine inverse) on a hand-built heatmap; `_hf.py` shim: both name sets (v4/v5) resolve, and the **box-convention golden test** with a fake processor |
| `test_controlmap_registry.py` | discovery from `config.json` `model_type` in a `tmp_path` dir, kind resolution + defaults, unknown name → `ValueError` with hint, 1-resident LRU (load/swap/unload counters), weights-free `canny` runs with an empty dir |
| `test_controlmap_controller.py` | `inference_lock` taken, `inference()` boundary active (`torch.is_inference_tensor` on the output), pnginfo populated, validation errors → 400-mapped `ValueError` |
| `test_api.py` (extend) | `GET /controlmaps` (model-free, empty + filled), `POST /controlmap` PNG and `out:"json"`, 400 unknown type, 400 missing weights, 400 bad base64 |
| `test_cli.py` / `test_entrypoints.py` (extend) | `controlmap` parsing incl. repeated `--type` and defaults; `--list`; `main()` dispatch with a monkeypatched controller; `serve --controlmap-dir` reaches `Settings` |

Everything runs in the existing suite's style (toy tensors, `torch.set_num_threads(1)`,
fakes injected into `Runtime`, `TestClient` for status-code paths — the same shape as
`test_api.py`'s `FakePipeline`/`_runtime`).

---

## 8. Performance expectations (to be measured, not claimed)

Rough order-of-magnitude on a Strix Halo iGPU in BF16, single 1024² image — these are
estimates to validate in the M2/M3 benchmarks, and the reason for the 1-resident policy:

| Stage | Weights | Time (est.) | VRAM delta |
|---|---|---|---|
| canny | 0 | ~20 ms | negligible |
| depth (DA-V2-S, 518²) | 99 MB | ~0.3–0.8 s | ~0.5 GB peak |
| person detect (RT-DETR-r18) | 81 MB | ~0.2–0.5 s | ~0.4 GB peak |
| keypoints (ViTPose-B, 256×192 × P crops, batched) | 344 MB | ~0.2–0.6 s | ~0.8 GB peak |

Pose also touches CPU numpy/scipy for the crop warps and DARK refinement unless D2 =
"vendored", which is where its tail latency will actually come from. First real runs
should also confirm which stages tolerate BF16 (§3).

---

## 9. Milestones

| # | Deliverable | User-visible? | Risk |
|---|---|---|---|
| **M1** ✅ | `controlmap` package (base/canny/manager), controller, CLI + `/controlmap` + `/controlmaps`, UI tab, docs, tests | ✅ works with **zero downloads** | — |
| **M2** | **depth** via Depth-Anything-V2, `scripts/download.py --model controlmap`, options, colormaps, residency/dtype tuned | ✅ | low |
| **M3** | **pose**: detector + estimator + renderer, box-convention golden test, options | ✅ | medium (crop/box conventions, v4/v5 drift) |
| **M4** | follow-ups, each its own decision: hands/face (wholebody), `normal`, folder/batch `--input` mode, GPU-resident controlmap flag, control-map *consumption* in generation (Phase 2). **Done:** turbo colormap, `lineart` and `softedge` controlmaps | — | — |

M1 is deliberately shippable alone: a working `controlmap` surface, tests, docs and a
real feature (canny) before any model weight is involved.

### M1 as built — two deliberate deviations

* **No `--controlmap-dir` yet.** Nothing in M1 reads a checkpoint, so the flag would
  be inert; it lands in M2 together with the `config.json` discovery it exists for.
  Until then `/controlmaps` reports what is implemented, not what is on disk.
* **No `docs/controlmap.md` yet.** With one kind there was nothing to add beyond the
  `controlmap` sections of `docs/cli.md` and `docs/api.md`; the dedicated page comes
  with checkpoints, downloads and per-kind quality notes.

Everything else shipped as specified, including the (a)/(b)/(c)/(f) structure from
§1.1: map kinds come from one `REGISTRY`, `detect()` and `render()` are separate steps,
depth-style fields stay fp32 until `render`, and no generation request gained a field.

### M1 after review

The review pass took surface out rather than adding to it, and generalised the parts that
had grown as twins of the upscale ones.

* **No target size, no fit** (§4.4, **D8**). `width`/`height`/`fit` left the request, the
  CLI flags, the UI form and `base.py` together: `fit_field` and `MAP_FITS` are gone and
  `render(found)` paints at the measured size. `render` also gained a default (paint the
  float field as greyscale), so a controlmap implements `detect` and nothing else
  unless it paints differently — which is why `canny.py` is now one method. Removing the
  resize then uncovered a real bug: the separable Gaussian padded all four sides on both
  passes, so its output was `kernel_size - 1` rows too tall and the resize back to the
  requested size had been hiding it (and shifting every edge) the whole time.
* **Shared with upscale, not mirrored.** `utils/png.py` gained `pnginfo_of` and
  `save_output` (now used by `generate`, `edit` and both model-free CLIs), and both
  provenance chunks are built by one `_provenance` helper; every image-producing route in
  `api.py` goes through one `image_job`, so the 400/500 mapping exists once;
  `utils/device.py` gained `release_device_memory` (collect, then empty the cache) for
  the controlmap pool and `Runtime._unload`; `helpers.js:fillSelect` takes an optional
  label function.
* **The UI speaks plainly** (§6.2): *Detect* / *Method* / *Edge cutoff* / *Strong edge
  cutoff* / *Line width* instead of *Control map* / *Implementation* / *Canny low* /
  *Canny high* / *Line px*.

---

## 10. Decisions

### Resolved

* **D1 — Phase 2 (control conditioning) is not committed.** Structure only, see §1.1.
* **D2 — HF ViTPose processors**, `scipy` in a `[controlmap]` extra (details below).
* **D3 — `transformers` floor stays `>=4.57.6`.** ⇒ `controlmap/_hf.py` carries one
  compat table, and it is small because v5 kept most surfaces compatible. Verified
  against the installed 5.18:

  | Concern | 4.57.x | 5.x | Shim |
  |---|---|---|---|
  | ViTPose class | `VitPoseForKeypointDetection` | `VitPoseForPoseEstimation` | resolve through `AutoModel` on `model_type="vitpose"` (name-independent) instead of importing a class |
  | keypoint post-processing | `post_process_keypoints` | `post_process_pose_estimation` (different box convention) | one adapter function, golden-tested (§4.3 ⚠️) |
  | dtype kwarg | `torch_dtype=` | `dtype=` (`torch_dtype` still accepted, logs a warning) | `_DTYPE_KW` chosen once from `transformers.__version__` — no silent kwarg swallowing |
  | processor backend | `use_fast=False` | `backend="pil"` (`use_fast` deprecated) | `_backend_kwargs()` helper |
  | `AutoModelForDepthEstimation`, `AutoModelForObjectDetection` | ✅ | ✅ | none needed |

* **D5 — naming: `controlmap`** (`controlmap` subcommand, `POST /controlmap`,
  `GET /controlmaps`, `--controlmap-dir`, UI tab "ControlMap").
* **D8 — a control map is its input's size** (§4.4). No `width`/`height`/`fit` anywhere
  in the surface: a precomputed map is paired with the image it was measured on, and
  whoever consumes it does its own fitting. Decided at the M1 review — M2/M3 should not
  reintroduce a size request.

**D2 / D4 / D6 / D7 are decided too; their consequences are spelled out next.**

### Decision consequences

* **D2 — Pose pre/post: HF processors.** ⇒ a new `[controlmap]` optional extra holding
  **`scipy`** only, installed with `uv pip install -e ".[controlmap]"`, documented in
  `docs/setup.md` and the pose docs, and a precise `ImportError`-era error message naming
  that command when it is missing (canny and depth must keep working without it).
  Vendoring the affine crop + DARK refinement in torch stays an M4 option if the CPU
  round-trip shows up in measurements.
* **D4 — Default checkpoints (decided): Depth-Anything-V2 Small + RT-DETR r18vd**, the
  larger pair behind `--depth-model` / `--pose-detector` selects, all four offered by
  `download.py --model controlmap` (defaults fetched by default, opt-ins behind a flag).

  | | Params | Weights | Relative cost | Quality note |
  |---|---|---|---|---|
  | **Depth-Anything-V2 Small** (ViT-S/14 + DPT) | 24.8 M | 99 MB | 1× | solid scene layout; hair, fences, thin rails and far-background separation get mushy |
  | Depth-Anything-V2 Base (ViT-B/14) | 97.5 M | 390 MB | ~3–4× | visibly crisper boundaries; the "I want a pretty depth map" choice |
  | Depth-Anything-V2 Large (ViT-L/14) | 335 M | 1.34 GB | ~10× | marginal over Base for control use; only worth it when RAM is free |
  | **RT-DETR r18vd (COCO+O365)** | ~20 M | 81 MB | 1× | person detection is an easy task (paper COCO AP ≈ 50 at 640²); misses only heavily truncated, tiny or crowded people |
  | RT-DETR r50vd | 42 M | 172 MB | ~2× | AP ≈ 53–55; better recall in crowds and under occlusion |

  Missing a person is a **hard** failure (that person is simply absent from the map); a
  false positive is a **soft** one (one stray skeleton). So if in doubt, spend the extra
  90 MB on the detector and keep Small for depth. Both stay selectable per request
  either way.
* **D6 — Residency (decided): last-used controlmap stays resident**, exactly like
  `PixelUpscalerManager.switch` — swap = drop + `gc.collect()` +
  `clean_memory_on_device()`; re-requesting the same controlmap is a no-op.
* **D7 — Hands/face (decided): COCO-18 body skeleton for v1** (17 COCO keypoints + the
  derived neck). Wholebody is not achievable with the transformers stack (§4.3);
  reopening it would mean the ONNX/DWPose route (CPU inference, non-commercial licence
  provenance) and is an explicit M4 decision, not a v1 one.

### Deferred (each needs its own go-ahead)

§1.1 items (d), (e), (g), (h) — i.e. any actual control conditioning inside `thenoise` —
plus the M4 list: wholebody (hands/face), `normal`, folder/batch
`--input` mode, and vendoring the pose crop/refine to drop `scipy`.
