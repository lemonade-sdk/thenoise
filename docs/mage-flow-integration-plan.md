# Mage-Flow integration plan

Target: `https://huggingface.co/Comfy-Org/Mage-Flow` (microsoft/Mage-Flow, MIT), already
downloaded to `models/mage_flow/` and already supported by `scripts/download.py`
(`--model mage-flow`, commit `4356b58`). Everything below is the engine work.

Reference implementations consulted (both fetched, MIT):

* ComfyUI `comfy/ldm/mage_flow/model.py` — the DiT (186 lines). **Primary reference**,
  because our DiT is a port of the same thing onto our own Qwen-Image blocks.
* ComfyUI `comfy/ldm/mage_flow/vae.py` — the Mage-VAE (477 lines). **Primary reference**
  for the codec.
* ComfyUI `comfy/text_encoders/mage_flow.py` — the Qwen3-VL conditioner wrapper.
* ComfyUI `comfy/supported_models.py` (`class MageFlow`), `comfy/model_base.py`
  (`class MageFlow(QwenImage)`).
* Upstream `microsoft/Mage` `mage_flow/{pipeline.py,models/utils.py,models/mage_flow.py}` —
  sampling grid, prompt templates, edit conditioning. Used to confirm the ComfyUI port
  and to settle defaults.

---

## 1. What Mage-Flow actually is (research findings)

**DiT** = a **12-layer** variant of the Qwen-Image *double-stream* block, with:

| knob | value | note |
|---|---|---|
| `num_layers` | 12 | Qwen-Image is 60 |
| `num_attention_heads` × `attention_head_dim` | 24 × 128 → inner 3072 | same as Qwen-Image |
| `in_channels` / `out_channels` | 128 / 128 | the VAE's raw latent, **no patchify** |
| `patch_size` | **1** | one token per latent cell (Qwen-Image packs 2×2 → 64ch) |
| `joint_attention_dim` | 2560 | Qwen3-VL-4B hidden size (Qwen-Image: 3584 / 2.1: 4096) |
| `axes_dims_rope` | (16, 56, 56), theta 10000 | identical to Qwen-Image |
| block internals | `img_mod.1`/`txt_mod.1`, `img_norm1/2`, `txt_norm1/2`, `attn` (`to_q/k/v`, `to_out.0`, `add_*_proj`, `to_add_out`, `norm_q/k`, `norm_added_q/k`), `img_mlp.net.0.proj`+`net.2` | **byte-identical names to Qwen-Image** |

Three architectural differences from Qwen-Image, all confirmed in the reference:

1. **`process_img` (patch 1)**: tokens are raw latent pixels
   `x.movedim(1,-1).reshape(B, H*W, C)`; position ids are `(t, h, w)` with
   `t = image index` (0 = target, 1..N = references), and h/w **centered by
   `r - (n - n//2)` = `r - ceil(n/2)`** — which is exactly our `grid_positions(centered=True)`
   convention (ComfyUI's *Qwen-Image* uses `-(n//2)`, Mage's uses `-ceil(n/2)`; the
   reference module calls this out in a comment).
2. **Text tokens are NOT rotated**: `txt_ids = zeros(B, L, 3)` (RoPE at 0 = identity).
   Qwen-Image advances text positions (`build_txt_positions`).
3. **Timestep table is rounded to the timestep dtype**:
   `emb = exp(-log(10000)·arange(128)/128)` → `.to(timestep.dtype)` → `×1000` →
   `[sin, cos]` → `flip_sin_to_cos` ⇒ `[cos(t·1000·f), sin(t·1000·f)]` with `f` **bf16-rounded**.
   Our shared `timestep_embedding(t, 256)` computes the same table in fp32. ComfyUI keeps
   the bf16 rounding on purpose (`model_base.MageFlow.process_timestep` forces bf16: "Mage
   runs in bf16 and rounds its timestep frequency table to the timestep dtype").
   The angle error at the high-frequency end is *not* negligible (up to ~1 rad), so we
   replicate the rounding.

No `zero_cond_t` / timestep-zero reference modulation anywhere in the Mage DiT: the
timestep embedding has **one row**, so reference tokens are modulated at `t` like the
target. Consequence: reference K/V are *not* step-invariant ⇒ **no KV cache for Mage**
(see decision D6).

No pooled/vector conditioning: upstream's transformer computes
`txt_vec = zeros(B, inner_dim); temb = temb + txt_vec` — a no-op. Our DiT needs no `vec`.

**VAE (Mage-VAE)** — a *symmetric one-step diffusion codec*, not a conventional VAE:

* `DConvEncoder` (image → latent): one-step prediction at `t = 0` with a **zero latent**
  `z_t` fed in, output `proj_out` = 256ch = `[mean(128), logvar(128)]`, take the mean.
  `patch_size 16`, `hidden 384`, `head 768` (2 `EncoderDiCoBlock`s), 21 `DiCoBlock`s,
  `mlp_ratio 4`.
* `DConvDenoiser` + `CoDDecoder` (latent → image): one-step at `t = 0` with a **zero noise**
  image, `hidden_size 384`, `hidden_size_x 32`, 21 cond blocks + 3 `SimpleMLPAdaLN` res
  blocks, `bottleneck 128`, `patch 16`, Nerf/DCT positional embedder (`max_freqs 8`),
  CoD decoder = 3 ResnetBlocks + 2 **patched** (windowed, `patch_size 32`) attention blocks.
* **Latent: 128 channels, 16× spatial, NO scaling/shift, NO BatchNorm, NO 2×2 packing.**
  ComfyUI's model config uses `latent_formats.Flux2` (128ch/16×, `scale_factor=1.0`,
  `shift_factor=0.0`). The anchor-KL regularises toward Flux.2-VAE latents, so the space is
  Flux.2-*anchored* but the tensor is used raw. This is the canonical latent here.
* Encode/decode are deterministic single forwards (upstream *samples* the posterior;
  ComfyUI takes the mean — we take the mean, our reference stage is cached and must be
  deterministic).

**Text encoder** = **Qwen3-VL-4B** (`qwen3vl_4b_bf16.safetensors`, `model.language_model.*`
+ `model.visual.*`), conditioning = the **last hidden state WITH the final RMSNorm applied**
(ComfyUI `layer_norm_hidden_state = True`) ⇒ HF `.last_hidden_state` directly (unlike
Qwen-Image 2.1, which taps *before* the norm — see `dit/qwen_image21/encoder.py`).
Templates (verified identical in upstream `models/utils.py::PROMPT_TEMPLATE` and ComfyUI):

(`S` below = `<\|im_start\|>`, `E` = `<\|im_end\|>`; both are the literal
Qwen chat markers, written escaped so this file stays plain text)

```
mage-flow      : S system\nDescribe the image by detailing the color, shape, size,
                 texture, quantity, text, spatial relationships of the objects and
                 background: E\n S user\n {prompt} E\n S assistant\n
                 drop_idx = 34
mage-flow-edit : same shape, system prompt = Qwen-Image-Edit's "Describe the key features
                 of the input image (color, shape, size, texture, objects, background),
                 then explain how the user's text instruction should alter or modify the
                 image. Generate a new image that meets the user's requirements while
                 maintaining consistency with the original input where appropriate."
                 drop_idx = 64
```
* Both end right after `S assistant\n`: **no thinking block is appended**
  (ComfyUI's wrapper passes `thinking=True` for exactly that reason). Upstream's
  `default-nonthinking` template is the only one that appends an empty
  "think" block; neither Mage template does.
* The t2i template is *the same string* as our `QWEN_VL_SYSTEM_PROMPT`/`QWEN_VL_PROMPT_SUFFIX`
  (`thenoise/utils/text_encoder`), i.e. `QWEN_VL_DROP_IDX = 34` is already right for Mage t2i.
* Edit multi-reference body: `"".join(f"Image {j}: {VISION_BLOCK}" for j in 1..N) + instruction`
  (upstream `_edit_prompt_body`, whose `_EDIT_IMAGE_PLACEHOLDER` is
  `<\|vision_start\|><\|vision_end\|>` — **no `<\|im_pad\|>` inside**; the HF
  processor expands/inserts the pad, and our other adapters use
  `VISION_START + IMAGE_PAD + VISION_END`, which is the form to use here),
  substituted into the template's `{}`. Vision tokens are **kept** in the conditioning
  (only the `drop_idx` prefix is removed) — exactly like our Qwen-Image adapter, where the
  reference reaches the DiT twice (vision tokens in the text stream + reference latent
  appended to the image stream).
* The drop index is derived, not counted: the position of the **second** chat-start
  marker plus its `user` / newline header tokens (3 tokens). Our
  `dit/qwen_image/utils.py::_compute_drop_idx` already implements exactly that — make it
  shared (see D10) instead of hard-coding 34 / 64.

**Sampling** (upstream `pipeline.build_scheduler` / `_get_scheduler`):
`FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=6.0,
use_dynamic_shifting=False)` with
`set_timesteps(sigmas=linspace(1.0, 1.0/steps, steps))`, static shift applied
`σ' = shift·σ / (1 + (shift−1)·σ)` and a terminal `0` appended. The model receives the
**shifted σ** as its timestep (`t_vec = sigma`, the ×1000 happening inside the timestep
embedder) — i.e. exactly our `Step.t` convention. **Static shift, never dynamic**, so the
grid is resolution-independent. `shift = 6.0` (also ComfyUI's `sampling_settings =
{"multiplier": 1.0, "shift": 6.0}`).
Note `generalized_time_shift(t, log(6), 1.0) ≡ 6t/(1+5t)` — the shared helper in
`thenoise/utils/math.py` reproduces the static shift exactly.

**Variants and defaults** (upstream README + model zoo):

| variant | steps | cfg |
|---|---|---|
| Base | 30 | 5.0 |
| RL-aligned (`mage_flow_*`) | 20 | 5.0 |
| Turbo | 4 | 1.0 |

Sizes: multiple of **16**, native **512–2048**, any aspect ratio (4:1 works; there is no
bucket quantisation — "native resolution" is the point of the architecture).
Edit: reference latents are VAE-encoded **at the output size** (upstream stretches to
`(h, w)`; our pipeline cover-crops — see D8); the *VL conditioning* image is downsized
(`vl_cond_long_edge = 384`) while the VAE path keeps full resolution.
Optional `renormalization` (CFG norm renorm per token) defaults to **False** — not ported.

**Checkpoint inventory** (`models/mage_flow/`):

| file | what | keys |
|---|---|---|
| `mage_flow_turbo_int8_convrot.safetensors` (4.1 GB) | t2i DiT, INT8+ConvRot | flat (`img_in.`, `txt_in.`, `txt_norm.`, `time_text_embed.timestep_embedder.linear_1/2`, `transformer_blocks.0..11.*`, `norm_out.linear`, `proj_out`) + `.weight_scale` F32[out,1] + `.comfy_quant` U8 markers |
| `mage_flow_edit_turbo_int8_convrot.safetensors` (4.1 GB) | edit DiT | **key-identical** to the t2i file |
| `mage_flow_vae_bf16.safetensors` (345 MB) | Mage-VAE | `student.dconv_encoder.*` → encoder, `pipeline.*` → decoder, **plus `pipeline.y_embedder.encoder.*` = the Flux.2 anchor VAE, dropped at load** |
| `qwen3vl_4b_bf16.safetensors` (8.9 GB) | Qwen3-VL-4B TE | HF `model.language_model.*` / `model.visual.*` layout |

All VAE shapes match the reference module's *defaults* exactly (verified from the header:
encoder 2 head blocks + 21 blocks, `proj_out` 256ch, decoder 21 cond blocks + 3 res blocks,
`x_embedder` in 99 = 3+32+8², `y_embedder_x` 8192 = 32·16², `s_embedder.proj1` 128, CoD
`conv_in` 128→384). The DiT header shapes give the whole architecture
(`img_in [3072,128]`, `txt_in [3072,2560]`, `attn.norm_q [128]`, 12 blocks).

---

## 2. What is reused, and how

| need | reuse | detail |
|---|---|---|
| transformer block | **`thenoise.dit.qwen_image.models.QwenImageTransformerBlock`** | names/structure identical; pass `eps=1e-6` (ComfyUI's value for both the block LayerNorms and the attention QK RMSNorms; our Qwen-Image adapter uses 1e-5). Already `@torch.compile(fullgraph=False)`, already handles `bufs=None` (no KV cache) |
| joint attention | same module's `Attention` | txt-first-then-image concat + `attend(...)`; `bufs=None` is the plain path |
| `FeedForward` / `TimestepEmbedding` / `AdaLayerNormContinuous` / `Attention` QK `RMSNorm` | same module | `AdaLayerNormContinuous(inner, inner, elementwise_affine=False, eps=1e-6)` == ComfyUI `LastLayer` (header: `norm_out.linear [6144,3072]`) |
| txt input RMSNorm | `thenoise.utils.rms_norm.RMSNorm` | `RMSNorm(2560, eps=1e-6)` == `txt_norm` |
| RoPE | `thenoise.utils.rope.RopeCache(matrix_rope([16,56,56], 10000))` | identical to ComfyUI's `EmbedND(128, 10000, [16,56,56])` |
| image position ids | **`thenoise.dit.qwen_image.models.build_video_positions`** | `(frame, h, w)` grid with `start=[i,0,0]`, `centered=[False,True,True]` — with `frame=1` per image this *is* Mage's `(index, h−ceil, w−ceil)`. `build_txt_positions` is **not** used (Mage text is unrotated) |
| timestep MLP | `dit/qwen_image.models.TimestepEmbedding` | only the *frequency table* is Mage-specific (D3) |
| quantized loading | `thenoise.utils.loader.load_dit` + `thenoise.dit.quantized.replace_linears/QuantizedLinear` | the int8+ConvRot export loads as-is; no `state_map`/`key_map`/`drop_keys` needed |
| text-encoder loading | `thenoise.utils.text_encoder.load_qwen3_vl_model` with the already-vendored `QWEN3_VL_4B_INSTRUCT_CONFIG` + `load_qwen3_vl_tokenizer`/`load_qwen3_vl_processor` + `QWEN3_VL_TOKENIZER_OVERRIDES` | Krea 2 already loads this exact file |
| prompt helpers | `QWEN_VL_SYSTEM_PROMPT`/`QWEN_VL_PROMPT_SUFFIX`/`QWEN_VL_DROP_IDX` (t2i template is literally the same), `thenoise.utils.sequence.pad_to_batch`, `thenoise/utils/image_tensor.resize_to_area`/`flatten_alpha` | see D9 for the 384 resize |
| VAE | **new** `thenoise/vae/mage_flow.py` (port of ComfyUI `mage_flow/vae.py`) | nothing in `vae/` is close enough (it is not a KL-VAE); the port is mechanical, only `comfy.ops` (`→ nn.*`) and `comfy.ldm...vae_attention` (`→ F.scaled_dot_product_attention`) change |
| scheduler / sampler / LoRA / memory / pipeline / API / CLI / UI | untouched | LoRA works out of the box: no fused modules (`to_q/to_k/to_v` separate, GELU MLP), so `lora_fusions = {}` and the identity `_lora_key_map` |

---

## 3. Implementation steps

### 3.1 `thenoise/dit/mage_flow/` (new package)

* `__init__.py` — package doc + re-exports.
* `models.py`
  * `MageFlowParams` (`in_channels=128`, `out_channels=128`, `num_layers=12`,
    `num_heads=24`, `head_dim=128`, `context_dim=2560`, `axes_dims=(16,56,56)`,
    `patch_size=1`) built from the checkpoint header (D1).
  * `MageTimestepProjEmbeddings` — the bf16-rounded table (D3) → shared `TimestepEmbedding`.
  * `MageFlowTransformer2DModel` — `pe_embedder` (RopeCache), `time_text_embed`, `txt_norm`,
    `img_in`, `txt_in`, 12× reused `QwenImageTransformerBlock(eps=1e-6)`, `norm_out`
    (`AdaLayerNormContinuous`, eps 1e-6), `proj_out` (`inner → out_channels`, no patch² factor).
    `forward(hidden_states, encoder_hidden_states, timestep, img_pe, txt_pe, ref_tokens=None)`
    → concat refs → `img_in` / `txt_norm`+`txt_in` → temb → block loop
    (`mark_token_axis` per block, `bufs=None`) → drop the ref tokens
    (`hidden_states[:, :num_img_tokens]`) → `norm_out`/`proj_out` → **tokens** `[B, L, 128]`
    (reshape back to `[B,128,h,w]` is the adapter's `finalize_latent`, mirroring Qwen-Image).
    No `timestep_zero_index`, no `kv`, no `control`, no `patches*` options.
  * `latent_to_tokens` / `tokens_to_latent` — the patch-1 (un)pack:
    `[B,C,H,W] ↔ [B, H·W, C]` (the analogue of `utils/latents.pack_latents`, which is 2×2
    and does not apply here).
  * `load_mage_flow_dit(path, params, device, dtype)` — `init_empty_weights` + `load_dit`.
* `utils.py` — `detect_params(path)` (header shapes, like `dit/qwen_image21/utils.py`),
  plus a `_header_shapes` helper of the same shape.
* `keys.py` — family/detection predicates (D2).
* `sampling.py` — `SHIFT = 6.0`, `get_sigmas(steps, shift=6.0)` =
  `generalized_time_shift(linspace(1, 1/steps, steps), log(shift), 1.0)`,
  `get_schedule(steps)` returning `steps+1` grid points ending at 0.
* `encoder.py` — `MageFlowTextEncoder(nn.Module)`: `(prompt, images=None) -> (embeds, mask)`.
  Built on the model pattern of `dit/qwen_image21/encoder.py` (tokenizer + processor +
  `qwen.model(...)`), but: take **`.last_hidden_state`** (post-norm), keep vision tokens,
  strip the template prefix with the shared drop-index helper, `pad_to_batch` +
  `make_key_padding_mask` for the (single-sequence, therefore all-ones) mask. Own templates
  (`MAGE_T2I_TEMPLATE`, `MAGE_EDIT_TEMPLATE`, `Image {j}: ` vision prefix).
  `load_mage_flow_text_encoder(path, dtype, device)`.

### 3.2 `thenoise/vae/mage_flow.py` (new)

`AutoencoderKLMageFlow` (`z_dim = 128`, `spatial_compression = 16`, `pixel_channels = 3`,
`dtype`, `device`) with `dconv_encoder` + `decoder_model`, `encode_pixels_to_latents`
(zeros `z_t`, zeros `t`, `proj_out` mean) and `decode_to_pixels` (CoD cond from `z`, zeros
noise, zeros `t`, clamp to `[-1, 1]`). Ported classes: `LayerNorm2d`, `TimestepEmbedder`,
`BottleneckPatchEmbed`, `DiCoBlock`, `EncoderDiCoBlock`, `NerfEmbedder`, `NerfFinalLayer`,
`MLPResBlock`, `SimpleMLPAdaLN`, `ResnetBlock`, `AttnBlock` (windowed, `patch_size=32`, SDPA),
`CoDDecoder`, `DConvEncoder`, `YEmbedder` (**decoder only**), `DConvDenoiser`.
`load_mage_vae(path, device, dtype)` remaps `student.dconv_encoder.* → dconv_encoder.*`,
`pipeline.* → decoder_model.*`, and **drops `pipeline.y_embedder.encoder.*`** (the anchor
Flux.2 VAE the export ships) and `__metadata__`; strict load.
Export from `thenoise/vae/__init__.py`.

### 3.3 `thenoise/models/mage_flow.py` (new adapter) + registration

* `class MageFlowModel(DiffusionModel)`, `name = "mage_flow"`,
  `DEFAULT_PREFS = {steps: 4, guidance_scale: 1.0, sampler: "euler"}` (D7),
  `CAPABILITIES = {edit: True, kv_cache: False}` (D6),
  `resolve_size` = `round_up(·, vae.spatial_compression)` = 16 (D8),
  `percent_to_sigma = 1.0 - percent` (the shifted grid starts exactly on 1.0, which the
  ER-SDE solver divides by).
* `detect(f)` per D2; register in `thenoise/models/__init__.py` (`MODEL_CATALOG`)
  **and** `pyproject.toml` `[tool.setuptools] packages` (`thenoise.dit.mage_flow`).
* `encode_prompt` — t2i template, or the edit template with the reference images when
  `args.image` is not `None` (normalise single-or-list); returns
  `Conditioning(cond, cond_mask, null, null_mask)`; only encodes `null` when
  `guidance_scale > 1.0`.
* `init_latents` — canonical `randn(1, 128, h//16, w//16)`.
* `prepare_latent` — `latent_to_tokens`, stash text + per-branch text, build
  `img_pe = build_video_positions([(1,h,w)] + refs)` split into `img`/`ref` (target first,
  refs with index 1..N), stash the **identity** `txt_pe` (D4). No KV caches started.
* `schedule` — `mage_flow.sampling.get_schedule(params.steps)` (resolution-independent;
  `Step(t=ts[i], delta=ts[i]-ts[i+1])`).
* `denoise_step` — autocast bf16, plain CFG (`neg + gs·(pos-neg)`, **no** Qwen norm
  renormalisation).
* `finalize_latent` — `tokens_to_latent`.
* `encode_reference` = `vae.encode_pixels_to_latents`;
  `pack_reference_latent(latents, method, ref_index)` = tokens +
  `build_video_positions([(1,h,w)], start=[ref_index,0,0], centered=[False,True,True])`
  — note the reference's frame axis is the **1-based image index**, so `ref_index` is used
  directly (no Flux.2-style `REF_INDEX·i` scaling), and it rejects anything but `index` (D6).
* `_create_upscaler` → `VAEPixelUpscaler(self.vae, scale=self.UPSCALE_SCALE)` (D5).

---

## 4. Decisions taken

**D1 — architecture from the header, not hard-coded.** `detect_params(path)` reads
`img_in`/`txt_in`/`attn.norm_q` shapes and the block count, the way
`dit/qwen_image21/utils.py::detect_params` does. Costs nothing and lets a future 24-layer
or different-width export load unchanged.

**D2 — detection: names + depth.** The Mage DiT's tensor *names* are a **superset-equal
match for Qwen-Image's** (`img_in`, `txt_in`, `txt_norm`, `time_text_embed.timestep_embedder.*`,
`transformer_blocks.*.attn.add_q_proj`, `img_mlp.net.2`, `norm_out.linear`, `proj_out`) and
the t2i/edit files are key-identical, so the existing `QwenImageModel.detect` would claim
the Mage file. Only two things separate them and neither is a name: **depth** (12 vs 60) and
**tensor shapes** (`img_in [3072,128]` vs `[3072,64]`). `detect(f)` is name-only by contract
(`safe_open`/`FakeHandle` expose names, `resolve()` iterates the catalog on names), so:
add `thenoise/dit/mage_flow/keys.py`-style helpers
(`is_qwen_image_family(keys)`, `dit_block_count(keys)`, `MAGE_LAYERS = 12`) and
* `MageFlowModel.detect` = family **and** depth == 12,
* `QwenImageModel.detect` = family **and** depth > 12 ("the deep member of the family").
Both stay positive predicates with no cross-model coupling, and catalog order stops mattering.
(Shape-based detection was rejected: `safetensors.safe_open` exposes no `.header`, and the
whole detection test-matrix is name-based.)

**D3 — replicate the bf16-rounded timestep frequency table** in a Mage-only
`MageTimestepProjEmbeddings` instead of reusing `timestep_embedding(t, 256)`: `exp(...)`
in fp32, `.to(t.dtype)`, then `×1000` and `[cos, sin]`. Faithful to both references; the
fp32 table drifts by up to ~1 radian at the high-frequency end.

**D4 — text RoPE = one identity matrix, broadcast.** Store the identity pe as
`pe_embedder.store("txt", zeros(1,1,3))` (cos 0 = 1, sin 0 = 0 → 2×2 identity) and let
`apply_rope` broadcast it over the text length. Expresses "text is unrotated" literally and
skips building (and reading) a `[B, L_txt, 64, 2, 2]` tensor per prompt.

**D5 — latent upscaler = `VAEPixelUpscaler` (decode → 2× bicubic → re-encode).** There is no
trained latent upscaler for this space: the Sesqui `flux2` weights are trained on Flux.2's
*BatchNorm-normalised packed* latent and Mage's latent is the raw anchor space (same shape,
different statistics), so reusing them would be a silent mismatch — Ming-Image set the
precedent of choosing the VAE round trip for exactly this reason, and Mage-VAE's whole point
is that its codec is ~12×/22× cheaper per pixel than Flux.2-VAE's, so the round trip is the
cheapest refine path in the repo.

**D6 — no KV cache, `ref_method` stays `index`.** The Mage DiT has no timestep-zero
conditioning (single-row `temb`), so reference K/V are *not* step-invariant and freezing them
would silently degrade edits. `CAPABILITIES["kv_cache"] = False` makes the pipeline reject
`kv_cache` outright (it already raises for a model without the capability, and the
kv_cache⇒`index_timestep_zero` auto-switch therefore never fires), and
`pack_reference_latent` rejects `index_timestep_zero` explicitly rather than behaving like
`index`. Reference positions differ from Flux.2/Qwen-Image: the frame axis is the plain
**1-based reference index** (0 = target).

**D7 — defaults are the Turbo recipe** (`steps=4`, `guidance_scale=1.0`, `euler`), matching
Flux-Klein's precedent ("distilled defaults, the common inference use; base models pass
`--steps/--guidance-scale`"), because the int8-convrot Turbo files are what the download
script and `models/mage_flow/` carry and none of the checkpoints carries a marker to detect
the variant (`__metadata__` is `{}`, so `checkpoint_prefs` is empty). Base = `--steps 30
--guidance-scale 5`, RL = `--steps 20 --guidance-scale 5`; documented in the model doc.

**D8 — geometry follows the repo, not upstream, where they differ harmlessly.** Sizes are
rounded **up** to 16 (`round_up`, repo-wide convention) rather than down (`_make_divisible_by_16`
in upstream). Reference images go through the pipeline's existing ComfyUI-style
cover-and-crop-then-`encode_reference` instead of upstream's aspect-stretching `TF.resize`.
Native 512–2048 is a quality range, not a hard constraint; no extra `resolve_size` clamping.

**D9 — the VL conditioning image is downsized with the existing `resize_to_area(img, 384·384)`,
not upstream's `_resize_long_edge(384)`.** Both exist to keep the vision tokens of a
full-resolution edit image from drowning the instruction, and the repo already uses the
area-based form for Qwen-Image; the encoder must **not** use
`args.width/height` / `resize_to_cover_center_crop` (unlike `qwen_image21`'s
`_encoder_images`) because Mage keeps vision tokens *and* reference latents as separate
token groups, so their counts need not agree. Exact long-edge parity with upstream is a
one-line swap if we prefer fidelity over consistency — **open for the maintainer.**

**D10 — share the drop-index helper.** `dit/qwen_image/utils.py::_compute_drop_idx`
(counted from the token ids: the **second** `<\|im_start\|>` plus the
`user` / `\n` tokens that follow it) is exactly Mage's `drop_idx` for *both* templates
(34 / 64), so it moves to `thenoise/utils/text_encoder/` as a public helper and both
adapters use it, instead of hard-coding 34/64.

**D11 — conditioning block list is trimmed, never masked.** Batch size is 1 everywhere, so
after trimming the template prefix (and padding to a batch of one) the mask is all-ones and
ComfyUI itself drops it (`extra["attention_mask"].sum() == numel → pop`). Mage therefore needs
no attention-mask plumbing in the reused `Attention`; the mask is carried in `Conditioning`
only to report lengths, exactly as Qwen-Image does.

**D12 — no `control` (ControlNet), no `patches`/`patches_replace`, no `renormalization`, no
packed varlen batching.** All four are in the reference and none has a thenoise equivalent;
our single-request-per-forward path already covers the multi-reference case by concatenating
the reference token groups.

**D13 — content screening (`screen_edit`, refusal images) is not ported.** It is upstream
product policy built on extra VL forward passes, not part of the model.

**D14 — VAE encode takes the posterior mean.** Deterministic, matches ComfyUI; upstream
samples the posterior with the global RNG, which would break the pipeline's cached
reference-latent stage.

---

## 5. Open questions

1. **D9**: keep the repo's area-based 384² resize for the VL image, or switch to upstream's
   long-edge-384 cap for exact parity?
2. **D5**: is the VAE round trip acceptable as Mage's `--upscale` path, or do we want a
   trained transcoder for the Mage space later (the `Qwen21TranscodeUpscaler` route)?
3. Variants: keep Turbo defaults (D7), or auto-detect Base/RL somehow? Nothing in the
   checkpoints distinguishes them, so the CLI/API must stay the source of truth.
4. Should the *edit* DiT be preferred over the t2i DiT in any way, or is "whatever `--dit`
   points at" enough (they are architecturally identical; the edit one just also accepts
   reference tokens)?
