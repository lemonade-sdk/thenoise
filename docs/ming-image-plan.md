# Ming-Image support plan

Status: **in progress** — P1 (shared `dit/lumina` core) and P2 (Ming DiT in both
exports, VAE loader, dynamic-shift schedule, adapter + catalog + docs + tests) have
landed. P3 is the text encoder: `MingImageModel._encode_prompt` is the one hole and
raises `NotImplementedError`, so every kernel downstream of it is real and tested but
nothing can draw yet (README marks the model `wip`).
Scope agreed: text-to-image only for the first cut
(no editing, no `Design-Layer`, no vision tower), default 1024×1024, int8-convrot supported
for both DiT and text encoder like the other models, refactoring welcome — clean architecture
outranks API stability.

> **P1 notes** (what actually got built, and two deviations from the sketch below):
>
> * The core exposes `pad_mode` + `cap_extra` as designed, but the pad *fill* moved out
>   of `patchify_and_embed`: streams now carry their real tokens plus
>   `(valid, padded)` lengths (`TokenStream`), and `utils.sequence.pad_to_length` fills
>   the pads once the features are at DiT width. Padding the caption before
>   `cap_embedder` cannot work at all once `cap_extra` (already dim-wide) is spliced in,
>   and the embedder is per-token, so the values are unchanged.
> * `load_dit`'s per-tensor `value_map` cannot express "three checkpoint tensors become
>   one parameter", so the `to_q/to_k/to_v → qkv` concat is a state-dict fold passed as
>   the new `load_dit(..., state_map=...)` hook (`dit/lumina/keys.py`), not a `value_map`.

> **P2 notes** (deviations from the sketch, all deliberate):
>
> * **Shift bucket**: `seq > 4096`, not the vendor's `>=`. The two differ only at
>   exactly 1024² — the model's own default bucket — where `>=` gives 3.857 while the
>   reference documents 3.16 there. `>` gives 3.158, i.e. ComfyUI's constant, and is
>   identical above 1024². Commented in `dit/ming_image/sampling.py`, asserted in a test.
> * **Runtime payload**: the Lumina exports carry `attention.comfy_attention.config`
>   (a U8 JSON blob of ComfyUI's attention helper) next to the weights. It is not
>   module state, so the loader runs `keys.drop_runtime_state` first — otherwise the
>   strict load rejects an otherwise-complete checkpoint.
> * **VAE**: one architecture constant and the two latent normalisations
>   (`latents_mean/std` xor `scale_factor/shift_factor`); the pixel width is read off
>   `encoder.conv_in`/`decoder.conv_out` and each named loader declares the width its
>   model needs, so a mismatched file raises instead of silently writing the wrong
>   number of channels. The latent width is validated against the file the same way.
> * Ming-Image **is** in `MODEL_CATALOG` (P2 acceptance, per §6). `detect` reads
>   `is_s3dit(keys) and not has_learned_pad_tokens(keys)`; Z-Image reads the same pair
>   the other way round, because the int8 exports name every module identically.

## 0. Reference sources

| Source | What it gives us |
|---|---|
| `Comfy-Org/ComfyUI` `comfy/ldm/lumina/model.py` | the NextDiT/`ming_image` forward path (`image_model == "ming_image"`, `masked_pad_multiple`, `direct_context`, `ref_frames`) |
| `Comfy-Org/ComfyUI` `comfy/text_encoders/ming_image.py` | the condensed BailingMM2 encoder: `BailingMoeV2` thinker, `MingConnector`, query tokens, `proj_directvlm`, tokenizer-in-safetensors |
| `Comfy-Org/ComfyUI` `comfy/model_detection.py`, `comfy/sd.py`, `comfy/latent_formats.py` | the Ming DiT hyper-parameters, the Wan-2.1-VAE detection branch, `scale_factor = 8.0064` |
| `inclusionAI/Ming-Image` (Apache-2.0) `diffusion/{transformer,pipeline,generator,padding}.py` | the *vendor* reference: `zero_masked` alignment padding, `cap_feats_2`, the dynamic-shift schedule, RGBA decode |
| `inclusionAI/Ming-Image-0.1-Design` `*/config.json` | authoritative configs for transformer / scheduler / mllm / connector / mlp / vae |
| `models/ming_image/*` (local weights) | tensor-level verification of every key/dtype/shape claim below |

License note: the reference code paths are Apache-2.0 (Ant Group `Ming-Image`, Alpha-VLLM
Lumina, diffusers) and the weights are MIT — compatible with this repo's Apache-2.0. Keep
attribution headers in the style of `thenoise/dit/zimage/models.py` and `thenoise/vae/wan22.py`.

## 1. What Ming-Image is (measured, not guessed)

### 1.1 DiT — the same S3-DiT family as Z-Image

`transformer/config.json` (vendor) + the local 12 GB bf16 header agree:

```
dim=3840  n_layers=30  n_refiner_layers=2  n_heads=30  n_kv_heads=30 (head_dim 128)
in_channels=16  patch_size=2  f_patch_size=1  cap_feat_dim=2560
axes_dims=[32,48,48]  axes_lens=[20480,512,512] (unused by our RoPE builder)  rope_theta=256.0
norm_eps=1e-5  qk_norm=True  t_scale=1000.0  ffn = int(dim/3*8) = 10240
alignment_padding_mode="zero_masked"  multi_frame_output=false  siglip=null
```

Deltas versus `thenoise/dit/zimage/models.py` — only **five**, and four are cosmetic:

1. **Checkpoint naming** (bf16 file only, see §1.5): `all_x_embedder["2-1"]`,
   `all_final_layer["2-1"]`, `attention.to_q/to_k/to_v`, `attention.to_out.0`,
   `attention.norm_q/norm_k`.
2. **No learned pad tokens.** The file has no `x_pad_token`/`cap_pad_token`. Padding to
   `SEQ_MULTI_OF=32` still happens, but the pad slots are filled with **zeros** and are
   **masked out of attention** (vendor `mask_out_alignment_padding`, ComfyUI
   `masked_pad_multiple=32`). Z-Image fills them with learned tokens and *attends* to them.
3. **A second conditioning tensor** (`cap_feats_2` / `direct_context` / "directVLM"),
   concatenated to the caption stream **after** `cap_embedder`, because it is already
   DiT-width (3840).
4. **Timestep**: `t = (1 - sigma)` then `t * 1000` — identical to our shared
   `timestep_embedding(time_factor=1000)`. No change.
5. **Sequence order** `[image, caption]` — already what our Z-Image port does. No change.

So ~90 % of the DiT is shared code with Z-Image; the refactor in §3 makes that explicit.

### 1.2 Conditioning is two tensors

The text encoder emits both (`mlp/config.json`: `selected_hidden_states_layers=[5,12,20]`,
`use_vlm_directvlm_condition=true`, `use_learnable_token_condition=true`):

* `cap_feats` `[256, 2560]` — 256 learned **query tokens** are appended to the prompt as an
  `<image><imagePatch></image>` block; the thinker's hidden states at that block go through
  `proj_in → connector (28-layer Qwen2, run bidirectionally) → proj_out`. This is the tensor
  that enters `cap_embedder`.
* `direct_context` `[N_text, 3840]` — `proj_directvlm(cat(hidden[5], hidden[12], hidden[20])[:, :N_text])`,
  i.e. per-prompt-token "shallow" VLM features, `N_text` = prompt+template tokens
  (everything before the `<image>` marker). Bypasses `cap_embedder`.

Caption stream layout in the DiT is therefore
`[cap(256) , direct(N) , pad → multiple of 32]`, caption RoPE positions `1..len_padded`, and
the image `t`-axis starts at `round_up(len(cap), 32) + 1` — exactly the Z-Image rule.

### 1.3 Text encoder — BailingMM2 "Ling-mini-2.0" (18.34 B params, 35 GB bf16)

| Part | Params | Notes |
|---|---|---|
| `thinker.*` (`BailingMoeV2`) | 16.27 B | 20 layers, hidden 2048, fused `attention.query_key_value` `[3072,2048]` (16 q / 4 kv heads × 128, no bias), per-head `q_norm`/`k_norm`, **partial RoPE: first 64 of 128 head dims**, `rope_theta=6e5`, layer 0 dense, layers 1-19 MoE: **256 experts, top-8**, `sigmoid(proj) + expert_bias` routing, 8 groups → 4 selected, renorm ×2.5, plus 1 shared expert **and a second `image_gate` router** (unused for pure text, but its weights exist and must load) |
| `connector.*` | 1.31 B | Qwen2 blocks ×28, hidden 1536, 12:2 GQA, `intermediate 8960`, `qkv_bias=True`, rope θ=1e6, final `norm` |
| `vision.*` | 0.71 B | Qwen2.5-VL ViT + `linear_proj` — **edit path only, deferred** |
| `query_tokens`, `proj_in/out`, `proj_directvlm` | ~0.03 B | see §1.2 |
| `tokenizer_json` | 12 MB | **the tokenizer is a U8 tensor inside the checkpoint** → `tokenizers.Tokenizer.from_str` |

Thinker positions are **not** a plain `arange`: Ling's `video_rope` treats each embed block as
a `1×W` grid centred on its own index (`t` axis compressed per block, `h`/`w` filling the first
24 of 32 frequency pairs, `t` the remaining 8). For t2i the blocks are `text (1×T)` at index 0
and `query (1×256)`; the resulting table is small and deterministic — port
`BailingMoeV2.freqs_cis` verbatim and unit-test it (see §6).

Prompt template (no system message supplied → the vendor default):

```
<role>SYSTEM</role>你是一个友好的AI助手。\n\ndetailed thinking off<|role_end|><role>HUMAN</role>{prompt}<|role_end|><role>ASSISTANT</role><image><imagePatch></image>
```

pad `156892`, `imagePatch = 157157`; batch is always 1, so no padding is required
(causal attention ⇒ trailing pads cannot influence real tokens).

### 1.4 VAE — already in the repo

`vae/config.json` is the **same architecture** as our Qwen-Image VAE (Wan2.1 family:
`base_dim=96, dim_mult=[1,2,4,4], num_res_blocks=2, z_dim=16, temperal_downsample=[F,T,T]`,
8× spatial). The local header matches `thenoise/vae/qwen_image.py::convert_comfyui_state_dict`
key-for-key (`decoder.upsamples.0..14`, `encoder.downsamples.0..10`, `.shortcut`, `head.0.gamma`,
`head.2`). Two differences only:

* `input_channels = 4` → **RGBA in and out** (`encoder.conv1` in-channels 4,
  `decoder.head.2` out-channels 4). `pixel_channels = 4` is already part of our VAE interface
  (Qwen-Image 2.1 uses it) and the pipeline already emits RGBA PNGs.
* normalisation is a **scalar**: `canonical = raw * 8.0064`, `shift = 0` (vendor
  `scaling_factor`, ComfyUI `latent_formats.MingImage.scale_factor`), *not* the Qwen
  per-channel z-score.

### 1.5 Sampling

* `FlowMatchEulerDiscreteScheduler`, `sigmas = linspace(1, 1/N, N)` + trailing 0.
* The shipped `scheduler_config.json` says `shift: 6.0, use_dynamic_shifting: false`, but
  `generator.py` forces `use_dynamic_shifting = True`, so the **dynamic** path is what actually
  runs:
  `seq = (H//16)*(W//16)`, `max_seq = max(4096, seq)`, `max_shift = 1.35 if seq > 4096 else 1.15`,
  `mu = 0.5 + (max_shift-0.5)·(seq-256)/(max_seq-256)`, **effective shift = e^mu**
  (1024² → 3.158, 2048² → 3.857). ComfyUI hard-codes the 1024 value (3.16), which is why we
  must implement the resolution-dependent form rather than copy `zimage/sampling.py`'s constant.
* Defaults: **12 steps, CFG 1.0 (guidance off)**, BF16, buckets 1024 / 2048, sizes multiple
  of 16 (= VAE 8 × patch 2).
* The reference CFG formula is `p + s·(p − n)`, i.e. standard CFG at scale `1+s`; our adapter
  keeps the standard formula used by every other adapter, so `guidance_scale=1.0` means off ✓.

### 1.6 Checkpoint naming trap (verified by header dumps)

| Module | bf16 file | int8-convrot file | our module tree |
|---|---|---|---|
| patch embed | `all_x_embedder.2-1.*` | `x_embedder.*` | `x_embedder` |
| output head | `all_final_layer.2-1.*` | `final_layer.*` | `final_layer` |
| attention | `to_q/to_k/to_v`, `to_out.0` | `qkv`, `out` (int8 + `weight_scale` + `comfy_quant`) | `qkv` (fused), `out` |
| QK norm | `attention.norm_q/k` | `attention.q_norm/k_norm` | `qk_norm.query_norm/key_norm` |

⇒ **Keep the fused `qkv` module** (matches the int8 file, Z-Image, one GEMM, quant-friendly,
and `FUSE_QKV` already covers LoRAs). The bf16 file needs a load-time rename **plus** a
`to_q|to_k|to_v → qkv` concat; the int8 file needs the plain `qk_norm_key_map` path only.
`load_dit(..., key_map=..., value_map=...)` supports both, but the two maps are
mutually exclusive → choose them on `is_quantized_checkpoint(path)` inside `load_ming_dit`.

## 2. Target file layout

```
thenoise/dit/lumina/__init__.py          NEW  shared Lumina/S3-DiT core package
thenoise/dit/lumina/models.py            NEW  LuminaTransformer2DModel (+ TimestepEmbedder,
                                             Attention, FeedForward, TransformerBlock,
                                             FinalLayer, patchify/unpatchify, prepare_rope)
thenoise/dit/lumina/keys.py              NEW  the shared checkpoint key/value maps
thenoise/dit/zimage/models.py            MOD  thin subclass, pad_mode="learned"
thenoise/dit/ming_image/__init__.py      NEW
thenoise/dit/ming_image/models.py        NEW  MingImageTransformer2DModel + MING_IMAGE_DIT_CONFIG
thenoise/dit/ming_image/utils.py         NEW  load_ming_dit / load_ming_text_encoder
thenoise/dit/ming_image/text_encoder.py  NEW  BailingMoeV2 + MingConnector + MingImageEncoder
thenoise/dit/ming_image/moe.py           NEW  MoE routing + expert banks (bf16 & int8)
thenoise/dit/ming_image/sampling.py      NEW  get_sigmas(steps, height, width) dynamic shift
thenoise/models/ming_image.py            NEW  MingImageModel adapter (+ MingConditioning)
thenoise/models/__init__.py              MOD  append MingImageModel to MODEL_CATALOG
thenoise/vae/qwen_image.py               MOD  config-driven loader + scalar scale/shift
thenoise/utils/sequence.py               MOD  alignment-padding mask helper
thenoise/dit/quantized.py or dit/ming_image/moe.py  MOD/NEW  int8 3-D expert bank leaf
thenoise/upscale/{sesqui,inference_adaptors}.py     MOD  "ming" latent format (affine 8.0064)
thenoise/utils/loader.py                 MOD  drop/extract non-module tensors (tokenizer_json,
                                             thinker.lm_head) in load_text_encoder_weights
scripts/download.py                      OK   `--model ming-image` already exists
docs/models/ming-image.md                NEW
README.md                                MOD  supported-models row (+ perf line later)
pyproject.toml                           MOD  package list
tests/{conftest,test_ming_image,test_lumina,test_vae,test_schedules}.py  MOD/NEW
```

## 3. Phase 1 — extract the shared Lumina core (no behaviour change)

Move Z-Image's DiT verbatim into `thenoise/dit/lumina/models.py` and parameterise the parts
that differ. Design rules: keep the *data flow* identical (so numerics are unchanged), and keep
the model's own `forward`/`prepare_rope` signatures stable for callers.

```python
class LuminaTransformer2DModel(nn.Module):
    def __init__(self, *, patch_size=2, f_patch_size=1, in_channels=16, dim=3840,
                 n_layers=30, n_refiner_layers=2, n_heads=30, n_kv_heads=30,
                 norm_eps=1e-5, cap_feat_dim=2560, rope_theta=256.0,
                 axes_dims=(32, 48, 48), pad_mode="learned"):   # "learned" | "zero_masked"
        ...
        # pad_mode == "learned" -> x_pad_token / cap_pad_token parameters exist
        # pad_mode == "zero_masked" -> no parameters, pads are zeros and are masked out

    def prepare_rope(self, x, cap_feats, cap_extra=None, key="", clear=True): ...

    def forward(self, x, t, cap_feats, cap_extra=None, rope_key=""): ...
```

Three explicit knobs, nothing else:

1. **`pad_mode`**
   * `learned` (Z-Image): fill aligned pad slots with `x_pad_token`/`cap_pad_token`; mask =
     `make_key_padding_mask(padded_lens)` (pads attended — this is current behaviour).
   * `zero_masked` (Ming): fill pad slots with `0.0` (reuse `pad_to_batch(pad_token=zeros)`,
     which already does exactly this) **and** build the attention mask with holes:
     `[1]*x_valid + [0]*x_pad + [1]*cap_valid + [0]*cap_pad` per item, for the x-refiner,
     cap-refiner and unified sequences.
   Implementation: one new shared helper in `thenoise/utils/sequence.py`,
   `make_key_padding_mask(item_seqlens, device, *, inner_pad_masks=None, segment_lengths=None)`
   (or a sibling `alignment_padding_mask(...)`), used by both models — Z-Image passes nothing
   new and keeps its current output bit-for-bit.
2. **`cap_extra`** — optional per-sample list of `[N_i, dim]` tensors concatenated to the
   embedded caption *after* `cap_embedder`, before padding. `None` ⇒ today's behaviour.
   `cap_extra` participates in positions and in the pad/mask math as the tail of the caption
   block (matching the vendor's `cap_padding_len` applied to the *last* caption tensor).
3. **config** (`rope_theta`, `axes_dims`, `cap_feat_dim`, counts) — already constructor data.

Then:

* `dit/zimage/models.py` → `class ZImageTransformer2DModel(LuminaTransformer2DModel)` with
  `pad_mode="learned"`; keeps its `create_coordinate_grid`/`_patchify_image` doc-comment
  provenance and the Apache header from the Z-Image team.
* `dit/lumina/keys.py` → shared key helpers used by both `dit/zimage/utils.py` and
  `dit/ming_image/utils.py`: the QK-norm rename (already `qk_norm_key_map`, called with the
  Lumina legacy names `norm_q`/`norm_k`), the block renames
  (`attention.to_out.0 → attention.out`, `all_x_embedder.2-1 → x_embedder`,
  `all_final_layer.2-1 → final_layer`), and the `to_q/to_k/to_v → qkv` fusion `value_map`
  (concat on dim 0, order q,k,v).
* Acceptance: `.venv/bin/python -m pytest tests/ -q` green, and a Z-Image regression image from
  the maintainer is unchanged.

## 4. Phase 2 — Ming DiT, VAE, adapter (t2i, using cached embeddings)

* `dit/ming_image/models.py`: `MING_IMAGE_DIT_CONFIG` = §1.1 values, `pad_mode="zero_masked"`,
  `cap_extra` on. `detect`-relevant fact: **no `x_pad_token`/`cap_pad_token` in the file**.
* `dit/ming_image/utils.py::load_ming_dit(path, device, dtype)`:
  * meta-construct with `init_empty_weights()`, `load_dit(...)`.
  * quantized file → `key_map=partial(qk_norm_key_map, q_legacy="q_norm", k_legacy="k_norm")`,
    no value fusion.
  * bf16 file → the §1.6 rename map + qkv `value_map` fusion.
  * `expected_missing=()` — the file must match strictly; nothing is dropped.
* `dit/ming_image/sampling.py::get_sigmas(steps, height, width, device)` implementing §1.5;
  `percent_to_sigma` keeps the "nudge sigma below 1" behaviour for ER-SDE.
* VAE: replace `load_qwen_vae(path, device, input_channels=3)` with a config-driven loader
  (`base_dim/z_dim/dim_mult/num_res_blocks/input_channels` + **either** `latents_mean/std`
  **or** `scale_factor/shift_factor`), and `load_ming_vae(path, ...)` =
  `input_channels=4, scale_factor=8.0064, shift_factor=0.0`. Keep the existing ComfyUI→official
  key conversion and the 5D→2D fold untouched; infer `input_channels` from
  `encoder.conv1`/`decoder.head.2` (they agree, both 4 here) and fail loudly on disagreement.
* Adapter `thenoise/models/ming_image.py`:

```python
class MingConditioning(Conditioning):          # same pattern as QwenImage21Conditioning
    cond_extra: Optional[torch.Tensor] = None   # direct_context for `cond`
    null_extra: Optional[torch.Tensor] = None

class MingImageModel(DiffusionModel):
    name = "ming_image"
    DEFAULT_PREFS = {**DiffusionModel.DEFAULT_PREFS,
                     "steps": 12, "guidance_scale": 1.0, "sampler": "euler",
                     "width": 1024, "height": 1024}
    CAPABILITIES = {**DiffusionModel.CAPABILITIES}          # edit False for now
    lora_fusions = FUSE_QKV
```

  * `detect(f)`: normalized keys contain `cap_embedder.1.weight` **and** `context_refiner.0.*`
    and (`all_x_embedder.2-1.` or `x_embedder.`) and **not** `x_pad_token`
    (the Z-Image/Ming separator).
  * `encode_prompt` → `MingConditioning(cond=cap, cond_extra=direct)` (+ `null`/`null_extra`
    only when `guidance_scale > 1`); `fuse_text` stays identity.
  * `init_latents`: `[1, 16, H//8, W//8]`; `prepare_latent` adds the F axis and calls
    `dit.prepare_rope(latents, cap, cap_extra)` per branch (`key=""` / `key="_neg"`);
    `denoise_step` mirrors `zimage.denoise_step` with `cap_extra` threaded and `v = -out`;
    `finalize_latent` drops the F axis.
  * `resolve_size`: `round_up(w|h, vae.spatial_compression * dit.patch_size)` = 16.
  * `schedule`: `get_sigmas(steps, height, width)` → `Step(t=sigma_i, delta=sigma_i-sigma_{i+1})`.
  * `_create_upscaler`: `SesquiLSRUpscaler("ming", ...)` — new format entry
    `("ming": (make_ming, "upscaler_Wan21.safetensors", 16))` with
    `make_ming() = _AffineAdaptor(external_channels=16, scale=8.0064, shift=0.0)`.
    Reuses the committed Wan2.1 Sesqui weights over the (identical-family) 16-channel Wan/Qwen
    VAE latent; **untested for Ming (esp. RGBA) — we will test it** and can drop it again by
    returning the base behaviour if it is poor.
  * `_lora_key_map`: reuse the §1.6 table so diffusers/ComfyUI-named LoRAs
    (`to_q/to_k/to_v`, `norm_q/k`, `to_out.0`) resolve against the fused modules.
* Docs/README/pyproject/tests per §6, plus a `docs/models/ming-image.md` that states the
  1024/2048 buckets, the 12-step/no-CFG default, and the RGBA/transparent-background
  behaviour (prompt-side "transparent background" phrasing is a model-level trick — document
  it, and note our PNG path already writes RGBA).

## 5. Phase 3 — the text encoder (the bulk of the work)

Self-contained in `dit/ming_image/text_encoder.py` + `moe.py`, mirroring ComfyUI's condensed
encoder (which is itself a port of the vendor `modeling_bailingmm2.py`):

* **Tokenizer** — read the `tokenizer_json` U8 tensor out of the TE file (read it via
  `MemoryEfficientSafeOpen.get_tensor` before the weight load, and drop it from the module
  load, see §5.1), then `tokenizers.Tokenizer.from_str`. Expose
  `MingTokenizer.encode(prompt) -> (input_ids, template_len)`; the `imagePatch` token id is
  replaced by the 256 `query_tokens` rows at embed time.
* **`BailingMoeV2`** (20 layers): `RMSNorm`, fused `query_key_value`, per-head q/k RMSNorm,
  **partial RoPE (64/128, θ=6e5)** — extend/reuse `thenoise/utils/rope.py`
  (`apply_rope_split_half` on a 64-dim slice + pass-through of the remaining 64 dims), causal
  SDPA via `thenoise/utils/attention.py` (GQA expand is already handled), and
  `freqs_cis(seq_len, blocks)` ported exactly per §1.3 (returns `cos, sin, -sin` for the
  partial-rotary apply). Capture hidden states at layers 5, 12 and the post-norm output.
* **MoE** (`dit/ming_image/moe.py`): routing =
  `sigmoid(proj(x.float())) + expert_bias` → group top-k (8 groups, top-2 per group summed,
  4 groups, 8 experts) → weights = `gather(sigmoid)·1/(sum+1e-20)·2.5`; `image_gate` is the
  second router, selected per token by an image mask (all-False for t2i, but the module must
  exist so the weights load). Shared expert = ordinary SwiGLU MLP. Expert execution:
  *group by expert* (`sort`/`bincount` → one `index_select`/`index_add_` per occupied expert)
  with `torch._grouped_mm` used when available, falling back to the loop. For a ~450-token
  prompt this is ≤256 tiny GEMMs × 19 layers — fine for a once-per-prompt encode; measure and
  optimise only if it shows up.
* **`MingConnector`**: 28 Qwen2-style blocks (separate `q/k/v_proj` **with bias**, 12:2 GQA,
  SwiGLU 1536→8960), 1-D RoPE θ=1e6, **non-causal**, final RMSNorm.
* **`MingImageEncoder`**: embed_tokens → splice query tokens → thinker (capture) →
  `proj_out(connector(proj_in(hidden[:, q_start:q_start+256])))` +
  `proj_directvlm(cat(captures[:, :q_start-1]))`; returns `(cap_feats, direct_context)`.
  `vision`/`linear_proj` are **not built** in phase 3 (weights are absent from the module
  tree, so the loader must tolerate their absence — see §5.1).

Compute notes: keep everything bf16 including the expert banks (ComfyUI upcasts experts to fp32
for multimodal encoders; we can A/B that later — fp32 experts would need ~30 GB of transient
scratch, which Strix Halo can't spare alongside the 47 GB of weights).

### 5.1 Loader work the int8-convrot text encoder needs

Measured from the int8 TE header (1708 tensors): all `thinker.*`/`connector.*` linears are
`I8` + `F32 weight_scale` (row-wise) + `U8 comfy_quant`; the **expert banks are 3-D**:
`thinker.layers.N.mlp.experts.gate_up_proj.weight [256,1024,2048] I8` +
`weight_scale [256,1024,1] F32`; `vision.*` and the router `gate.proj`/`expert_bias` stay
bf16/f32; `tokenizer_json` is a U8 tensor. Two gaps to close, both generic (they'll serve the
next MoE model too):

1. **A 3-D quantized expert leaf.** `load_quantized_state_dict` currently routes low-bit
   weights to `module.load_quantized` and expects a 2-D `QuantizedLinear`;
   `comfy_kitchen.TensorWiseINT8Layout` has no 3-D grouped matmul. Add a small
   `QuantizedMoEExperts` module (in `dit/ming_image/moe.py`, or `dit/quantized.py` if we want
   it shared) that stores `qweight`/`weight_scale` as-is, implements `load_quantized`
   compatible with `load_quantized_state_dict`'s contract, and dequantizes **per occupied
   expert** into the compute dtype inside the grouped matmul — halving the memory traffic that
   actually dominates a MoE forward. `build_quantized_restore_map` / LoRA undo keep working
   (they key on `.weight` + `.weight_scale`, and LoRAs never target expert banks today).
2. **Non-module tensors and nested heads in `load_text_encoder_weights`.** It currently
   hard-drops only top-level `lm_head*`; the Ming TE needs `tokenizer_json` (a payload, not a
   parameter) and `thinker.lm_head.weight` (322 M params) excluded. Add an explicit
   `drop_keys: tuple[str, ...]` argument (prefix match, applied to both the bf16 and quantized
   paths) and have the Ming loader pass `("tokenizer_json", "thinker.lm_head.")`.

Everything else already matches the existing pattern (`replace_linears` on the meta-built
model + `load_text_encoder_weights(..., key_map=...)`), so the bf16/int8 choice is automatic,
"like the other models": `scripts/download.py --model ming-image [--int8-convrot]` already
fetches the right trio.

## 6. Tests (all runnable without weights / GPU)

* `tests/conftest.py`: `MODEL_KEYSETS["ming_image"]` (+ `_wrapped`) built from the **real** bf16
  header keys, and `MODEL_KEYSETS["ming_image_int8"]` from the int8 header keys
  (`attention.qkv.weight`, `q_norm`, `x_embedder.weight`) — both must resolve to
  `MingImageModel`, and neither may be claimed by `ZImageModel` (and vice versa).
  `test_detect.py`/`test_catalog.py` then cover detection, defaults and catalog order.
* `test_schedules.py`: dynamic-shift sigma grid — `shift == 3.158…` at 1024², `3.857…` at 2048²,
  monotone, `sigma[0] == 1.0`, trailing 0.
* `test_dit_utils.py`: the new alignment-padding mask (holes in the right places, uniform fast
  path preserved, Z-Image's mask unchanged).
* `test_ming_image.py`:
  * a **tiny** `MingImageTransformer2DModel` (dim 128, 2 layers, 2 heads, axes (32,48,48) →
    head_dim 128) built twice with both `pad_mode`s: shapes, pad-slot zeros, pad tokens masked,
    `cap_extra` changing the caption length and the image `t` offset;
  * the Z-Image subclass of the same size is pad-token-filled and attends to them;
  * key/value maps: a fake Lumina state dict round-trips into the fused layout
    (`qkv == cat(q,k,v)`, `qk_norm.query_norm`), and the quantized-naming map is identity for
    `qkv`;
  * tokenizer/template construction from an inline fake `tokenizer_json` payload;
  * MoE routing determinism + expert-coverage (all routed experts used, weights sum to 2.5),
    and the tiny expert bank matches a dense reference in bf16 and via the int8 leaf;
  * `get_sigmas`/`resolve_size`/`detect` unit checks.
* `test_vae.py`: Ming VAE loader from a synthetic tiny state dict (4 channels, scale 8.0064
  round-trip `encode → decode` normalisation check).
* `tests/test_pipeline.py`/`test_prefs.py` pick the new adapter up through the existing stub
  machinery (RGBA path is already covered for Qwen-Image 2.1).

## 7. Memory & performance expectations (gfx1151, 128 GB unified)

* Resident weights: TE 35 GB (bf16) or ~10 GB (int8) + DiT 12 GB (bf16) / ~4 GB (int8) +
  VAE 0.24 GB. bf16 total ≈ 47 GB, which is inside the current
  `_RESIDENT_VRAM_FRACTION = 0.6` auto-detect budget on Strix Halo (no per-request moves);
  on smaller GPUs the TE will offload per prompt, which is brutally slow — document
  `--offload-device`/`--int8-convrot` in `docs/models/ming-image.md`.
* Text encoding is new-load-order work per prompt (20 layers × 256 experts + 28 connector
  layers over 256+T tokens). One-off per prompt; the DiT loop is otherwise the usual
  `2 refiners + 30 blocks` on `seq = 4096 (1024²) + ~512` caption tokens, i.e. in Z-Image's
  class at 1024² but ~12 steps instead of 8 → expect roughly Z-Image-Turbo × 1.4 per image,
  more at 2048 (4× the image tokens, plus shift 3.86).
* RoPE caches are per-prompt like Z-Image's (`prepare_rope` + `RopeCache`), so the DiT loop
  rebuilds nothing per step.

## 8. Risks / watch-list

1. **Thinker position ids** (`video_rope` block centring) are the easiest thing to get subtly
   wrong and it silently degrades output — hence the dedicated unit test and a golden-prompt
   A/B against ComfyUI at 1024², seed-fixed.
2. **MoE routing details** (sigmoid+bias, group selection, `1e-20` renorm, `routed_scaling_factor`,
   `first_k_dense_replace=1`) must match exactly; test against a dense fp64 reference on random
   weights.
3. **bf16 vs fp32 experts** (ComfyUI runs the multimodal encoder in fp32) — if output quality
   looks off, that's suspect #1; a per-layer fp32 upcast switch is cheap to add.
4. **`zero_masked` vs `learned` padding** — if we mask too much/too little the model soft-fails
   (wash-out / detail loss); the unified-sequence mask holes are the specific risk (§3.1).
5. **Int8 3-D expert banks** need real-hardware validation of the dequant-per-expert path
   (bandwidth, dtype handling) and are the one place where `load_quantized_state_dict`'s
   2-D assumption must be generalised.
6. **Sesqui refine on Ming latents/RGBA** is borrowed from Wan2.1 — wired through as agreed,
   pending measurement.
7. **Dynamic shift** (§1.5): if we ever copy ComfyUI's static 3.16, contrast/structure changes
   at 2048 — keep the resolution-dependent form.

## 9. Sequencing

1. **P1 — done.** Shared `dit/lumina` core + Z-Image rewire + padding-mask helper + tests →
   no visible change (maintainer regression check still open).
2. **P2 — done.** Ming DiT (bf16 *and* int8 load paths), Ming VAE loader, dynamic-shift
   schedule, adapter + catalog + docs + tests. Needs a real-hardware pass: bf16 + int8
   loads of the shipped files, and a `--dit` run reaching the `NotImplementedError`.
3. **P3** text encoder (tokenizer, thinker, MoE, connector) bf16 → int8 expert leaf →
   golden-prompt validation against ComfyUI.
4. **P4** polish: Sesqui `"ming"` format measurement, perf numbers for the README table,
   and (separately) the edit/`Design-Layer` capability
   (`vision` tower + `ref_frames` on the F axis + `edit`/`kv_cache` capabilities).
