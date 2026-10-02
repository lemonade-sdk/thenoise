"""Ming-Image: the DiT it builds, the checkpoints it loads, the schedule it samples at.

Weight-free and GPU-free: a tiny DiT (dim 256, 2 heads of 128 = the real 32+48+48
axis split) stands in for the 12 GB one, and synthetic checkpoints for both exports.
"""
from __future__ import annotations

import math

import pytest
import torch

from accelerate import init_empty_weights

from conftest import comfy_quant, write_safetensors
from thenoise.dit.lumina.models import LuminaTransformerBlock
from thenoise.dit.ming_image.models import MING_IMAGE_DIT_CONFIG, MingImageTransformer2DModel
from thenoise.dit.ming_image.sampling import (
    dynamic_mu,
    dynamic_shift,
    get_sigmas,
    image_seq_len,
)
from thenoise.dit.ming_image.utils import load_ming_dit
from thenoise.models.config import EncodePromptArgs, SamplingParams
from thenoise.models.ming_image import MingConditioning, MingImageModel

# dim 256 / 2 heads -> head_dim 128, exactly what the (32, 48, 48) axis split needs.
TINY_CONFIG = dict(
    patch_size=2,
    f_patch_size=1,
    in_channels=16,
    dim=256,
    n_layers=2,
    n_refiner_layers=1,
    n_heads=2,
    n_kv_heads=2,
    norm_eps=1e-5,
    cap_feat_dim=96,
    rope_theta=256.0,
    axes_dims=(32, 48, 48),
)

CAP_DIM = TINY_CONFIG["cap_feat_dim"]
DIM = TINY_CONFIG["dim"]

# The projections the released int8 export keeps in BF16 (measured from its header):
# everything the attention and the feed-forward are not.
_NOT_QUANTIZED_IN_THE_FILE = ("cap_embedder.", "x_embedder.", "final_layer.", "t_embedder.")


@pytest.fixture(autouse=True)
def _blocks_run_eagerly(monkeypatch):
    """Call the transformer blocks un-compiled (see ``tests/test_lumina.py``)."""
    monkeypatch.setattr(
        LuminaTransformerBlock,
        "forward",
        LuminaTransformerBlock.forward._torchdynamo_orig_callable,
        raising=False,
    )


def _tiny_dit(dtype: torch.dtype = torch.float32) -> MingImageTransformer2DModel:
    torch.manual_seed(0)
    model = MingImageTransformer2DModel(**TINY_CONFIG)
    return model.to(dtype).eval().requires_grad_(False)


def _bare_model(**attrs) -> MingImageModel:
    """An adapter instance without ``__init__`` (no weights, no device moves)."""
    model = object.__new__(MingImageModel)
    model.device = "cpu"
    model.dtype = torch.float32
    for key, value in attrs.items():
        setattr(model, key, value)
    return model


def _params(**overrides) -> SamplingParams:
    base = dict(
        height=32, width=32, steps=2, seed=0, guidance_scale=1.0, sampler="euler"
    )
    base.update(overrides)
    return SamplingParams(**base)


# ---------------------------------------------------------------- the DiT itself


def test_ming_pads_with_masked_zeros_and_ships_no_pad_tokens():
    """A ``learned`` default would make the released file unloadable: it has no pad
    tokens to fill, and padding it attends to would be garbage.
    """
    dit = _tiny_dit()

    assert dit.pad_mode == "zero_masked"
    assert dit.x_pad_token is None and dit.cap_pad_token is None
    assert not {"x_pad_token", "cap_pad_token"} & set(dit.state_dict())
    # The subclass sets a default, not a cage.
    assert (
        MingImageTransformer2DModel(pad_mode="learned", **TINY_CONFIG).pad_mode
        == "learned"
    )


def test_released_config_builds_a_consistent_dit():
    """The part of the measured config a wrong edit can hide: the head/axis split and
    the patch-embedder width stay consistent, and the whole config still constructs.
    """
    cfg = MING_IMAGE_DIT_CONFIG
    assert cfg["dim"] // cfg["n_heads"] == sum(cfg["axes_dims"]) == 128
    assert cfg["patch_size"] ** 2 * cfg["f_patch_size"] * cfg["in_channels"] == 64
    with init_empty_weights():
        MingImageTransformer2DModel(**cfg)


# ------------------------------------------------------------- the two checkpoints


def _as_legacy_lumina_names(state_dict: dict) -> dict:
    """Rename a fused state dict back to the bf16 release's legacy spelling."""
    out = {}
    for key, value in state_dict.items():
        if key.endswith("attention.qkv.weight"):
            head = key[: -len("qkv.weight")]
            rows = value.shape[0] // 3
            for i, part in enumerate(("to_q", "to_k", "to_v")):
                out[f"{head}{part}.weight"] = value[i * rows : (i + 1) * rows]
        else:
            out[key] = value
    return {
        k.replace("x_embedder.", "all_x_embedder.2-1.")
        .replace("final_layer.", "all_final_layer.2-1.")
        .replace("attention.out.", "attention.to_out.0.")
        .replace("qk_norm.query_norm.", "norm_q.")
        .replace("qk_norm.key_norm.", "norm_k."): v
        for k, v in out.items()
    }


#: U8 JSON payload a ComfyUI export writes for its attention helper: not module
#: state, so a strict load only survives if it is dropped.
COMFY_RUNTIME_PAYLOAD = torch.tensor(
    list(b'{"skip_output_projection": false}'), dtype=torch.uint8
)


def test_load_ming_dit_reads_the_legacy_bf16_export(tmp_path):
    """Legacy names AND the split attention land on the fused tree, strictly. The
    state dict is built FROM the model, so only the naming and the fold are under test.
    """
    ref = _tiny_dit()
    sd = {k: v.detach().clone() for k, v in ref.state_dict().items()}
    legacy = _as_legacy_lumina_names(sd)
    legacy["layers.0.attention.comfy_attention.config"] = COMFY_RUNTIME_PAYLOAD
    assert any(".to_q.weight" in k for k in legacy)  # the fold had something to do
    path = write_safetensors(tmp_path / "ming_bf16.safetensors", legacy)

    dit = load_ming_dit(path, device="cpu", dtype=torch.float32, config=TINY_CONFIG)

    loaded = dit.state_dict()
    assert set(loaded) == set(sd)
    for key, value in sd.items():
        assert torch.equal(loaded[key], value), key
    # One fused parameter, holding q, k, v in that row order.
    qkv = loaded["layers.0.attention.qkv.weight"]
    rows = qkv.shape[0] // 3
    assert torch.equal(qkv[:rows], legacy["layers.0.attention.to_q.weight"])
    assert torch.equal(
        qkv[rows:],
        torch.cat([legacy["layers.0.attention.to_k.weight"],
                   legacy["layers.0.attention.to_v.weight"]]),
    )


def _as_int8_convrot(state_dict: dict) -> dict:
    """Re-export a state dict the way the int8-convrot file does: quantized
    projections become I8 + a per-row F32 ``weight_scale`` + a U8 ``comfy_quant``.
    """
    out = {}
    for key, value in state_dict.items():
        if value.dim() != 2 or key.startswith(_NOT_QUANTIZED_IN_THE_FILE):
            out[key] = value
            continue
        scale = (value.float().abs().amax(dim=1, keepdim=True) / 127).clamp_min(1e-8)
        out[key] = (value.float() / scale).round().clamp(-127, 127).to(torch.int8)
        out[f"{key}_scale"] = scale.to(torch.float32)
        # convrot=False: these are plain per-row quantized, not rotated.
        out[key[: -len("weight")] + "comfy_quant"] = comfy_quant(convrot=False)
    out["layers.0.attention.comfy_attention.config"] = COMFY_RUNTIME_PAYLOAD
    return out


def test_load_ming_dit_reads_the_int8_convrot_export(tmp_path):
    """The already-fused, partly-int8 file loads with no folding, the QK norms
    renamed, and the LoRA-undo restore map keyed on the renamed modules.
    """
    ref = _tiny_dit()
    sd = {k: v.detach().clone() for k, v in ref.state_dict().items()}
    path = write_safetensors(tmp_path / "ming_int8.safetensors", _as_int8_convrot(sd))

    dit = load_ming_dit(path, device="cpu", dtype=torch.bfloat16, config=TINY_CONFIG)
    dit.eval().requires_grad_(False)

    qkv = dit.layers[0].attention.qkv.weight
    assert hasattr(qkv, "dequantize"), "the qkv weight did not land quantized"
    want = sd["layers.0.attention.qkv.weight"]
    abs_max = float(want.abs().max())
    # One int8 step at the widest row, plus the BF16 the layout emits in.
    tolerance = abs_max / 127 + abs_max * 2**-8
    assert (qkv.dequantize().float() - want).abs().max() <= tolerance

    # What the file keeps in full precision is assigned, not rounded twice.
    assert torch.equal(
        dit.cap_embedder[1].weight, sd["cap_embedder.1.weight"].to(torch.bfloat16)
    )
    restore = dit._quantized_restore_map
    assert "layers.0.attention.qkv" in restore
    assert not any("comfy_attention" in key for key in restore)

    x = torch.zeros(16, 1, 8, 8, dtype=torch.bfloat16)
    cap = torch.zeros(20, CAP_DIM, dtype=torch.bfloat16)
    dit.prepare_rope([x], [cap])
    out = dit([x], torch.tensor([0.5], dtype=torch.bfloat16), [cap])[0]
    assert out.shape == (16, 1, 8, 8)
    assert torch.isfinite(out).all()


def test_load_ming_dit_is_strict(tmp_path):
    """A half-file is an error: nothing is dropped, nothing silently zero-filled."""
    sd = {k: v.clone() for k, v in _tiny_dit().state_dict().items()}
    del sd["cap_embedder.1.bias"]
    path = write_safetensors(tmp_path / "ming_partial.safetensors", sd)

    with pytest.raises(RuntimeError, match="cap_embedder"):
        load_ming_dit(path, device="cpu", dtype=torch.float32, config=TINY_CONFIG)


# ------------------------------------------------------------------------ schedule


@pytest.mark.parametrize(
    "height,width,mu",
    [
        (512, 512, 0.63),  # 1024 tokens: interpolated between the two buckets
        (1024, 1024, 1.15),  # the 4096-token reference bucket
        (2048, 2048, 1.35),  # its ceiling
    ],
)
def test_dynamic_shift_matches_the_reference_buckets(height, width, mu):
    assert dynamic_mu(height, width) == pytest.approx(mu, rel=1e-6)
    assert dynamic_shift(height, width) == pytest.approx(math.exp(mu), rel=1e-6)


def test_image_seq_len_counts_2x2_patches_of_the_vae_latent():
    assert image_seq_len(1024, 1024) == 4096
    assert image_seq_len(2048, 2048) == 16384
    assert image_seq_len(1024, 512) == 2048


def test_the_shift_bucket_flips_above_the_reference_sequence():
    """The one place ``>=`` and ``>`` differ: at 1024² we take the 1.15 branch (shift
    3.158), matching ComfyUI's hard-coded ``{"shift": 3.16}`` for Ming-Image. A bigger
    image must hold the grid high for longer, never lower it."""
    assert image_seq_len(1024, 1024) == 4096
    assert dynamic_mu(1024, 1024) == pytest.approx(1.15)
    assert image_seq_len(1040, 1024) == 4160  # 65 * 64 tokens, just past the bucket
    assert dynamic_mu(1040, 1024) == pytest.approx(1.35)

    small = [float(s) for s in get_sigmas(12, 512, 512, torch.device("cpu"))[:12]]
    large = [float(s) for s in get_sigmas(12, 2048, 2048, torch.device("cpu"))[:12]]
    plain = [1.0 - i / 12 for i in range(12)]  # the unshifted linspace(1, 1/N, N)

    assert small != plain  # the shift really moves the curve
    assert small[0] == large[0] == 1.0
    assert all(h > l for h, l in zip(large[1:], small[1:]))


# ------------------------------------------------------------------------ adapter


def test_encode_prompt_says_what_is_missing():
    """Ming-Image editing is not implemented: an image must be refused, not ignored."""
    model = _bare_model()
    with pytest.raises(ValueError, match="editing is not implemented"):
        model.encode_prompt(
            EncodePromptArgs(prompt="a cat", guidance_scale=1.0, image=object())
        )


def test_encode_prompt_pairs_each_branch_with_its_second_tensor(monkeypatch):
    """Two tensors per branch, and the uncond branch is the conditioning zeroed out
    (not a second encode) — with no uncond branch at all when guidance is off."""
    model = _bare_model()
    cap = torch.zeros(1, 4, CAP_DIM)
    extra = torch.zeros(1, 3, DIM)
    monkeypatch.setattr(model, "_encode_prompt", lambda prompt: (cap, extra))

    cond = model.encode_prompt(EncodePromptArgs(prompt="a cat", guidance_scale=1.0))
    assert isinstance(cond, MingConditioning)
    assert cond.cond is cap and cond.cond_extra is extra
    assert cond.null is None and cond.null_extra is None

    both = model.encode_prompt(
        EncodePromptArgs(prompt="a cat", negative_prompt="blur", guidance_scale=3.0)
    )
    assert both.null is not cap and both.null_extra is not extra
    assert torch.equal(both.null, torch.zeros_like(cap))
    assert torch.equal(both.null_extra, torch.zeros_like(extra))


class _SpyVAE:
    z_dim = 16
    spatial_compression = 8
    pixel_channels = 4


def test_init_latents_uses_the_vae_geometry_and_the_seed():
    model = _bare_model(vae=_SpyVAE())
    params = _params(height=1024, width=512, seed=7)

    latents = model.init_latents(params)
    assert latents.shape == (1, 16, 128, 64)
    assert torch.equal(latents, model.init_latents(params))


def test_prepare_latent_adds_the_frame_axis_and_one_rope_per_branch(monkeypatch):
    """Each branch's positions depend on ITS caption length, so each gets a build —
    and the second must not ``clear``, or it throws the conditional branch away.
    """
    model = _bare_model(dit=_tiny_dit())
    prepared = []
    monkeypatch.setattr(
        model.dit, "prepare_rope", lambda *args, **kwargs: prepared.append((args, kwargs))
    )
    latents = torch.zeros(1, 16, 4, 4)
    cond = MingConditioning(
        cond=torch.zeros(1, 5, CAP_DIM),
        cond_extra=torch.zeros(1, 3, DIM),
        null=torch.zeros(1, 2, CAP_DIM),
        null_extra=torch.zeros(1, 1, DIM),
    )

    out = model.prepare_latent(latents, cond, _params(guidance_scale=3.0))

    assert out.shape == (1, 16, 1, 4, 4)  # the F axis the DiT wants
    assert len(prepared) == 2
    (x, cap, extra), kwargs = prepared[0]
    assert x[0].shape == (16, 1, 4, 4) and cap[0].shape[0] == 5
    assert extra[0].shape == (3, DIM)  # the DiT wants per-sample lists, not a batch
    assert kwargs == {}
    (_, neg, neg_extra), neg_kwargs = prepared[1]
    assert neg[0].shape[0] == 2 and neg_extra[0].shape[0] == 1
    assert neg_kwargs == {"key": "_neg", "clear": False}


def test_prepare_latent_prepares_once_without_a_second_tensor(monkeypatch):
    """``cap_extra`` is optional all the way through, exactly as in the core."""
    model = _bare_model(dit=_tiny_dit())
    prepared = []
    monkeypatch.setattr(
        model.dit, "prepare_rope", lambda *args, **kwargs: prepared.append(args)
    )

    model.prepare_latent(
        torch.zeros(1, 16, 4, 4),
        MingConditioning(cond=torch.zeros(1, 5, CAP_DIM)),
        _params(),
    )
    assert len(prepared) == 1 and prepared[0][2] is None


def test_denoise_step_negates_the_velocity_and_threads_the_second_tensor(monkeypatch):
    """``x + dt * model_out`` (``dt < 0``) is our ``x -= delta * v`` only for
    ``v = -model_out``, and each branch carries its own extra tensor and RoPE table.
    """
    dit = _tiny_dit()
    model = _bare_model(dit=dit)
    latents = torch.randn(1, 16, 1, 4, 4)
    cond = MingConditioning(
        cond=torch.randn(1, 33, CAP_DIM),
        cond_extra=torch.randn(1, 8, DIM),
        null=torch.randn(1, 9, CAP_DIM),
        null_extra=torch.randn(1, 2, DIM),
    )
    dit.prepare_rope([latents[0]], [cond.cond[0]], [cond.cond_extra[0]])
    dit.prepare_rope(
        [latents[0]], [cond.null[0]], [cond.null_extra[0]], key="_neg", clear=False
    )

    calls, outputs = [], []
    real_forward = dit.forward

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        outputs.append(real_forward(*args, **kwargs))
        return outputs[-1]

    monkeypatch.setattr(dit, "forward", spy)
    t = torch.tensor(0.75)

    v = model.denoise_step(latents, t, cond, guidance_scale=2.0, i=0)

    assert len(calls) == 2
    (x_pos, timestep, cap_pos, extra_pos), pos_kwargs = calls[0]
    (x_neg, _, cap_neg, extra_neg), neg_kwargs = calls[1]
    assert x_neg[0] is x_pos[0]  # both branches denoise the same noisy latent
    assert pos_kwargs == {} and neg_kwargs == {"rope_key": "_neg"}
    assert cap_pos[0].shape[0] == 33 and cap_neg[0].shape[0] == 9
    assert extra_pos[0].shape == (8, DIM) and extra_neg[0].shape == (2, DIM)
    assert float(timestep) == pytest.approx(0.25)  # the model timestep is 1 - sigma

    v_pos, v_neg = -outputs[0][0].unsqueeze(0), -outputs[1][0].unsqueeze(0)
    assert torch.allclose(v, v_neg + 2.0 * (v_pos - v_neg))

    # Guidance off: one forward, and the velocity really is the negated DiT output.
    calls.clear(), outputs.clear()
    plain = model.denoise_step(latents, t, cond, guidance_scale=1.0, i=0)
    assert len(calls) == 1
    assert torch.allclose(plain, -outputs[0][0].unsqueeze(0))


def test_denoise_step_without_a_second_or_uncond_branch():
    """A run with neither extra tensor nor negative prompt still just works."""
    model = _bare_model(dit=_tiny_dit())
    latents = torch.randn(1, 16, 1, 4, 4)
    cond = MingConditioning(cond=torch.randn(1, 32, CAP_DIM))
    model.dit.prepare_rope([latents[0]], [cond.cond[0]])

    v = model.denoise_step(latents, torch.tensor(0.5), cond, guidance_scale=1.0, i=0)
    assert v.shape == latents.shape
    assert torch.isfinite(v).all()


def test_finalize_latent_returns_the_canonical_4d_latent():
    latents = _bare_model().finalize_latent(torch.zeros(1, 16, 1, 4, 4), _params())
    assert latents.shape == (1, 16, 4, 4)


@pytest.mark.parametrize(
    "key,expected",
    [
        # A LoRA trained against the bf16 release (legacy names)...
        ("layers.0.attention.to_out.0.lora_up.weight", "layers.0.attention.out.lora_up.weight"),
        ("layers.0.attention.norm_q.weight", "layers.0.attention.qk_norm.query_norm.weight"),
        ("all_x_embedder.2-1.weight", "x_embedder.weight"),
        # ...or against this repo's fused tree: unchanged, both QK spellings.
        ("layers.0.attention.qkv.lora_up.weight", "layers.0.attention.qkv.lora_up.weight"),
        ("layers.0.attention.q_norm.weight", "layers.0.attention.qk_norm.query_norm.weight"),
    ],
)
def test_lora_keys_from_either_generation_resolve(key, expected):
    assert _bare_model()._lora_key_map(key) == expected
