"""Mage-Flow: patch-1 tokens, the static schedule, the unrotated text stream, the
edit path and the one-step codec.

Weight-free: a tiny DiT (2 heads x 128, so the released axis split stays), a narrow
stand-in for the codec, and synthetic checkpoints written out of those.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from conftest import FakeHandle, comfy_quant, write_safetensors
from thenoise.dit.mage_flow import sampling as mage_sampling
from thenoise.dit.mage_flow.encoder import (
    MAGE_EDIT_TEMPLATE,
    MAGE_T2I_TEMPLATE,
    VL_COND_LONG_EDGE,
    MageFlowTextEncoder,
    prompt_template,
)
from thenoise.dit.mage_flow.keys import MAGE_LAYERS, dit_block_count, is_qwen_image_family
from thenoise.dit.mage_flow.models import (
    MageFlowParams,
    MageFlowTransformer2DModel,
    MageTimestepProjEmbeddings,
    latent_to_tokens,
    load_mage_flow_dit,
    tokens_to_latent,
)
from thenoise.dit.mage_flow.utils import detect_params
from thenoise.dit.qwen_image.models import build_video_positions
from thenoise.models.base import Conditioning
from thenoise.models.config import EncodePromptArgs, SamplingParams
from thenoise.models.mage_flow import MageFlowModel
from thenoise.samplers import EulerSampler
from thenoise.upscale import VAEPixelUpscaler
from thenoise.utils.rope import apply_rope
from thenoise.utils.timestep import timestep_embedding
from thenoise.vae import mage_flow as mage_vae
from thenoise.vae.mage_flow import AutoencoderKLMageFlow, DConvEncoder, DConvDenoiser

# Two heads of the released 128, so the (16, 56, 56) axis split is the real one.
TINY = dict(in_channels=8, out_channels=8, num_layers=2, num_heads=2, head_dim=128, context_dim=96)
Z, CTX = TINY["in_channels"], TINY["context_dim"]


def _tiny_dit(dtype=torch.bfloat16) -> MageFlowTransformer2DModel:
    torch.manual_seed(0)
    return MageFlowTransformer2DModel(MageFlowParams(**TINY)).to(dtype).eval().requires_grad_(False)


def _bare(**attrs) -> MageFlowModel:
    """An adapter instance without ``__init__`` (no weights, no device moves)."""
    model = object.__new__(MageFlowModel)
    model.device = "cpu"
    model.dtype = torch.bfloat16
    for key, value in attrs.items():
        setattr(model, key, value)
    return model


class FakeVAE:
    """Latent geometry and a pixel round trip, with no codec behind it."""

    z_dim = Z
    spatial_compression = 16
    pixel_channels = 3
    dtype = torch.bfloat16
    device = torch.device("cpu")

    def encode_pixels_to_latents(self, pixels):
        cells = pixels[:, :1, :: self.spatial_compression, :: self.spatial_compression]
        return cells.repeat(1, self.z_dim, 1, 1)

    def decode_to_pixels(self, latents):
        pixels = latents[:, : self.pixel_channels]
        return pixels.repeat_interleave(self.spatial_compression, -2).repeat_interleave(
            self.spatial_compression, -1
        )


def _params(**overrides) -> SamplingParams:
    base = dict(height=64, width=48, steps=2, seed=0, guidance_scale=1.0, sampler="euler")
    base.update(overrides)
    return SamplingParams(**base)


# ------------------------------------------------------------- family detection


def test_block_count_is_the_highest_index_plus_one():
    keys = ["img_in.weight", "transformer_blocks.0.attn.to_q.weight", "proj_out.weight"]
    assert dit_block_count(keys) == 1
    assert dit_block_count(keys + [f"transformer_blocks.{MAGE_LAYERS - 1}.attn.to_q.weight"]) == MAGE_LAYERS
    assert dit_block_count(keys + ["transformer_blocks.59.attn.to_q.weight"]) == 60
    assert dit_block_count(["x_embedder.weight"]) == 0
    assert dit_block_count(["transformer_blocks.ln.weight"]) == 0  # not an index


def test_names_are_the_family_and_depth_is_the_separator():
    names = [
        "img_in.weight",
        "txt_in.weight",
        "time_text_embed.timestep_embedder.linear_1.weight",
        "transformer_blocks.0.attn.add_q_proj.weight",
        "transformer_blocks.0.img_mlp.net.2.weight",
        "norm_out.linear.weight",
        "proj_out.weight",
    ]
    assert is_qwen_image_family(names)
    assert dit_block_count(names) < MAGE_LAYERS  # names alone cannot tell the members apart

    # The single-stream Qwen-Image 2.1 shares the projections but has no text stream.
    assert not is_qwen_image_family([n for n in names if "add_q_proj" not in n])

    names.append(f"transformer_blocks.{MAGE_LAYERS - 1}.attn.to_q.weight")
    assert MageFlowModel.detect(FakeHandle(names)) is True


# ---------------------------------------------------------------- token layout


def test_a_token_is_one_latent_cell():
    torch.manual_seed(0)
    latent = torch.randn(1, Z, 4, 3)
    tokens = latent_to_tokens(latent)

    assert tokens.shape == (1, 12, Z)  # one token per cell, width stays the channel count
    assert torch.equal(tokens[0, 5], latent[0, :, 1, 2])
    assert torch.equal(tokens_to_latent(tokens, 4, 3), latent)


# ---------------------------------------------------------------- schedule


@pytest.mark.parametrize("steps", [1, 4, 20])
def test_sigmas_are_the_static_shift_of_a_uniform_grid(steps):
    """``FlowMatchEulerDiscreteScheduler(shift=6)`` with dynamic shifting off."""
    shift = mage_sampling.SHIFT
    grid = torch.linspace(1.0, 1.0 / steps, steps)
    for sigma, shifted in zip(grid, mage_sampling.get_sigmas(steps)):
        expected = shift * float(sigma) / (1 + (shift - 1) * float(sigma))
        assert float(shifted) == pytest.approx(expected, rel=1e-6)


def test_schedule_is_the_grid_plus_a_terminal_zero():
    ts = mage_sampling.get_schedule(4)
    assert len(ts) == 5
    assert float(ts[0]) == 1.0 and float(ts[-1]) == 0.0
    assert all(float(a) > float(b) for a, b in zip(ts, ts[1:]))
    # A shift above 1 holds every point of the grid above the unshifted one (both
    # start at pure noise).
    grid = torch.linspace(1.0, 0.25, 4)
    assert all(float(t) > float(sigma) for t, sigma in zip(ts[1:], grid[1:]))


# ------------------------------------------------------------ timestep embedding


def _raw_timestep_features(t: float) -> torch.Tensor:
    embedder = MageTimestepProjEmbeddings(64)
    embedder.timestep_embedder = torch.nn.Identity()
    return embedder(torch.tensor([t]), torch.zeros(1, 4))


def test_timestep_features_are_cos_then_sin():
    raw = _raw_timestep_features(0.0)
    assert raw.shape == (1, 256)
    assert torch.equal(raw[:, :128], torch.ones(1, 128))
    assert torch.equal(raw[:, 128:], torch.zeros(1, 128))


@pytest.mark.parametrize("t", [0.25, 0.5, 1.0])
def test_the_timestep_frequency_table_is_rounded_to_bf16(t):
    """The table is rounded before the x1000 time factor: ~0.6 rad at the top end."""
    freqs = torch.exp(-math.log(10000) * torch.arange(128, dtype=torch.float32) / 128)
    angles = 1000 * t * freqs.to(torch.bfloat16).float()[None, :]
    assert torch.equal(_raw_timestep_features(t), torch.cat([angles.cos(), angles.sin()], -1))

    # The shared fp32 helper, at the same scaled timestep, lands elsewhere.
    unrounded = timestep_embedding(torch.tensor([t * 1000.0]), 256, time_factor=1.0)
    assert (_raw_timestep_features(t) - unrounded).abs().max() > 0.1


# ------------------------------------------------------------------------- DiT


@pytest.fixture
def dit():
    return _tiny_dit()


def _store_positions(dit, target=(4, 3), refs=(), dtype=torch.bfloat16):
    dit.pe_embedder.clear()
    pe = build_video_positions([(1, *target)] + [(1, *ref) for ref in refs], device="cpu")
    dit.pe_embedder.store("img", pe, dtype=dtype)
    dit.pe_embedder.store("txt", torch.zeros(1, 1, 3), dtype=dtype)


def _forward(dit, tokens, ctx, refs=None, dtype=torch.bfloat16):
    with torch.no_grad():
        return dit(
            hidden_states=tokens.to(dtype),
            encoder_hidden_states=ctx.to(dtype),
            timestep=torch.tensor([0.5], dtype=dtype),
            img_pe=dit.pe_embedder["img"],
            txt_pe=dit.pe_embedder["txt"],
            ref_tokens=None if refs is None else refs.to(dtype),
        )


def test_forward_returns_only_the_target_tokens(dit):
    tokens, ctx = torch.randn(1, 12, Z), torch.randn(1, 5, CTX)
    _store_positions(dit)
    t2i = _forward(dit, tokens, ctx)
    assert t2i.shape == (1, 12, Z)
    assert torch.isfinite(t2i.float()).all()

    _store_positions(dit, refs=[(2, 3)])
    edit = _forward(dit, tokens, ctx, torch.randn(1, 6, Z))
    assert edit.shape == t2i.shape
    assert not torch.allclose(edit.float(), t2i.float(), atol=1e-2)  # the references are read


def test_the_text_stream_is_not_rotated(dit):
    """One identity row broadcast over the prompt: q/k come back unchanged."""
    _store_positions(dit)
    txt_pe = dit.pe_embedder["txt"]
    assert txt_pe.shape[1] == 1

    q = torch.randn(1, 2, 5, TINY["head_dim"])
    rotated, _ = apply_rope(q, q, txt_pe.float())
    assert torch.equal(rotated, q)


def _as_int8(state_dict) -> dict:
    out = {}
    for key, value in state_dict.items():
        if value.dim() != 2:
            out[key] = value
            continue
        scale = (value.float().abs().amax(dim=1, keepdim=True) / 127).clamp_min(1e-8)
        out[key] = (value.float() / scale).round().clamp(-127, 127).to(torch.int8)
        out[f"{key}_scale"] = scale.float()
        out[key[: -len("weight")] + "comfy_quant"] = comfy_quant(convrot=False)
    return out


def test_load_needs_no_key_remapping(tmp_path):
    source = {k: v.clone() for k, v in _tiny_dit(torch.float32).state_dict().items()}
    path = write_safetensors(tmp_path / "mage_bf16.safetensors", source)

    dit = load_mage_flow_dit(path, MageFlowParams(**TINY), device="cpu", dtype=torch.bfloat16)
    assert set(dit.state_dict()) == set(source)
    assert torch.equal(dit.proj_out.weight, source["proj_out.weight"].to(torch.bfloat16))

    partial = {k: v for k, v in source.items() if k != "proj_out.bias"}
    with pytest.raises(RuntimeError, match="[Mm]issing"):
        load_mage_flow_dit(
            write_safetensors(tmp_path / "partial.safetensors", partial),
            MageFlowParams(**TINY), device="cpu", dtype=torch.float32,
        )


def test_load_reconstructs_the_int8_export(tmp_path):
    source = {k: v.clone() for k, v in _tiny_dit(torch.float32).state_dict().items()}
    path = write_safetensors(tmp_path / "mage_int8.safetensors", _as_int8(source))

    dit = load_mage_flow_dit(path, MageFlowParams(**TINY), device="cpu", dtype=torch.bfloat16)

    assert hasattr(dit.img_in.weight, "dequantize"), "img_in did not land quantized"
    step = float(source["img_in.weight"].abs().max()) / 127
    assert (dit.img_in.weight.dequantize().float() - source["img_in.weight"]).abs().max() <= step * 2
    assert "img_in" in dit._quantized_restore_map

    _store_positions(dit)
    assert torch.isfinite(_forward(dit, torch.randn(1, 12, Z), torch.randn(1, 5, CTX)).float()).all()


def test_detect_params_reads_the_geometry_from_the_header(tmp_path):
    path = write_safetensors(
        tmp_path / "mage_header.safetensors",
        {k: v.clone() for k, v in _tiny_dit(torch.float32).state_dict().items()},
    )
    assert detect_params(path) == MageFlowParams(
        in_channels=Z,
        out_channels=TINY["out_channels"],
        num_layers=TINY["num_layers"],
        num_heads=TINY["num_heads"],
        head_dim=TINY["head_dim"],
        context_dim=CTX,
    )


def test_detect_params_refuses_a_foreign_checkpoint(tmp_path):
    path = write_safetensors(tmp_path / "other.safetensors", {"img_in.weight": torch.zeros(4, 4)})
    with pytest.raises(ValueError, match="not a Mage-Flow DiT"):
        detect_params(path)


# --------------------------------------------------------------------- adapter


def test_init_latents_uses_the_vae_geometry_and_the_seed():
    model = _bare(vae=FakeVAE())
    latents = model.init_latents(_params(seed=7))

    assert latents.shape == (1, Z, 4, 3)  # one latent cell per 16 pixels
    assert torch.equal(latents, model.init_latents(_params(seed=7)))
    assert not torch.equal(latents, model.init_latents(_params(seed=8)))


def test_prepare_latent_stashes_tokens_positions_and_text(dit):
    model = _bare(dit=dit, vae=FakeVAE())
    cond = Conditioning(cond=torch.randn(1, 5, CTX), null=torch.randn(1, 3, CTX))
    tokens = model.prepare_latent(torch.randn(1, Z, 4, 3), cond, _params())

    assert tokens.shape == (1, 12, Z)
    assert model._txt.shape == (1, 5, CTX)
    assert model._null_txt.shape == (1, 3, CTX)
    assert model._ref_tokens is None
    assert dit.pe_embedder["img"].shape[1] == 12  # the target only
    assert dit.pe_embedder["txt"].shape[1] == 1


def test_prepare_latent_keeps_the_references_in_the_image_stream(dit):
    """No KV cache: the references are never separated from the target again."""
    model = _bare(dit=dit, vae=FakeVAE())
    seen = []

    def pack(latents, method, ref_index):
        seen.append((method, ref_index))
        return MageFlowModel.pack_reference_latent(model, latents, method, ref_index)

    model.pack_reference_latent = pack
    refs = [torch.randn(1, Z, 2, 3), torch.randn(1, Z, 2, 2)]
    tokens = model.prepare_latent(torch.randn(1, Z, 4, 3), Conditioning(cond=torch.randn(1, 5, CTX)), _params(), ref=refs)

    assert tokens.shape == (1, 12, Z)
    assert model._ref_tokens.shape == (1, 10, Z)  # both references, one stream
    assert dit.pe_embedder["img"].shape[1] == 12 + 10  # target then references
    assert seen == [("index", 1), ("index", 2)]  # 1-based: 0 is the target


def test_reference_positions_put_the_reference_on_its_own_frame_index():
    tokens, pe = _bare().pack_reference_latent(torch.randn(1, Z, 2, 3), ref_index=2)

    assert tokens.shape == (1, 6, Z)
    assert torch.equal(pe[..., 0], torch.full((1, 6), 2.0))  # the plain index, no scale
    assert torch.equal(pe[0, :, 1], torch.tensor([-1.0, -1.0, -1.0, 0.0, 0.0, 0.0]))
    assert torch.equal(pe[0, :, 2], torch.tensor([-2.0, -1.0, 0.0, -2.0, -1.0, 0.0]))


@pytest.mark.parametrize("method", ["index_timestep_zero", "crop"])
def test_index_is_the_only_reference_method(method):
    with pytest.raises(ValueError, match="unsupported ref_latents_method"):
        _bare().pack_reference_latent(torch.randn(1, Z, 2, 2), method=method)


def test_encode_reference_encodes_pixels_at_full_size():
    encoded = _bare(vae=FakeVAE()).encode_reference(torch.randn(3, 32, 48))
    assert encoded.shape == (1, Z, 2, 3)


def test_reference_images_reach_both_conditioning_branches():
    seen = []

    def text_encoder(prompt, images=None):
        seen.append((prompt, images))
        return torch.zeros(1, 4, CTX), torch.ones(1, 4)

    model = _bare(text_encoder=text_encoder)
    images = [Image.new("RGB", (8, 8))]
    cond = model.encode_prompt(
        EncodePromptArgs(prompt="cat", negative_prompt="blur", guidance_scale=3.0, image=images)
    )
    assert [prompt for prompt, _ in seen] == ["cat", "blur"]
    assert all(got is images for _, got in seen)  # the negative branch is image-conditioned too
    assert cond.null is not None

    seen.clear()
    plain = model.encode_prompt(EncodePromptArgs(prompt="cat", guidance_scale=1.0))
    assert len(seen) == 1 and seen[0][1] is None and plain.null is None


def test_denoise_step_feeds_the_shifted_sigma_and_does_plain_cfg(dit):
    model = _bare(dit=dit, vae=FakeVAE())
    cond = Conditioning(cond=torch.randn(1, 5, CTX), null=torch.randn(1, 3, CTX))
    x = model.prepare_latent(torch.randn(1, Z, 4, 3), cond, _params())

    seen, outputs = [], []
    real_forward = dit.forward

    def spy(*args, **kwargs):
        seen.append(kwargs)
        outputs.append(real_forward(*args, **kwargs))
        return outputs[-1]

    dit.forward = spy
    t = torch.tensor(0.75)
    with torch.no_grad():
        v = model.denoise_step(x, t, cond, 3.0, 0)

    assert len(seen) == 2
    assert torch.equal(seen[0]["timestep"], torch.full((1,), 0.75, dtype=torch.bfloat16))
    assert [s["encoder_hidden_states"].shape for s in seen] == [(1, 5, CTX), (1, 3, CTX)]
    assert seen[0]["hidden_states"] is seen[1]["hidden_states"]  # both branches, same x

    pos, neg = (out.float() for out in outputs)
    assert v.shape == x.shape
    # Unnormalised: exactly ``neg + scale * (pos - neg)``.
    assert torch.allclose(v.float(), neg + 3.0 * (pos - neg), rtol=2e-2, atol=1e-3)

    seen.clear(), outputs.clear()
    with torch.no_grad():
        plain = model.denoise_step(x, t, cond, 1.0, 0)
    assert len(outputs) == 1 and torch.equal(plain, outputs[0])


def test_the_euler_loop_keeps_the_token_layout(dit):
    model = _bare(dit=dit, vae=FakeVAE())
    params = _params(steps=3)
    cond = Conditioning(cond=torch.zeros(1, 5, CTX))
    x = model.prepare_latent(model.init_latents(params), cond, params)

    x = EulerSampler(model).sample(x, model.schedule(params), cond, 1.0, params.seed)
    assert x.shape == (1, 12, Z)

    latent = model.finalize_latent(x, params)
    assert latent.shape == (1, Z, 4, 3)
    assert torch.isfinite(latent.float()).all()


def test_sizes_round_up_to_the_pixel_cell_of_one_token():
    model = _bare(vae=FakeVAE(), dit=SimpleNamespace(patch_size=1))
    assert model._pixels_per_token == 16  # 16x VAE x patch 1
    assert model.resolve_size(100, 60) == (112, 64)
    assert model.resolve_size(512, 2048) == (512, 2048)  # no bucket, no clamp


def test_upscaling_is_the_weight_free_vae_round_trip():
    model = _bare(vae=FakeVAE(), _upscaler=None)
    upscaler = model.get_upscaler()
    assert isinstance(upscaler, VAEPixelUpscaler)
    assert upscaler.vae is model.vae  # the model's own VAE, not a second set of weights
    assert upscaler.scale == MageFlowModel.UPSCALE_SCALE


# ---------------------------------------------------------------- text encoder


def test_the_edit_template_labels_every_reference():
    assert prompt_template("a cat", 0) == MAGE_T2I_TEMPLATE.format("a cat")

    edit = prompt_template("a cat", 2)
    assert edit.startswith(MAGE_EDIT_TEMPLATE.split("{}")[0])
    assert edit.index("Image 1") < edit.index("Image 2") < edit.index("a cat")
    assert " " in prompt_template("", 1)  # an empty turn would retokenize the markers


class _FakeLM(torch.nn.Module):
    """A stand-in vision LM: row ``i`` of its output is token ``i``, and it records its input."""

    def __init__(self, length: int, hidden: int, dtype: torch.dtype):
        super().__init__()
        self.probe = torch.nn.Parameter(torch.zeros(1, dtype=dtype))
        self.seen = {}
        self.model = _FakeBody(self, torch.arange(length * hidden, dtype=torch.float32).reshape(1, length, hidden))


class _FakeBody:
    def __init__(self, lm, features):
        self.lm, self.features = lm, features

    def __call__(self, **inputs):
        self.lm.seen = inputs
        return SimpleNamespace(last_hidden_state=self.features)


def _fake_conditioner(length=20, hidden=6, dtype=torch.float32):
    """A conditioner with a stand-in LM and a lambda processor."""

    def processor(text, images=None):
        ids = [151644] + [7] * 9 + [151644] + [8] * 9  # system turn, then the user turn
        processor.text, processor.images = text[0], images
        inputs = {
            "input_ids": torch.tensor([ids]),
            "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
        }
        if images:
            inputs.update(
                pixel_values=torch.zeros(4, 8),
                image_grid_thw=torch.tensor([[1, 2, 2]]),
                mm_token_type_ids=torch.zeros(1, len(ids), dtype=torch.long),
            )
        return inputs

    processor.text = processor.images = None
    lm = _FakeLM(length, hidden, dtype)
    return MageFlowTextEncoder(lm, None, processor), lm, processor


def test_the_conditioner_keeps_the_user_turn_and_drops_the_template():
    encoder, lm, processor = _fake_conditioner()
    with torch.no_grad():
        embeds, mask = encoder.encode("a cat")

    assert processor.text == prompt_template("a cat", 0)
    assert processor.images is None and "pixel_values" not in lm.seen
    # Kept from three tokens past the second chat marker to the end of the sequence.
    assert embeds.shape == (1, 7, 6)
    assert torch.equal(embeds[0, 0], torch.arange(13 * 6, 14 * 6, dtype=torch.float32))
    assert torch.equal(mask, torch.ones(1, 7, dtype=torch.bool))


def test_the_conditioner_downsizes_the_reference_and_feeds_the_vision_tower():
    encoder, lm, processor = _fake_conditioner()
    with torch.no_grad():
        embeds, _ = encoder.encode("a cat", [Image.new("RGB", (1600, 800))])

    assert max(processor.images[0].size) == VL_COND_LONG_EDGE
    assert {"pixel_values", "image_grid_thw", "mm_token_type_ids"} <= set(lm.seen)
    assert lm.seen["pixel_values"].dtype == encoder.dtype
    assert embeds.shape[0] == 1


def test_a_bare_image_is_a_one_image_batch():
    encoder, _, processor = _fake_conditioner()
    with torch.no_grad():
        encoder.encode("a cat", Image.new("RGB", (100, 40)))
    assert processor.images[0].size == (100, 40)  # under the cap: not resized, not enlarged


# ------------------------------------------------------------------------- VAE


_REAL_VAE = AutoencoderKLMageFlow


def _tiny_vae():
    """The released geometry (128ch latent, 16x, 16px patches) at working widths."""
    vae = _REAL_VAE()
    vae.dconv_encoder = DConvEncoder(z_ch=128, hidden_size=32, num_blocks=1, head_size=32, num_head_blocks=1)
    vae.decoder_model = DConvDenoiser(hidden_size=32, hidden_size_x=8, num_blocks=2, num_cond_blocks=1, bottleneck_dim=128)
    return vae.eval().requires_grad_(False)


@pytest.fixture(scope="module")
def codec():
    return _tiny_vae()


def test_the_codec_is_128_channels_at_16x(codec):
    assert (codec.z_dim, codec.spatial_compression, codec.pixel_channels) == (128, 16, 3)
    latent = codec.encode_pixels_to_latents(torch.randn(1, 3, 64, 48))
    assert latent.shape == (1, 128, 4, 3)
    assert codec.decode_to_pixels(latent).shape == (1, 3, 64, 48)


def test_encode_is_the_posterior_mean_not_a_sample(codec):
    pixels = torch.randn(1, 3, 32, 32)
    assert torch.equal(codec.encode_pixels_to_latents(pixels), codec.encode_pixels_to_latents(pixels))


def test_decode_clamps(codec, monkeypatch):
    monkeypatch.setattr(
        codec.decoder_model, "forward", lambda x, t, cond: torch.full((1, 3, 32, 32), 4.0)
    )
    assert codec.decode_to_pixels(torch.zeros(1, 128, 2, 2)).unique().tolist() == [1.0]


@pytest.mark.parametrize("size", [(2, 2), (3, 5), (1, 7)])
def test_the_windowed_attention_pads_and_crops(codec, size):
    """The 32x32 attention windows cannot tile an odd map; it is padded and cropped."""
    h, w = size
    assert codec.decode_to_pixels(torch.randn(1, 128, h, w)).shape == (1, 3, h * 16, w * 16)
    assert codec.encode_pixels_to_latents(torch.randn(1, 3, h * 16, w * 16)).shape == (1, 128, h, w)


def _training_names(state_dict) -> dict:
    """The same tensors under the names the training artifact ships."""
    renamed = {}
    for key, value in state_dict.items():
        if key.startswith("dconv_encoder."):
            renamed[f"student.{key}"] = value
        else:
            renamed["pipeline." + key[len("decoder_model.") :]] = value
    return renamed


def test_load_remaps_the_training_names_and_drops_the_anchor(tmp_path, monkeypatch):
    source = {k: v.clone() for k, v in _tiny_vae().state_dict().items()}
    export = _training_names(source)
    export["pipeline.y_embedder.encoder.conv_in.weight"] = torch.zeros(4, 128, 3, 3)
    monkeypatch.setattr(mage_vae, "AutoencoderKLMageFlow", _tiny_vae)

    vae = mage_vae.load_mage_vae(
        write_safetensors(tmp_path / "mage_vae.safetensors", export), device="cpu", dtype=torch.bfloat16
    )

    state = vae.state_dict()
    assert set(state) == set(source)  # strict, and the anchor encoder was dropped
    assert not any("y_embedder.encoder" in key for key in state)
    assert state["decoder_model.y_embedder.decoder.conv_in.weight"].dtype == torch.bfloat16
    assert torch.allclose(
        state["dconv_encoder.proj_out.weight"].float(),
        source["dconv_encoder.proj_out.weight"].float(),
        atol=1e-2,
    )


def test_load_rejects_a_file_that_is_not_the_codec(tmp_path, monkeypatch):
    monkeypatch.setattr(mage_vae, "AutoencoderKLMageFlow", _tiny_vae)

    # A key outside the two branches is not this codec, and is not quietly dropped.
    mixed = {"pipeline.decoder_model.conv_in.weight": torch.zeros(4), "img_in.weight": torch.zeros(4)}
    with pytest.raises(ValueError, match="unexpected Mage-VAE key"):
        mage_vae.load_mage_vae(write_safetensors(tmp_path / "mixed.safetensors", mixed), device="cpu")

    # The anchor encoder on its own leaves nothing behind.
    anchor = {"pipeline.y_embedder.encoder.conv_in.weight": torch.zeros(4)}
    with pytest.raises(ValueError, match="not a Mage-VAE"):
        mage_vae.load_mage_vae(write_safetensors(tmp_path / "anchor.safetensors", anchor), device="cpu")

    # Half a file is an error, not a zero-filled decoder.
    source = {k: v.clone() for k, v in _tiny_vae().state_dict().items()}
    partial = _training_names(source)
    del partial["student.dconv_encoder.proj_out.bias"]
    with pytest.raises(Exception, match="[Mm]issing"):
        mage_vae.load_mage_vae(write_safetensors(tmp_path / "half.safetensors", partial), device="cpu")
