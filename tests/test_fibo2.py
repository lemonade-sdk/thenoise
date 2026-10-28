"""Fibo 2.0: the six text bundles, the gist/image/reference layout, the two AdaLN rows
and the geometry read back off the checkpoint.

The catalog-wide contracts (detection matrix, schedule shape, defaults, capabilities,
size alignment) live in ``test_detect.py`` / ``test_schedules.py`` / ``test_catalog.py``.
"""
from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from conftest import comfy_quant, write_safetensors
from thenoise.dit.fibo2.encoder import (
    BUNDLE_LAYERS,
    BUNDLE_TAPS,
    NORM_EPS,
    NULL_CAPTION,
    TEMPLATE,
    VISION_LONG_EDGE,
    Fibo2TextEncoder,
    build_prompt,
    parse_prompt,
)
from thenoise.dit.fibo2.models import (
    GIST_FULL_LENGTH,
    GIST_MIN_TOKENS,
    Fibo2Params,
    Fibo2Transformer2DModel,
    gist_count,
    scale_tokens,
    sequence_positions,
)
from thenoise.dit.fibo2.sampling import SHIFT, get_sigmas, shift
from thenoise.dit.fibo2.sizes import TRAINED_AREA, trained_size
from thenoise.dit.kvcache import KVCache
from thenoise.dit.fibo2.utils import detect_params, load_fibo2_dit
from thenoise.models import Fibo2Model
from thenoise.models.base import Conditioning
from thenoise.models.config import EncodePromptArgs, SamplingParams
from thenoise.upscale import SesquiLSRUpscaler

# The released rope split covers a 128-wide head and nothing narrower, so the tiny
# model keeps the real head and trims the depth instead. 512 gist queries is the
# ceiling the Perceiver emits; the floor it emits is not negotiable either.
TINY = dict(
    dim=256,
    ffn_dim=128,
    num_heads=2,
    num_layers=3,
    num_refiner_layers=1,
    perceiver_layers=1,
    injection_layers=(1,),
    latent_channels=8,
    cap_dim=6,
    max_gist_tokens=512,
    adaln_dim=256,
    t_embed_dim=64,
    freq_dim=256,
)
D, Z = TINY["dim"], TINY["latent_channels"]
CTX = TINY["cap_dim"] * (1 + len(TINY["injection_layers"]))
GIST = TINY["max_gist_tokens"]


def _params_dict(**overrides) -> Fibo2Params:
    values = dict(TINY)
    values.update(overrides)
    return Fibo2Params(**values)


def _tiny_dit(dtype: torch.dtype = torch.bfloat16) -> Fibo2Transformer2DModel:
    torch.manual_seed(0)
    dit = Fibo2Transformer2DModel(_params_dict())
    # Two shipped weights a module built without a checkpoint leaves inert: open the
    # injection gate and give the Perceiver its queries, or there is nothing to read.
    with torch.no_grad():
        dit.layers[TINY["injection_layers"][0]].cross_attn_gate.fill_(0.5)
        dit.perceiver.latent_queries.normal_()
    return dit.to(dtype).eval().requires_grad_(False)


def _bare(**attrs) -> Fibo2Model:
    """An adapter instance without ``__init__`` (no weights, no device moves)."""
    model = object.__new__(Fibo2Model)
    model.device = "cpu"
    model.dtype = torch.bfloat16
    for key, value in attrs.items():
        setattr(model, key, value)
    return model


class FakeVAE:
    """Latent geometry with no codec behind it."""

    z_dim = Z
    spatial_compression = 16
    pixel_channels = 3
    dtype = torch.bfloat16

    def encode_pixels_to_latents(self, pixels):
        self.seen = pixels.shape
        return torch.zeros(1, Z, pixels.shape[-2] // 16, pixels.shape[-1] // 16)


def _params(**overrides) -> SamplingParams:
    base = dict(height=64, width=48, steps=2, seed=0, guidance_scale=1.0, sampler="euler")
    base.update(overrides)
    return SamplingParams(**base)


def _n(*shape, dtype=torch.bfloat16) -> torch.Tensor:
    """Activations arrive in the compute dtype — the adapter casts at the edges."""
    return torch.randn(*shape, dtype=torch.float32).to(dtype)


@contextlib.contextmanager
def _compute():
    """The adapter runs the DiT under autocast, like every other model here."""
    with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        yield


@pytest.fixture
def dit():
    return _tiny_dit()


@pytest.fixture(autouse=True)
def _blocks_run_eagerly(monkeypatch):
    """Call the blocks un-compiled: a ``fullgraph=True`` block raises rather than
    falling back when dynamo is off, which ``conftest`` disables suite-wide."""
    from thenoise.dit.fibo2.models import Block, PerceiverLayer

    for cls in (Block, PerceiverLayer):
        monkeypatch.setattr(
            cls, "forward", cls.forward._torchdynamo_orig_callable, raising=False
        )


def _run(dit, x, ctx, refs=(), timestep=0.7, dtype=torch.bfloat16, kv=None):
    """One forward over the layout ``x``/``refs`` imply, positions and gist included."""
    positions = sequence_positions(
        gist_count(ctx.shape[1], GIST), x.shape[-2], x.shape[-1],
        [(r.shape[-2], r.shape[-1]) for r in refs], GIST,
    )
    dit.pe_embedder.clear()
    dit.pe_embedder.store("pe", positions)
    with torch.no_grad():
        context = ctx.to(dtype)
        gist = dit.gist_tokens(context)
        return dit(
            x.to(dtype), torch.tensor([timestep], dtype=dtype), context, gist,
            dit.pe_embedder["pe"], [r.to(dtype) for r in refs], kv,
        )


# --------------------------------------------------------------------- gist count


def test_gist_count_is_a_floor_that_rises_in_whole_query_blocks():
    """256 queries cover a short prompt; past it the count climbs to the ceiling in
    blocks of 64, so the sequence length is a step function of the prompt."""
    assert gist_count(1, GIST) == GIST_MIN_TOKENS
    assert gist_count(GIST_MIN_TOKENS, GIST) == GIST_MIN_TOKENS
    assert gist_count(GIST_MIN_TOKENS + 1, GIST) == GIST_MIN_TOKENS + 64
    assert gist_count(GIST_FULL_LENGTH, GIST) == GIST


def test_gist_count_never_leaves_the_valid_range_and_never_shrinks():
    counts = [gist_count(length, GIST) for length in range(1, 3000)]
    assert all(GIST_MIN_TOKENS <= c <= GIST for c in counts)
    assert all(a <= b for a, b in zip(counts, counts[1:]))


# ---------------------------------------------------------------------- positions


def test_the_image_plane_does_not_move_with_the_prompt_length():
    """The image sits one plane past the *ceiling*, not past this run's gist count: a
    shorter prompt must not slide the picture under a different rotation."""
    short = sequence_positions(GIST_MIN_TOKENS, 2, 3, (), GIST)
    long = sequence_positions(GIST_MIN_TOKENS + 64, 2, 3, (), GIST)

    assert short.shape == (1, GIST_MIN_TOKENS + 6, 3)
    assert short[0, :GIST_MIN_TOKENS, 0].tolist() == list(range(1, GIST_MIN_TOKENS + 1))
    assert torch.equal(short[0, GIST_MIN_TOKENS:], long[0, long.shape[1] - 6:])
    assert (short[0, GIST_MIN_TOKENS:, 0] == GIST + 1).all()
    # The spatial axes are the token grid, uncentered: the gist rides only the t axis.
    assert short[0, GIST_MIN_TOKENS:, 1].tolist() == [0, 0, 0, 1, 1, 1]
    assert short[0, GIST_MIN_TOKENS:, 2].tolist() == [0, 1, 2, 0, 1, 2]


def test_each_reference_gets_the_plane_after_the_image():
    refs = [(2, 3), (1, 2)]
    ids = sequence_positions(GIST_MIN_TOKENS, 2, 2, refs, GIST)
    image, first, second = 4, 6, 2

    assert ids.shape == (1, GIST_MIN_TOKENS + image + first + second, 3)
    t = ids[0, GIST_MIN_TOKENS:, 0]
    assert (t[:image] == GIST + 1).all()
    assert (t[image:image + first] == GIST + 2).all()
    assert (t[image + first:] == GIST + 3).all()


# --------------------------------------------------------------------- modulation


def test_scale_tokens_splits_the_adaln_batch_at_the_span():
    """Rows are ``[clean; noisy]`` stacked over the batch: the span alone decides
    which tokens are the noisy image."""
    y = torch.ones(2, 5, 1)
    rows = torch.tensor([[[1.0]], [[2.0]], [[10.0]], [[20.0]]])  # 2 branches x clean, noisy

    out = scale_tokens(y, rows, (1, 3))
    assert out.flatten().tolist() == [1, 10, 10, 1, 1, 2, 20, 20, 2, 2]
    # With no span there is nothing to split: each branch gets its one row set.
    assert scale_tokens(y, rows[:2]).flatten().tolist() == [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]


@pytest.mark.parametrize("span,rows", [((0, 6), slice(1, None)), ((3, 3), slice(None, 1))])
def test_a_span_over_all_or_nothing_is_one_row_set_alone(dit, span, rows):
    """The refiner's two cases, checked against the same block fed one row set: what
    the span selects is the half of the AdaLN batch, and nothing else."""
    block = dit.layers[0]  # not an injection block: no text to feed it
    x, adaln = _n(1, 6, D), dit.t_embedder(torch.tensor([0.0, 0.7]))
    dit.pe_embedder.clear()
    dit.pe_embedder.store("pe", sequence_positions(0, 2, 3, (), GIST))
    pe = dit.pe_embedder["pe"]

    with torch.no_grad():
        mixed = block(x, pe, adaln, span)
        single = block(x, pe, adaln[rows], None)
    assert torch.allclose(mixed.float(), single.float(), atol=1e-5)


# --------------------------------------------------------------------------- DiT


def test_forward_answers_in_the_canonical_latent(dit):
    out = _run(dit, _n(1, Z, 4, 3), _n(1, 6, CTX))
    assert out.shape == (1, Z, 4, 3)
    assert torch.isfinite(out.float()).all()


def test_the_references_join_the_sequence_and_are_read(dit):
    x, ctx = _n(1, Z, 4, 3), _n(1, 6, CTX)
    plain = _run(dit, x, ctx)
    edited = _run(dit, x, ctx, refs=[_n(1, Z, 2, 3), _n(1, Z, 1, 2)])

    assert edited.shape == plain.shape  # only the image tokens answer
    assert not torch.allclose(edited.float(), plain.float(), atol=1e-2)


def test_the_gist_count_follows_the_prompt_length(dit):
    short = dit.gist_tokens(_n(1, 6, CTX))
    long = dit.gist_tokens(_n(1, 300, CTX))

    assert short.shape == (1, GIST_MIN_TOKENS, D)
    assert long.shape == (1, GIST_MIN_TOKENS + 64, D)


def test_only_an_injection_block_carries_a_cross_attention(dit):
    injection = set(TINY["injection_layers"])
    assert {i for i, block in enumerate(dit.layers) if block.cross_attention} == injection
    assert dit._bundle_of == {1: 1}  # bundle 0 belongs to the Perceiver


def test_an_injection_block_reads_its_own_bundle(dit):
    """Zeroing the second bundle silences the cross-attention of layer 1 while the
    Perceiver keeps the first, so the two really are separate reads."""
    x = _n(1, Z, 4, 3)
    ctx = _n(1, 6, CTX)
    half = CTX // 2
    without_text = torch.cat((ctx[..., :half], torch.zeros_like(ctx[..., half:])), dim=-1)

    assert not torch.allclose(
        _run(dit, x, ctx).float(), _run(dit, x, without_text).float(), atol=1e-2
    )


def test_the_head_is_not_modulated_by_the_clean_row(dit):
    """The output head is fed the noisy half of the AdaLN batch on its own — a run
    with no references still reads the image at ``t``, not at ``t = 0``."""
    adaln = dit.t_embedder(torch.tensor([0.0, 0.7]))
    head, tokens = dit.final_layer, _n(1, 12, D)
    with torch.no_grad():
        assert not torch.allclose(
            head(tokens, adaln[1:]).float(), head(tokens, adaln[:1]).float(), atol=1e-3
        )


# ------------------------------------------------------------------ sampling grid


def test_schedule_is_the_shift_of_a_uniform_grid():
    ts = get_sigmas(4, torch.device("cpu"))

    assert ts.shape == (5,)
    assert float(ts[0]) == pytest.approx(1.0)
    assert float(ts[-1]) == 0.0
    expected = [shift(float(s)) for s in torch.linspace(1.0, 0.0, 5)]
    assert ts.tolist() == pytest.approx(expected)
    # The shift pulls every interior point towards noise, and 1 onto itself.
    # Above 1 the shift holds every interior point closer to noise than the uniform
    # grid, which is what a few-step sampler wants, and leaves 1 on 1.
    grid = torch.linspace(1.0, 0.0, 5)
    assert (ts[1:-1] > grid[1:-1]).all()
    assert SHIFT > 1.0


def test_percent_to_sigma_is_the_same_shift():
    model = _bare(vae=FakeVAE())
    assert model.percent_to_sigma(0.0) == 1.0
    assert model.percent_to_sigma(1.0) == 0.0
    assert model.percent_to_sigma(0.4) == shift(0.6)


# --------------------------------------------------------------- checkpoint loading


def _as_int8(state_dict) -> dict:
    out = {}
    for key, value in state_dict.items():
        if value.dim() != 2 or not key.endswith(".weight"):
            # ``perceiver.latent_queries`` is 2D but no projection: no scale beside it.
            out[key] = value
            continue
        scale = (value.float().abs().amax(dim=1, keepdim=True) / 127).clamp_min(1e-8)
        out[key] = (value.float() / scale).round().clamp(-127, 127).to(torch.int8)
        out[f"{key}_scale"] = scale.float()
        out[key[: -len("weight")] + "comfy_quant"] = comfy_quant(convrot=False)
    return out


def test_detect_params_reads_the_geometry_from_the_header(tmp_path):
    path = write_safetensors(
        tmp_path / "tiny_fibo2.safetensors",
        {k: v.clone() for k, v in _tiny_dit(torch.float32).state_dict().items()},
    )
    assert detect_params(path) == _params_dict()


def test_detect_params_reads_the_unfused_export(tmp_path):
    """An export with ``w1``/``w3`` apart is the same model, and its hidden width is
    read from one matrix instead of half of the fused one."""
    source = {k: v.clone() for k, v in _tiny_dit(torch.float32).state_dict().items()}
    ff = "layers.0.feed_forward."
    fused = source.pop(ff + "w13.weight")
    source[ff + "w1.weight"], source[ff + "w3.weight"] = fused.chunk(2, dim=0)
    source.pop(ff + "w13.bias", None)

    assert detect_params(write_safetensors(tmp_path / "split.safetensors", source)) == (
        _params_dict(fused_ffn=False)
    )


def test_detect_params_refuses_a_foreign_checkpoint(tmp_path):
    path = write_safetensors(tmp_path / "other.safetensors", {"img_in.weight": torch.zeros(4, 4)})
    with pytest.raises(ValueError, match="not a Fibo 2.0 DiT"):
        detect_params(path)


def test_the_checkpoint_loads_back_exactly(tmp_path):
    source = {k: v.clone() for k, v in _tiny_dit(torch.float32).state_dict().items()}
    path = write_safetensors(tmp_path / "fibo2_bf16.safetensors", source)

    dit = load_fibo2_dit(path, _params_dict(), device="cpu", dtype=torch.bfloat16)
    assert set(dit.state_dict()) == set(source)
    assert torch.equal(dit.x_embedder.weight, source["x_embedder.weight"].to(torch.bfloat16))

    partial = {k: v for k, v in source.items() if k != "perceiver.latent_queries"}
    with pytest.raises(RuntimeError, match="[Mm]issing"):
        load_fibo2_dit(
            write_safetensors(tmp_path / "partial.safetensors", partial),
            _params_dict(), device="cpu", dtype=torch.float32,
        )


def test_both_qk_norm_spellings_load(tmp_path):
    """The rotated self-attention and the unrotated cross-attention are named
    differently in the release; the loader accepts either on either."""
    source = {k: v.clone() for k, v in _tiny_dit(torch.float32).state_dict().items()}
    renamed = {}
    for key, value in source.items():
        for legacy, ours in (("qk_norm.query_norm", "norm_q"), ("qk_norm.key_norm", "norm_k")):
            key = key.replace(legacy, ours)
        renamed[key] = value
    assert renamed, "no QK-norm keys to rename"

    dit = load_fibo2_dit(
        write_safetensors(tmp_path / "legacy_qknorm.safetensors", renamed),
        _params_dict(), device="cpu", dtype=torch.float32,
    )
    assert set(dit.state_dict()) == set(source)


def test_load_reconstructs_the_int8_export(tmp_path):
    source = {k: v.clone() for k, v in _tiny_dit(torch.float32).state_dict().items()}
    path = write_safetensors(tmp_path / "fibo2_int8.safetensors", _as_int8(source))

    dit = load_fibo2_dit(path, _params_dict(), device="cpu", dtype=torch.bfloat16)

    assert hasattr(dit.x_embedder.weight, "dequantize"), "x_embedder did not land quantized"
    step = float(source["x_embedder.weight"].abs().max()) / 127
    restored = dit.x_embedder.weight.dequantize().float()
    assert (restored - source["x_embedder.weight"]).abs().max() <= step * 2
    assert torch.isfinite(_run(dit, _n(1, Z, 4, 3), _n(1, 6, CTX)).float()).all()


# ------------------------------------------------------------------------ adapter


def test_init_latents_uses_the_vae_geometry_and_the_seed():
    model = _bare(vae=FakeVAE())
    latents = model.init_latents(_params(seed=7))

    assert latents.shape == (1, Z, 4, 3)  # one latent cell per 16 pixels
    assert torch.equal(latents, model.init_latents(_params(seed=7)))
    assert not torch.equal(latents, model.init_latents(_params(seed=8)))


class FakeEncoder:
    """A conditioner that records what it was asked to encode."""

    dtype = torch.bfloat16

    def __init__(self, hidden=CTX):
        self.hidden = hidden
        self.seen = []

    def __call__(self, prompt, images=None):
        self.seen.append((prompt, images))
        return torch.zeros(1, 4 + len(self.seen), self.hidden)


def test_encode_prompt_shows_the_images_to_one_branch_only():
    model = _bare(text_encoder=FakeEncoder())
    image = Image.new("RGB", (32, 32))

    cond = model.encode_prompt(
        EncodePromptArgs(prompt="make it night", guidance_scale=1.0, image=[image])
    )
    assert cond.null is None
    assert model.text_encoder.seen == [("make it night", [image])]

    model.text_encoder.seen.clear()
    both = model.encode_prompt(
        EncodePromptArgs(prompt="a cat", negative_prompt="blurry", guidance_scale=4.0)
    )
    assert both.cond.shape[0] == 1 and both.null.shape[0] == 1
    assert [(prompt, images) for prompt, images in model.text_encoder.seen] == [
        ("a cat", None),
        ("blurry", None),  # the negative of an edit carries no vision tokens
    ]


def test_prepare_latent_stashes_one_gist_and_one_table_per_branch(dit):
    model = _bare(dit=dit, params=_params_dict(), vae=FakeVAE())
    cond = Conditioning(cond=_n(1, 6, CTX), null=_n(1, 8, CTX))

    x = model.prepare_latent(_n(1, Z, 4, 3), cond, _params())

    assert x.shape == (1, Z, 4, 3)  # the latent needs no packing
    assert model._refs == []
    assert set(model._branches) == {"cond", "uncond"}
    assert dit.pe_embedder["cond"].shape[1] == GIST_MIN_TOKENS + 12
    assert dit.pe_embedder["uncond"].shape[1] == GIST_MIN_TOKENS + 12


def test_each_branch_gets_its_own_table_because_the_length_differs(dit):
    """Neither branch can be padded to fit the other: padding would be cross-attended,
    and it is the length that fixes the gist count."""
    model = _bare(dit=dit, params=_params_dict(), vae=FakeVAE())
    cond = Conditioning(cond=_n(1, 300, CTX), null=_n(1, 6, CTX))

    model.prepare_latent(_n(1, Z, 4, 3), cond, _params())

    assert model._branches["cond"][1].shape[1] == GIST_MIN_TOKENS + 64
    assert model._branches["uncond"][1].shape[1] == GIST_MIN_TOKENS
    assert dit.pe_embedder["cond"].shape != dit.pe_embedder["uncond"].shape


def test_prepare_latent_puts_each_reference_on_its_own_plane(dit):
    model = _bare(dit=dit, params=_params_dict(), vae=FakeVAE())
    refs = [torch.ones(1, Z, 2, 3), torch.ones(1, Z, 1, 2)]

    model.prepare_latent(
        _n(1, Z, 4, 3), Conditioning(cond=_n(1, 6, CTX)), _params(), ref=refs,
        ref_method="index_timestep_zero",
    )

    table = dit.pe_embedder["cond"]
    assert table.shape[1] == GIST_MIN_TOKENS + 12 + 6 + 2  # gist, image, both references

    # The first rope axis is the t one, so a token's rotation over it says which stream
    # it belongs to: one rotation across a plane, a different one on the next plane.
    def plane(i: int):
        return table[0, i, : dit.params.axes_dims[0] // 2]

    image, first_ref, second_ref = GIST_MIN_TOKENS, GIST_MIN_TOKENS + 12, -1
    assert torch.equal(plane(first_ref), plane(first_ref + 5))
    assert not torch.equal(plane(image), plane(first_ref))
    assert not torch.equal(plane(first_ref), plane(second_ref))


def test_the_accepted_reference_method_is_the_one_the_model_uses(dit):
    """A reference joins the sequence on its own plane at ``t = 0`` and there is no
    other reading of it, so the one accepted name is also the model's own default."""
    model = _bare(dit=dit, params=_params_dict(), vae=FakeVAE())
    assert Fibo2Model.REF_METHODS == ("index_timestep_zero",)
    assert Fibo2Model.DEFAULT_PREFS["ref_method"] == "index_timestep_zero"

    x = model.prepare_latent(
        _n(1, Z, 4, 3), Conditioning(cond=_n(1, 6, CTX)), _params(),
        ref=[torch.ones(1, Z, 2, 2)], ref_method="index_timestep_zero",
    )
    with _compute():
        velocity = model._dit_forward(x, torch.full((1,), 0.5, dtype=torch.float32), "cond")
    assert velocity.shape == x.shape


@pytest.mark.parametrize("method", ["index", "channel", "index_timestep_one"])
def test_an_unsupported_reference_method_is_refused(dit, method):
    """Including the plain ``index`` other edit models offer: there is no variant of
    this architecture where a reference carries the step's timestep."""
    model = _bare(dit=dit, params=_params_dict(), vae=FakeVAE())

    with pytest.raises(ValueError, match="unsupported ref_latents_method"):
        model.prepare_latent(
            _n(1, Z, 4, 3), Conditioning(cond=_n(1, 6, CTX)), _params(),
            ref=[torch.ones(1, Z, 2, 2)], ref_method=method,
        )


def test_the_method_is_only_asked_of_a_run_that_carries_references(dit):
    """The refiner pass of ``--upscale`` re-enters ``prepare_latent`` with no reference
    and the base default method; a name this model was never offered must not stop it."""
    model = _bare(dit=dit, params=_params_dict(), vae=FakeVAE())

    x = model.prepare_latent(
        _n(1, Z, 4, 3), Conditioning(cond=_n(1, 6, CTX)), _params(), ref_method="index"
    )
    assert x.shape[-2:] == (4, 3)


def test_the_cache_reads_back_the_sequence_it_replaced(dit):
    """The filling pass runs the references, the reading pass leaves them out and takes
    their K/V from the buffers. At one timestep and one latent the frozen slice holds
    what the full sequence would have computed, so the two agree exactly."""
    x, context, refs = _n(1, Z, 4, 3), _n(1, 6, CTX), [_n(1, Z, 2, 2)]

    full = _run(dit, x, context, refs=refs)
    cache = KVCache("cond")
    _run(dit, x, context, refs=refs, kv=cache)
    assert cache.filled
    read = _run(dit, x, context, refs=refs, kv=cache)

    assert torch.allclose(read, full, atol=1e-5), (read - full).abs().max().item()


def test_the_cache_is_filled_by_the_first_step_and_read_by_the_rest(dit):
    model = _bare(dit=dit, params=_params_dict(), vae=FakeVAE())
    cond = Conditioning(cond=_n(1, 6, CTX))
    x = model.prepare_latent(
        _n(1, Z, 4, 3), cond, _params(kv_cache=True), ref=[torch.ones(1, Z, 2, 2)],
        ref_method="index_timestep_zero",
    )
    cache = model.kv_cache("cond")
    assert cache is not None and not cache.filled

    with _compute():
        model.denoise_step(x, 0.6, cond, 1.0, 0)
        assert cache.filled
        velocity = model.denoise_step(x, 0.4, cond, 1.0, 1)
    assert velocity.shape == x.shape

    model.finalize_latent(velocity, _params())
    assert model.kv_cache("cond") is None  # gone before anything decodes


def test_every_branch_gets_its_own_frozen_references(dit):
    model = _bare(dit=dit, params=_params_dict(), vae=FakeVAE())
    cond = Conditioning(cond=_n(1, 6, CTX), null=_n(1, 6, CTX))
    model.prepare_latent(
        _n(1, Z, 4, 3), cond, _params(kv_cache=True, guidance_scale=3.0),
        ref=[torch.ones(1, Z, 2, 2)], ref_method="index_timestep_zero",
    )

    assert model.kv_cache("cond") is not None
    assert model.kv_cache("cond") is not model.kv_cache("uncond")


def test_a_run_without_references_starts_no_cache(dit):
    """What the cache freezes is reference K/V; a plain generation has none to freeze."""
    model = _bare(dit=dit, params=_params_dict(), vae=FakeVAE())
    model.prepare_latent(
        _n(1, Z, 4, 3), Conditioning(cond=_n(1, 6, CTX)), _params(kv_cache=True)
    )

    assert model.kv_cache("cond") is None


def test_encode_reference_encodes_pixels_at_full_size():
    vae = FakeVAE()
    encoded = _bare(vae=vae).encode_reference(torch.randn(3, 32, 48))
    assert vae.seen == (1, 3, 32, 48)  # one image, unscaled
    assert encoded.shape == (1, Z, 2, 3)


def test_denoise_step_does_plain_cfg_over_the_two_branches(dit):
    model = _bare(dit=dit, params=_params_dict(), vae=FakeVAE())
    cond = Conditioning(cond=_n(1, 6, CTX), null=_n(1, 9, CTX))
    x = model.prepare_latent(_n(1, Z, 4, 3), cond, _params())
    t, t_full = 0.6, torch.full((1,), 0.6, dtype=torch.bfloat16)

    with _compute():
        guided = model.denoise_step(x, t, cond, guidance_scale=3.0, i=0)
        pos = model._dit_forward(x, t_full, "cond")
        neg = model._dit_forward(x, t_full, "uncond")

    assert guided.shape == x.shape
    assert torch.allclose(guided.float(), (neg + 3.0 * (pos - neg)).float(), atol=1e-4)

    with _compute():
        assert torch.equal(
            model.denoise_step(x, t, cond, guidance_scale=1.0, i=0).float(),
            model._dit_forward(x, t_full, "cond").float(),
        )


def test_upscaling_is_the_trained_flux2_transcoder():
    """The Flux.2 AE latent goes through the Flux.2 Sesqui bridge, as its own would."""
    model = _bare(vae=FakeVAE(), _upscaler=None)
    upscaler = model.get_upscaler()

    assert isinstance(upscaler, SesquiLSRUpscaler)
    assert upscaler.format == "flux2"
    assert upscaler.scale == Fibo2Model.UPSCALE_SCALE
    assert upscaler(torch.randn(1, 128, 8, 8)).shape == (1, 128, 16, 16)


# ------------------------------------------------------------------------ sizing


def test_the_trained_grid_is_the_least_cropping_one_near_a_megapixel():
    assert trained_size(1.0) == (1024, 1024)
    assert trained_size(3 / 4) == (864, 1152)
    assert trained_size(16 / 9) == (1424, 800)

    for ratio in (0.2, 3 / 4, 1.0, 4 / 3, 16 / 9, 3.0):
        width, height = trained_size(ratio)
        assert (width % 16, height % 16) == (0, 0)
        assert 0.9 * TRAINED_AREA <= width * height <= 1.1 * TRAINED_AREA
        assert min(width / height / ratio, ratio * height / width) > 0.98


def test_the_trained_grid_turns_with_the_aspect():
    assert trained_size(2.0) == tuple(reversed(trained_size(0.5)))
    with pytest.raises(ValueError, match="aspect ratio must be positive"):
        trained_size(0)


def test_an_edit_without_a_size_asks_for_the_trained_grid():
    """Not a long-edge cap: the reference's aspect at the size the model was shown."""
    assert _bare().default_edit_size(768, 1024) == (864, 1152)


def test_a_single_reference_is_laid_over_the_canvas():
    """The edit lines up with its one reference pixel for pixel, so it is given the
    canvas's own grid — which is also what keeps the result framed like the input."""
    (fitted,) = _bare().fit_references([Image.new("RGB", (768, 1024))], (864, 1152))

    assert fitted.size == (864, 1152)


def test_the_canvas_crop_keeps_the_middle_of_the_reference():
    banded = Image.new("RGB", (100, 200), "white")
    banded.paste("black", (0, 0, 100, 50))       # a quarter of the height at each end,
    banded.paste("black", (0, 150, 100, 200))    # which is what the crop drops

    (fitted,) = _bare().fit_references([banded], (100, 100))

    assert fitted.getcolors(maxcolors=1024) == [(100 * 100, (255, 255, 255))]


def test_several_references_keep_their_own_shape():
    """No canvas fits two aspects at once, so they take the shared per-image fit."""
    first, second = _bare().fit_references(
        [Image.new("RGB", (4000, 1000)), Image.new("RGB", (1000, 4000))], (1024, 1024)
    )

    assert first.size == (2048, 512)
    assert second.size == (512, 2048)


# ---------------------------------------------------------------- prompt building


def test_a_plain_prompt_is_its_own_caption():
    assert parse_prompt("a cat on a roof") == ("a cat on a roof", None)
    assert parse_prompt("[1, 2]") == ("[1, 2]", None)  # JSON, but not a caption


def test_a_caption_keeps_its_shape_and_loses_its_scores():
    caption, instruction = parse_prompt(
        '{"short_description":"a cat","aesthetic_score":7.2,'
        '"lighting":{"aesthetic_score":9,"mood":"dim"},"background_setting":""}'
    )
    assert instruction is None
    assert caption == {"short_description": "a cat", "lighting": {"mood": "dim"}}


def test_an_empty_text_list_surves_because_it_says_something():
    """A caption of only a description and a text list says "no text in the image",
    which is a thing the model was trained to be told."""
    caption, _ = parse_prompt('{"short_description":"a sign","text_render":[]}')
    assert caption == {"short_description": "a sign", "text_render": []}


def test_the_null_caption_is_a_whole_turn_taken_as_it_stands():
    assert NULL_CAPTION.startswith(TEMPLATE.split("{}")[0])
    assert NULL_CAPTION.endswith(TEMPLATE.split("{}")[1])
    assert NULL_CAPTION.count("<|im_start|>") == 2  # one turn, not a turn in a turn
    assert build_prompt("", Image.new("RGB", (32, 32)))[0] == NULL_CAPTION


def test_a_plain_instruction_over_images_is_the_whole_edit_json():
    text, vision = build_prompt("make it night", Image.new("RGB", (1600, 800)))

    assert text == TEMPLATE.format(
        '{"image":"<|vision_start|><|image_pad|><|vision_end|>",'
        '"edit_instruction":"make it night"}'
    )
    assert len(vision) == 1
    assert max(vision[0].size) == VISION_LONG_EDGE  # Qwen3-VL never sees more


def test_several_images_are_numbered_and_a_caption_rides_along():
    prompt = (
        '{"edit_instruction":"swap them","structured_caption":'
        '{"short_description":"two cats","aesthetic_score":8}}'
    )
    text, vision = build_prompt(prompt, [Image.new("RGB", (64, 64)), Image.new("RGB", (64, 64))])

    assert '"image_1":"<|vision_start|><|image_pad|><|vision_end|>"' in text
    assert '"image_2":"<|vision_start|><|image_pad|><|vision_end|>"' in text
    assert '"structured_caption":{"short_description":"two cats"}' in text
    assert '"edit_instruction":"swap them"}' in text
    assert len(vision) == 2


def test_caption_text_is_handed_to_the_vision_tower_as_a_picture():
    """Where the text to draw is comes from: its marker goes inside the JSON and the
    rendered strip joins the images the vision tower sees."""
    caption = '{"short_description":"a sign","text_render":[{"text":"HELLO","font":"serif"}]}'
    text, vision = build_prompt(caption)

    assert '"image_tokens":"<|vision_start|><|image_pad|><|vision_end|>"' in text
    assert len(vision) == 1
    assert vision[0].width > 4 and vision[0].height > 4
    assert vision[0].getpixel((0, 0)) == (255, 255, 255)  # black type on white


def test_a_caption_that_asks_for_no_text_gets_no_strip():
    _, vision = build_prompt('{"short_description":"a cat","text_render":[{"text":""}]}')
    assert vision == []


# ------------------------------------------------------------- the conditioner


class _FakeBody:
    def __init__(self, lm, states):
        self.lm, self.states = lm, states

    def __call__(self, **inputs):
        self.lm.seen = inputs
        return SimpleNamespace(hidden_states=self.states)


class _FakeLM(torch.nn.Module):
    """A stand-in Qwen3-VL: hidden state of tap ``i`` is a ramp offset by ``1000 * i``."""

    def __init__(self, taps, length, hidden):
        super().__init__()
        self.probe = torch.nn.Parameter(torch.zeros(1))
        self.seen = {}
        ramp = torch.arange(length * hidden, dtype=torch.float32).reshape(1, length, hidden)
        self.model = _FakeBody(self, tuple(ramp + 1000.0 * i for i in range(taps)))


def _fake_conditioner(length=7, hidden=4):
    def processor(text, images=None):
        processor.text, processor.images = text[0], images
        inputs = {
            "input_ids": torch.arange(1, length + 1).reshape(1, length),
            "attention_mask": torch.ones(1, length, dtype=torch.long),
        }
        if images:
            inputs.update(
                pixel_values=torch.zeros(4 * hidden, hidden),
                image_grid_thw=torch.tensor([[1, 2, 2]]),
                mm_token_type_ids=torch.zeros(1, length, dtype=torch.long),
            )
        return inputs

    processor.text = processor.images = None
    lm = _FakeLM(max(BUNDLE_LAYERS[-1]) + 1, length, hidden)
    return Fibo2TextEncoder(lm, None, processor), lm, processor


def test_the_conditioner_is_six_bundles_of_three_taps_at_unit_rms():
    length, hidden = 7, 4
    encoder, lm, processor = _fake_conditioner(length, hidden)

    with torch.no_grad():
        out = encoder.encode("a cat")

    assert processor.text == TEMPLATE.format("a cat")
    assert out.shape == (1, length, len(BUNDLE_TAPS) * hidden)
    assert BUNDLE_TAPS == (
        9, 20, 31, 5, 16, 27, 7, 18, 29, 9, 20, 31, 11, 22, 33, 13, 24, 35
    )  # six bundles of three taps, straight from the release

    states = lm.model.states
    expected = torch.stack(
        [F.rms_norm(states[tap].float(), (hidden,), eps=NORM_EPS) for tap in BUNDLE_TAPS], dim=2
    ).flatten(2)
    assert torch.allclose(out.float(), expected, atol=1e-6)

    # The tap is read out of the model's own layer numbering: the first output row is
    # the state entering layer 9, the layer the Perceiver was trained on.
    first = out.float()[:, :, :hidden]
    assert torch.allclose(first, F.rms_norm(states[9].float(), (hidden,), eps=NORM_EPS), atol=1e-6)
    assert not torch.allclose(first, F.rms_norm(states[0].float(), (hidden,), eps=NORM_EPS))

    rms = out.float().unflatten(-1, (len(BUNDLE_TAPS), hidden)).pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)  # every tap normalized


def test_the_conditioner_feeds_the_vision_tower_and_casts_its_pixels():
    encoder, lm, processor = _fake_conditioner()
    with torch.no_grad():
        out = encoder.encode("make it night", [Image.new("RGB", (1600, 800))])

    assert max(processor.images[0].size) == VISION_LONG_EDGE
    assert {"pixel_values", "image_grid_thw", "mm_token_type_ids"} <= set(lm.seen)
    assert lm.seen["pixel_values"].dtype == encoder.dtype
    assert out.shape[0] == 1


def test_the_conditioner_runs_without_vision_inputs_at_all():
    encoder, lm, _ = _fake_conditioner()
    with torch.no_grad():
        encoder.encode("a cat")
    assert "pixel_values" not in lm.seen  # no empty vision batch for a plain prompt
