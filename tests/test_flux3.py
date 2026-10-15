"""Flux.3: mode routing, the timestep-zero streams, the flow-map target time, the
Qwen3-VL taps and the two checkpoint folds.

The catalog-wide contracts (detection matrix, schedule shape, defaults, capabilities,
size alignment) live in ``test_detect.py`` / ``test_schedules.py`` / ``test_catalog.py``.
"""
from __future__ import annotations

import contextlib

import pytest
import torch

from conftest import write_safetensors
from thenoise.dit.flux3.encoder import (
    OUTPUT_LAYERS,
    PAD_MULTIPLE,
    PAD_TOKEN_ID,
    Flux3TextEncoder,
    prompt_template,
)
from thenoise.dit.flux3.models import (
    IMAGE_ROPE_SCALE,
    Flux3,
    Flux3Params,
    image_ids,
    latent_to_tokens,
    ref_ids,
    text_ids,
    tokens_to_latent,
)
from thenoise.dit.flux3.sampling import FLOW_SHIFT, get_schedule, target_times
from thenoise.dit.flux3.utils import detect_params, fuse_qkv_state
from thenoise.models import Flux3Model
from thenoise.models.base import Conditioning
from thenoise.models.config import EncodePromptArgs, SamplingParams
from thenoise.utils.lora import _fuse_stacked
from thenoise.utils.timestep import timestep_embedding

# Two heads of the released 128; ``(2, 2, 2, 2)`` is the released 4-axis split.
TINY = dict(
    in_channels=8,
    context_in_dim=24,
    hidden_size=16,
    num_heads=2,
    depth=1,
    depth_single_blocks=2,
    axes_dim=[2, 2, 2, 2],
)
Z, CTX = TINY["in_channels"], TINY["context_in_dim"]


def _tiny_dit(flow_map: bool = False, dtype=torch.bfloat16) -> Flux3:
    torch.manual_seed(0)
    return Flux3(Flux3Params(**TINY, flow_map=flow_map)).to(dtype).eval().requires_grad_(False)


def _bare(**attrs) -> Flux3Model:
    """An adapter instance without ``__init__`` (no weights, no device moves)."""
    model = object.__new__(Flux3Model)
    model.device = "cpu"
    model.dtype = torch.bfloat16
    model.flow_map = False
    for key, value in attrs.items():
        setattr(model, key, value)
    return model


class FakeVAE:
    """Latent geometry and an encode stub, with no codec behind it."""

    z_dim = Z
    spatial_compression = 16
    pixel_channels = 3
    dtype = torch.bfloat16
    device = torch.device("cpu")

    def encode_pixels_to_latents(self, pixels):
        cells = pixels[:, :1, :: self.spatial_compression, :: self.spatial_compression]
        return cells.repeat(1, self.z_dim, 1, 1)


def _params(**overrides) -> SamplingParams:
    base = dict(height=64, width=48, steps=2, seed=0, guidance_scale=1.0, sampler="euler")
    base.update(overrides)
    return SamplingParams(**base)


# ------------------------------------------------------------- tokens / positions


def test_image_positions_scale_the_grid_and_keep_the_mode_on_the_l_axis():
    ids = image_ids(2, 3)
    assert ids.shape == (1, 6, 4)
    assert torch.equal(ids[0, :, 0], torch.zeros(6))  # still image: the t axis stays 0
    assert torch.equal(ids[0, :, 1], torch.tensor([0.0, 0.0, 0.0, 0.5, 0.5, 0.5]))
    assert torch.equal(ids[0, :, 2], torch.tensor([0.0, 0.5, 1.0, 0.0, 0.5, 1.0]))
    assert torch.equal(ids[0, :, 3], torch.zeros(6))  # the target owns l = 0


def test_image_positions_order_matches_the_token_pack():
    """Token ``i`` of the row-major pack is row ``i // w``, column ``i % w``: the ids
    and the tokens have to agree or every token gets the position of another one."""
    latents = torch.arange(6, dtype=torch.float32).reshape(1, 1, 2, 3)
    assert torch.equal(latent_to_tokens(latents)[0, :, 0], torch.arange(6, dtype=torch.float32))

    ids = image_ids(2, 3)[0] / IMAGE_ROPE_SCALE
    assert torch.equal(ids[:, 1], torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.float32))
    assert torch.equal(ids[:, 2], torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.float32))


def test_references_take_negative_l_axes_so_they_never_land_on_the_target():
    ids = ref_ids([(2, 2), (1, 1)])
    assert ids.shape == (1, 5, 4)
    assert torch.equal(ids[0, :, 3], torch.tensor([-1.0, -1.0, -1.0, -1.0, -2.0]))
    assert torch.equal(ids[0, :4, 1], torch.tensor([0.0, 0.0, 0.5, 0.5]))


def test_text_positions_span_the_l_axis_and_nothing_else():
    ids = text_ids(4)
    assert ids.shape == (1, 4, 4)
    assert torch.equal(ids[0, :, 3], torch.arange(4, dtype=torch.float32))
    assert torch.equal(ids[0, :, :3], torch.zeros(4, 3))


def test_tokens_round_trip_the_canonical_latent():
    latents = torch.randn(1, Z, 4, 3)
    assert torch.equal(tokens_to_latent(latent_to_tokens(latents), 4, 3), latents)


# ------------------------------------------------------------------------ the DiT


@pytest.fixture
def dit():
    return _tiny_dit()


def _positions(dit: Flux3, target=12, refs=6, txt=5):
    dit.pe_embedder.clear()
    dit.pe_embedder.store("img", torch.zeros(1, target, 4, dtype=torch.float32))
    dit.pe_embedder.store("txt", torch.zeros(1, txt, 4, dtype=torch.float32))
    dit.pe_embedder.store("txt_uncond", torch.zeros(1, txt, 4, dtype=torch.float32))
    if refs:
        dit.pe_embedder.store("ref", torch.zeros(1, refs, 4, dtype=torch.float32))


def _t(value=0.7, dtype=torch.bfloat16):
    """A step's timestep, in the DiT's compute dtype (what the adapter hands over)."""
    return torch.full((1,), value, dtype=dtype)


def _n(*shape, dtype=torch.bfloat16):
    """Activations arrive in the compute dtype too — the adapter casts at the edges."""
    return torch.randn(*shape, dtype=dtype)


@contextlib.contextmanager
def _compute():
    """The adapter runs the DiT under autocast: ``Modulation`` upcasts its vector to
    fp32 for numerical stability and the GEMMs come back down to the compute dtype."""
    with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        yield


def test_forward_returns_the_target_tokens_only(dit):
    _positions(dit)
    vec = dit.timestep_vec(_t())
    with _compute():
        txt_states = dit.text_states(_n(1, 5, CTX), vec[1:], dit.pe_embedder["txt"])
        ref_states = dit.reference_states(_n(1, 6, Z), vec[1:], dit.pe_embedder["ref"])
        out = dit(
            _n(1, 12, Z), vec, dit.pe_embedder["img"],
            txt_states, dit.pe_embedder["txt"], ref_states, dit.pe_embedder["ref"],
        )
        plain = dit(
            _n(1, 12, Z), vec, dit.pe_embedder["img"],
            txt_states, dit.pe_embedder["txt"],
        )

    assert out.shape == (1, 12, Z)
    assert torch.isfinite(out.float()).all()
    assert plain.shape == out.shape  # references join the sequence, never the output


def test_only_the_early_blocks_read_one_mode_they_join_in_the_joint_blocks(dit):
    """The mode routing is the whole idea of the architecture: a target early block
    must never see a reference or a text token, and the joint blocks must see all."""
    _positions(dit)
    widths = {"early": [], "joint": []}
    dit.content_mode_blocks["image"][0].register_forward_pre_hook(
        lambda m, a: widths["early"].append(a[0].shape[1])
    )
    dit.single_blocks[0].register_forward_pre_hook(
        lambda m, a: widths["joint"].append(a[0].shape[1])
    )

    vec = dit.timestep_vec(_t())
    with _compute():
        dit(
            _n(1, 12, Z), vec, dit.pe_embedder["img"],
            dit.text_states(_n(1, 5, CTX), vec[1:], dit.pe_embedder["txt"]),
            dit.pe_embedder["txt"],
            dit.reference_states(_n(1, 6, Z), vec[1:], dit.pe_embedder["ref"]),
            dit.pe_embedder["ref"],
        )

    assert widths["early"] == [12]  # the target alone
    assert widths["joint"] == [12 + 6 + 5]  # image, references, text, in one sequence


def test_text_and_references_are_modulated_at_timestep_zero(dit):
    """Row 1 of the modulation vector drives both other streams; the target reads row 0."""
    _positions(dit)
    seen = {}
    for mode in ("image", "image_cond", "txt"):
        dit.early_stream_modulations[mode].register_forward_pre_hook(
            lambda m, a, mode=mode: seen.__setitem__(mode, a[0])
        )

    vec = dit.timestep_vec(_t())
    hidden = TINY["hidden_size"]
    with _compute():
        dit.text_states(_n(1, 5, CTX), vec[1:], dit.pe_embedder["txt"])
        dit.reference_states(_n(1, 6, Z), vec[1:], dit.pe_embedder["ref"])
        dit(
            _n(1, 12, Z), vec, dit.pe_embedder["img"],
            torch.zeros(1, 5, hidden, dtype=torch.bfloat16), dit.pe_embedder["txt"],
        )

    assert torch.equal(seen["txt"], vec[1:])
    assert torch.equal(seen["image_cond"], vec[1:])
    assert torch.equal(seen["image"], vec[:1])


def test_a_flow_map_adds_the_step_target_to_every_row_and_demands_it():
    dit = _tiny_dit(flow_map=True)
    with pytest.raises(ValueError, match="flow-map Flux.3"):
        dit.timestep_vec(_t())

    t, target = _t(), _t(0.4)
    vec = dit.timestep_vec(t, target)
    expected = dit.time_in(timestep_embedding(torch.cat([t, t.new_zeros(1)]), 256).to(t.dtype)) + \
        dit.ref_time_in(timestep_embedding(target, 256).to(t.dtype))
    assert torch.allclose(vec.float(), expected.float(), atol=1e-2)
    assert not torch.allclose(vec[0].float(), vec[1].float(), atol=1e-3)  # t row vs zero row


def test_a_plain_dit_has_no_way_to_embed_a_target_time(dit):
    assert not hasattr(dit, "ref_time_in")
    assert dit.timestep_vec(_t()).shape == (2, TINY["hidden_size"])


# ------------------------------------------------------------------- sampling grid


def test_schedule_is_the_reference_shift_of_a_linear_grid():
    ts = get_schedule(4)
    assert len(ts) == 5
    assert ts[0] == 1.0 and ts[-1] == 0.0  # the shift maps its own endpoints exactly
    assert all(a > b for a, b in zip(ts, ts[1:]))

    raw = torch.linspace(1, 0, 5).tolist()
    shifted = [(s * FLOW_SHIFT) / (1 + (FLOW_SHIFT - 1) * s) for s in raw]
    assert ts == pytest.approx(shifted)


def test_target_times_are_the_grid_each_step_jumps_to():
    schedule = _bare().schedule(_params(steps=4))
    assert target_times(schedule) == pytest.approx([float(s.t) for s in schedule[1:]] + [0.0])


# -------------------------------------------------------------------- the adapter


def test_init_latents_uses_the_vae_geometry_and_the_seed():
    model = _bare(vae=FakeVAE())
    latents = model.init_latents(_params(seed=7))

    assert latents.shape == (1, Z, 4, 3)  # one latent cell per 16 pixels
    assert torch.equal(latents, model.init_latents(_params(seed=7)))
    assert not torch.equal(latents, model.init_latents(_params(seed=8)))


def test_encode_prompt_leaves_the_references_to_the_latent_path():
    seen = []

    def encode(prompt):
        seen.append(prompt)
        return torch.zeros(1, 6, CTX)

    model = _bare(text_encoder=encode)
    cond = model.encode_prompt(
        EncodePromptArgs(prompt="cat", negative_prompt="blur", guidance_scale=3.0,
                         image="an image the conditioner must not see")
    )
    assert seen == ["cat", "blur"]
    assert cond.null is not None

    seen.clear()
    model.encode_prompt(EncodePromptArgs(prompt="cat", guidance_scale=1.0))
    assert seen == ["cat"]  # guidance off: no unconditional branch


def test_prepare_latent_stashes_tokens_positions_and_text(dit):
    model = _bare(dit=dit, vae=FakeVAE())
    cond = Conditioning(cond=torch.randn(1, 5, CTX), null=torch.randn(1, 3, CTX))
    tokens = model.prepare_latent(torch.randn(1, Z, 4, 3), cond, _params())

    assert tokens.shape == (1, 12, Z)
    assert model._txt.shape == (1, 5, CTX)
    assert model._un_txt.shape == (1, 3, CTX)
    assert model._branches == ("cond", "uncond")
    assert model._ref_tokens is None
    cached = dit.pe_embedder._cache
    assert cached["img"].shape[1] == 12  # the target only
    assert cached["txt"].shape[1] == 5
    assert cached["txt_uncond"].shape[1] == 3  # the branches differ in length
    assert "ref" not in cached


def test_prepare_latent_gives_each_reference_its_own_position_slice(dit):
    model = _bare(dit=dit, vae=FakeVAE())
    seen = []
    pack = model.pack_reference_latent

    def spy(latents, method="index", ref_index=1):
        seen.append((method, ref_index))
        return pack(latents, method, ref_index)

    model.pack_reference_latent = spy
    refs = [torch.randn(1, Z, 2, 3), torch.randn(1, Z, 2, 2)]
    model.prepare_latent(
        torch.randn(1, Z, 4, 3), Conditioning(cond=torch.randn(1, 5, CTX)), _params(), ref=refs
    )

    assert model._ref_tokens.shape == (1, 10, Z)  # both references, one stream
    assert dit.pe_embedder._cache["ref"].shape[1] == 10  # and one position row each
    assert seen == [("index", 1), ("index", 2)]  # 1-based: the target owns 0


def test_without_a_flow_map_the_timestep_zero_streams_run_once_per_run(dit):
    model = _bare(dit=dit, vae=FakeVAE())
    calls = []
    real = dit.text_states
    dit.text_states = lambda *a, **k: calls.append(1) or real(*a, **k)

    cond = Conditioning(cond=torch.randn(1, 5, CTX), null=torch.randn(1, 3, CTX))
    x = model.prepare_latent(torch.randn(1, Z, 4, 3), cond, _params())
    assert len(calls) == 2  # both branches, once each
    assert set(model._txt_states) == {"cond", "uncond"}

    model.begin_sampling(model.schedule(_params(steps=3)))
    with torch.no_grad():
        for i in range(3):
            model.denoise_step(x, torch.tensor(0.7), cond, 1.0, i)
    assert len(calls) == 2  # step-invariant: nothing was recomputed


def test_a_flow_map_redoes_the_timestep_zero_streams_every_step():
    dit = _tiny_dit(flow_map=True)
    model = _bare(dit=dit, vae=FakeVAE(), flow_map=True)
    calls = []
    real = dit.text_states
    dit.text_states = lambda *a, **k: calls.append(1) or real(*a, **k)

    cond = Conditioning(cond=torch.randn(1, 5, CTX))
    x = model.prepare_latent(torch.randn(1, Z, 4, 3), cond, _params())
    assert model._txt_states == {}  # nothing to carry across steps

    model.begin_sampling(model.schedule(_params(steps=3)))
    with torch.no_grad():
        for i in range(3):
            model.denoise_step(x, torch.tensor(0.7), cond, 1.0, i)
    assert len(calls) == 3


def test_references_are_embedded_once_per_step_even_under_cfg():
    """The branches differ only in their text, so one reference stream serves both."""
    dit = _tiny_dit(flow_map=True)
    model = _bare(dit=dit, vae=FakeVAE(), flow_map=True)
    cond = Conditioning(cond=torch.randn(1, 5, CTX), null=torch.randn(1, 3, CTX))
    x = model.prepare_latent(
        torch.randn(1, Z, 4, 3), cond, _params(), ref=[torch.randn(1, Z, 2, 2)]
    )

    calls = []
    real = dit.reference_states
    dit.reference_states = lambda *a, **k: calls.append(1) or real(*a, **k)

    schedule = model.schedule(_params(steps=2))
    model.begin_sampling(schedule)
    with torch.no_grad():
        for i in range(2):
            model.denoise_step(x, schedule[i].t, cond, 3.0, i)
    assert len(calls) == 2  # per step, not per branch


def test_denoise_step_feeds_t_and_target_and_does_plain_cfg():
    dit = _tiny_dit(flow_map=True)
    model = _bare(dit=dit, vae=FakeVAE(), flow_map=True)
    cond = Conditioning(cond=torch.randn(1, 5, CTX), null=torch.randn(1, 3, CTX))
    x = model.prepare_latent(torch.randn(1, Z, 4, 3), cond, _params())

    seen, outputs = [], []
    real_forward = dit.forward

    def spy(*args, **kwargs):
        seen.append(args)
        outputs.append(real_forward(*args, **kwargs))
        return outputs[-1]

    dit.forward = spy
    schedule = model.schedule(_params(steps=4))
    model.begin_sampling(schedule)
    with torch.no_grad():
        v = model.denoise_step(x, schedule[1].t, cond, 3.0, 1)

    assert len(seen) == 2  # the conditional and the unconditional branch
    t = torch.full((1,), float(schedule[1].t), dtype=torch.bfloat16)
    target = torch.full((1,), float(schedule[1].t) - float(schedule[1].delta), dtype=torch.bfloat16)
    assert torch.equal(seen[0][1], dit.timestep_vec(t, target))  # the step's own vec
    assert torch.equal(seen[0][1], seen[1][1])  # both branches, one vec
    assert seen[0][0] is seen[1][0]  # both branches, same x
    assert [s[3].shape[1] for s in seen] == [5, 3]  # the branches' own text lengths

    pos, neg = (out.float() for out in outputs)
    assert v.shape == x.shape
    # Unnormalised: exactly ``neg + scale * (pos - neg)``.
    assert torch.allclose(v.float(), neg + 3.0 * (pos - neg), rtol=2e-2, atol=1e-3)

    seen.clear()
    with torch.no_grad():
        model.denoise_step(x, schedule[1].t, cond, 1.0, 1)
    assert len(seen) == 1  # guidance off: one forward


def test_finalize_latent_unpacks_the_target_tokens():
    latents = torch.randn(1, Z, 4, 3)
    out = _bare(vae=FakeVAE()).finalize_latent(latent_to_tokens(latents), _params())
    assert torch.equal(out.float(), latents)


def test_references_are_packed_on_negative_l_axes():
    tokens, ids = _bare().pack_reference_latent(torch.randn(1, Z, 2, 3), ref_index=2)

    assert tokens.shape == (1, 6, Z)
    assert ids.shape == (1, 6, 4)
    assert torch.equal(ids[0, :, 3], torch.full((6,), -2.0))
    assert torch.equal(ids[0, :, 1] / IMAGE_ROPE_SCALE, torch.tensor([0.0] * 3 + [1.0] * 3))


@pytest.mark.parametrize("method", ["crop", "index_timestep_one"])
def test_only_the_two_index_reference_methods_are_accepted(method):
    """``index`` and ``index_timestep_zero`` are one packing here (Flux.3 always
    modulates references at zero); anything else is a user error, not a silent choice."""
    with pytest.raises(ValueError, match="unsupported ref_latents_method"):
        _bare().pack_reference_latent(torch.randn(1, Z, 2, 2), method=method)


def test_encode_reference_encodes_pixels_at_full_size():
    assert _bare(vae=FakeVAE()).encode_reference(torch.randn(3, 32, 48)).shape == (1, Z, 2, 3)


# --------------------------------------------------------------- checkpoint folds


def _split_qkv(rows=4, cols=3, per_row_scale=True):
    """A split ``q_proj``/``k_proj``/``v_proj`` export, one block, distinguishable parts."""
    base = torch.arange(rows * cols, dtype=torch.float32).reshape(rows, cols)
    scale = torch.ones(rows, 1) if per_row_scale else torch.ones(1)
    tensors = {}
    for i, proj in enumerate(("q_proj", "k_proj", "v_proj")):
        tensors[f"single_blocks.0.{proj}.weight"] = base + 100 * i
        tensors[f"single_blocks.0.{proj}.weight_scale"] = scale.clone()
        tensors[f"single_blocks.0.{proj}.comfy_quant"] = torch.tensor(
            list(b"{}"), dtype=torch.uint8
        )
    return tensors


def test_a_split_qkv_export_is_stacked_in_q_k_v_row_order():
    state = fuse_qkv_state(_split_qkv())

    fused = state["single_blocks.0.qkv.weight"]
    assert fused.shape == (12, 3)
    assert torch.equal(fused[:4], torch.arange(12, dtype=torch.float32).reshape(4, 3))
    assert torch.equal(fused[4:8], fused[:4] + 100)
    assert torch.equal(fused[8:], fused[:4] + 200)
    assert state["single_blocks.0.qkv.weight_scale"].shape == (12, 1)
    assert not [k for k in state if any(p in k for p in ("q_proj", "k_proj", "v_proj"))]


def test_a_shared_per_tensor_scale_survives_the_fold_once():
    assert fuse_qkv_state(_split_qkv(per_row_scale=False))["single_blocks.0.qkv.weight_scale"].numel() == 1


def test_mixed_scale_granularities_are_refused():
    state = _split_qkv()
    state["single_blocks.0.k_proj.weight_scale"] = torch.ones(1)
    with pytest.raises(ValueError, match="per-tensor and per-row"):
        fuse_qkv_state(state)


def test_parts_quantized_differently_cannot_share_one_profile():
    state = _split_qkv()
    state["single_blocks.0.k_proj.comfy_quant"] = torch.tensor(list(b"{\"a\":1}"), dtype=torch.uint8)
    with pytest.raises(ValueError, match="different quantization profiles"):
        fuse_qkv_state(state)


def test_a_partial_qkv_set_is_refused():
    state = _split_qkv()
    del state["single_blocks.0.v_proj.weight"]
    with pytest.raises(ValueError, match="partial weight set"):
        fuse_qkv_state(state)


def test_folding_is_a_noop_for_an_already_fused_state():
    state = {"single_blocks.0.qkv.weight": torch.zeros(3, 3)}
    assert fuse_qkv_state(state) is state


def test_loras_naming_the_separate_projections_land_on_the_fused_one():
    """The DiT only ever has ``qkv``; a LoRA trained on the separate projections is
    stacked onto it before matching, in the fused matrix's row order."""
    rank, out, in_dim = 2, 4, 3
    projects = ("q_proj", "k_proj", "v_proj")
    parts, deltas = {}, {}
    for i, proj in enumerate(projects):
        down = torch.arange(rank * in_dim, dtype=torch.float32).reshape(rank, in_dim) + i
        up = torch.arange(out * rank, dtype=torch.float32).reshape(out, rank) + 10 * i
        parts[f"single_blocks.0.{proj}.lora_A.weight"] = down
        parts[f"single_blocks.0.{proj}.lora_B.weight"] = up
        deltas[proj] = up @ down

    fused = _fuse_stacked(parts, tuple(Flux3Model.lora_fusions["qkv"]), "qkv")

    assert set(fused) == {
        "single_blocks.0.qkv.lora_A.weight",
        "single_blocks.0.qkv.lora_B.weight",
    }
    assert fused["single_blocks.0.qkv.lora_A.weight"].shape == (3 * rank, in_dim)
    merged = fused["single_blocks.0.qkv.lora_B.weight"] @ fused["single_blocks.0.qkv.lora_A.weight"]
    # One fused delta: the three separate ones, stacked in q/k/v row order.
    assert torch.allclose(merged, torch.cat([deltas[p] for p in projects], dim=0))


# ------------------------------------------------------------------ text encoder


class _FakeTokenizer:
    """Any text becomes ``length`` tokens; records how it was called."""

    def __init__(self, length=100):
        self.length = length
        self.seen = {}

    def __call__(self, text, add_special_tokens=True):
        self.seen = {"text": text, "add_special_tokens": add_special_tokens}
        return {"input_ids": list(range(10, 10 + self.length))}


class _FakeBody:
    """``hidden_states[i]`` is a constant ``i``, so the tap set and order are visible."""

    def __init__(self, lm):
        self.lm = lm

    def __call__(self, **inputs):
        self.lm.seen = inputs
        # The LM sees what the encoder built, pads included, and answers at that length.
        length = inputs["input_ids"].shape[1]
        return type(
            "Out", (),
            {"hidden_states": tuple(
                torch.full((1, length, self.lm.hidden), float(i)) for i in range(37)
            )},
        )()


class _FakeLM(torch.nn.Module):
    def __init__(self, length, hidden, dtype=torch.float32):
        super().__init__()
        self.probe = torch.nn.Parameter(torch.zeros(1, dtype=dtype))
        self.length, self.hidden, self.seen = length, hidden, {}
        self.model = _FakeBody(self)


def _fake_encoder(length=100, hidden=6):
    tokenizer = _FakeTokenizer(length)
    lm = _FakeLM(length, hidden)
    return Flux3TextEncoder(lm, tokenizer), tokenizer, lm


def test_the_template_is_a_user_turn_without_a_thinking_block():
    text = prompt_template("a cat")
    assert text == "<|im_start|>user\na cat<|im_end|>\n<|im_start|>assistant\n"
    assert "<think>" not in text and "system" not in text


@pytest.mark.parametrize("length,padded", [(80, 80), (81, 160), (100, 160), (1, 80)])
def test_the_sequence_is_aligned_with_endoftext_pads(length, padded):
    encoder, tokenizer, lm = _fake_encoder(length)
    with torch.no_grad():
        embeds = encoder.encode("a cat")

    assert tokenizer.seen["text"] == prompt_template("a cat")
    assert tokenizer.seen["add_special_tokens"] is False  # the template IS the sequence
    ids = lm.seen["input_ids"]
    assert ids.shape == (1, padded)
    assert torch.equal(ids[0, :length], torch.arange(10, 10 + length, dtype=torch.long))
    assert torch.equal(ids[0, length:], torch.full((padded - length,), PAD_TOKEN_ID, dtype=torch.long))
    assert embeds.shape == (1, padded, len(OUTPUT_LAYERS) * 6)


def test_the_padding_is_attended_and_counted_like_a_real_token():
    """Nothing is masked out: the reference weights the pads 1.0, causality keeps the
    prompt clean of them, and the DiT attends to the whole padded block."""
    length = 95
    encoder, _, lm = _fake_encoder(length)
    with torch.no_grad():
        encoder.encode("a cat")

    assert torch.equal(lm.seen["attention_mask"], torch.ones(1, PAD_MULTIPLE * 2, dtype=torch.long))
    assert torch.equal(
        lm.seen["position_ids"], torch.arange(PAD_MULTIPLE * 2, dtype=torch.long).unsqueeze(0)
    )
    assert lm.seen["use_cache"] is False  # one pass, no generation


def test_the_context_is_the_eight_taps_concatenated_in_layer_order():
    encoder, _, _ = _fake_encoder(80, hidden=6)
    with torch.no_grad():
        embeds = encoder.encode("a cat")

    assert embeds.shape == (1, 80, len(OUTPUT_LAYERS) * 6)
    assert torch.equal(
        embeds[0, 0],
        torch.tensor([float(layer) for layer in OUTPUT_LAYERS for _ in range(6)]),
    )


# --------------------------------------------------------------- header detection


def test_a_tiny_checkpoint_is_enough_to_read_the_geometry(tmp_path):
    dit = _tiny_dit(flow_map=True)
    tensors = {  # the export spells the QK-norm leaves ``scale``
        k.replace(".norm.query_norm.weight", ".norm.query_norm.scale")
         .replace(".norm.key_norm.weight", ".norm.key_norm.scale"): v.clone()
        for k, v in dit.state_dict().items()
    }
    path = write_safetensors(tmp_path / "tiny_flux3.safetensors", tensors)

    params = detect_params(path)
    assert (params.hidden_size, params.num_heads) == (TINY["hidden_size"], TINY["num_heads"])
    assert params.depth == TINY["depth"]
    assert params.depth_single_blocks == TINY["depth_single_blocks"]
    assert params.in_channels == TINY["in_channels"]
    assert params.context_in_dim == TINY["context_in_dim"]
    assert params.flow_map is True and params.fused_qkv is True
    assert sum(params.axes_dim) == TINY["hidden_size"] // TINY["num_heads"]
