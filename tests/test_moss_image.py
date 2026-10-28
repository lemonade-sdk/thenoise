"""MOSS-Image: the split-half rope convention, the joint sequence, the per-branch
positions, the step-invariant text front-end and the checkpoint folds.

The catalog-wide contracts (detection matrix, schedule shape, defaults, capabilities,
size alignment) live in ``test_detect.py`` / ``test_schedules.py`` / ``test_catalog.py``.
"""
from __future__ import annotations

import contextlib

import pytest
import torch

from conftest import write_safetensors
from thenoise.dit.moss_image.encoder import (
    MAX_SEQUENCE_LENGTH,
    MossImageTextEncoder,
    prompt_template,
)
from thenoise.dit.moss_image.models import (
    SITU_CAP,
    MossImage,
    MossImageParams,
    joint_positions,
    situ,
    situ_unscaled,
)
from thenoise.dit.moss_image.sampling import FLOW_SHIFT, get_schedule
from thenoise.dit.moss_image.utils import detect_params, load_moss_image_dit
from thenoise.models import MossImageModel
from thenoise.models.base import Conditioning
from thenoise.models.config import EncodePromptArgs, SamplingParams
from thenoise.utils.lora import _fuse_stacked
from thenoise.utils.rope import apply_rope_split_half, split_half_rope_4d

# head_dim 8 is the smallest this architecture accepts (4 rope axes x 2 groups).
TINY = dict(
    dim=16,
    ffn_dim=32,
    num_heads=2,
    num_layers=2,
    text_dim=24,
    latent_channels=8,
    freq_dim=16,
)
DIM, Z, CTX = TINY["dim"], TINY["latent_channels"], TINY["text_dim"]


def _tiny_dit(dtype: torch.dtype = torch.bfloat16) -> MossImage:
    torch.manual_seed(0)
    return MossImage(MossImageParams(**TINY)).to(dtype).eval().requires_grad_(False)


def _bare(**attrs) -> MossImageModel:
    """An adapter instance without ``__init__`` (no weights, no device moves)."""
    model = object.__new__(MossImageModel)
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


def _params(**overrides) -> SamplingParams:
    base = dict(height=48, width=32, steps=2, seed=0, guidance_scale=1.0, sampler="er_sde")
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


@pytest.fixture(autouse=True)
def _blocks_run_eagerly(monkeypatch):
    """Call the joint blocks un-compiled: a ``fullgraph=True`` block raises rather than
    falling back when dynamo is off, which ``conftest`` disables suite-wide."""
    from thenoise.dit.moss_image.models import Block

    monkeypatch.setattr(
        Block,
        "forward",
        Block.forward._torchdynamo_orig_callable,
        raising=False,
    )


# --------------------------------------------------------------------- rope convention


def _reference_rope(height: int, width: int, text_length: int, bands: int, base: float):
    """An independent ``[N, head_dim]`` table of sin|cos, to pin the convention."""
    scale = 1 / (base ** (torch.arange(bands, dtype=torch.float64) / bands))
    h, w = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    image = (
        torch.stack((torch.zeros_like(h), torch.full_like(h, text_length), h, w), -1)
        .flatten(0, 1)
        .double()
    )
    image[:, 2] -= (height - 1) / 2
    image[:, 3] -= (width - 1) / 2
    text = torch.zeros(text_length, 4, dtype=torch.float64)
    text[:, 1] = torch.arange(text_length)
    phase = torch.cat((image, text), 0).unsqueeze(-1) * scale
    return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2)


def _reference_rotate(x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    """The rotation of that table, written out on our ``[B, head, token, head_dim]``."""
    sin, cos = rope.chunk(2, -1)
    sin = torch.cat((sin, sin), -1)[None, None]  # [1, 1, token, head_dim]
    cos = torch.cat((cos, cos), -1)[None, None]
    a, b = x.chunk(2, -1)
    return x * cos + torch.cat((-b, a), -1) * sin


@pytest.mark.parametrize("head_dim", [32, 128])
@pytest.mark.parametrize("heads", [1, 4])
def test_the_rope_rotates_the_pairs_and_axes_the_reference_does(head_dim, heads):
    """Pairing, and which axis each pair belongs to, pinned against a second
    construction of the same rope rather than against this one."""
    height, width, text_length = 3, 4, 5
    tokens = height * width + text_length
    torch.manual_seed(0)
    q = torch.randn(2, heads, tokens, head_dim, dtype=torch.float64)

    cos, sin = split_half_rope_4d(head_dim, 10000.0)(
        joint_positions(height, width, text_length, dtype=torch.float64)
    )
    ours = apply_rope_split_half(q, cos, sin)
    reference = _reference_rotate(
        q, _reference_rope(height, width, text_length, head_dim // 8, 10000.0)
    )

    assert torch.equal(ours, reference)


def test_the_rope_tables_are_duplicated_over_the_head_dim():
    cos, sin = split_half_rope_4d(64, 10000.0)(joint_positions(2, 2, 3, dtype=torch.float64))
    assert cos.shape == sin.shape == (7, 1, 1, 64)
    assert torch.equal(cos[..., :32], cos[..., 32:])
    assert torch.equal(sin[..., :32], sin[..., 32:])


def test_a_head_dim_that_will_not_split_into_four_axes_is_refused():
    with pytest.raises(ValueError, match="divisible by 8"):
        split_half_rope_4d(12)


# ------------------------------------------------------------------- tokens, positions


def test_the_image_sits_after_the_text_on_f_with_a_centred_grid():
    ids = joint_positions(2, 3, 4)

    assert ids.shape == (2 * 3 + 4, 4)
    assert torch.equal(ids[:, 0], torch.zeros(10))  # r: still images
    assert torch.equal(ids[:6, 1], torch.full((6,), 4.0))  # the grid shares f = text length
    assert torch.equal(ids[6:, 1], torch.arange(4, dtype=torch.float32))
    assert torch.equal(ids[6:, 2:], torch.zeros(4, 2))  # the text has no spatial position
    assert torch.equal(ids[:6, 2], torch.tensor([-0.5] * 3 + [0.5] * 3))
    assert torch.equal(ids[:6, 3], torch.tensor([-1.0, 0.0, 1.0] * 2))


def test_token_order_of_the_video_pack_matches_the_position_rows():
    """Otherwise every token gets rotated as some other token."""
    latents = torch.arange(6, dtype=torch.float32).reshape(1, 1, 2, 3)
    packed = latents.flatten(2).transpose(1, 2)
    assert packed.shape == (1, 6, 1)
    assert torch.equal(packed[:, :, 0], torch.arange(6, dtype=torch.float32).unsqueeze(0))

    ids = joint_positions(2, 3, 0)
    assert torch.equal(ids[:, 2], torch.tensor([-0.5] * 3 + [0.5] * 3))
    assert torch.equal(ids[:, 3], torch.tensor([-1.0, 0.0, 1.0] * 2))


# ---------------------------------------------------------------------------- the DiT


@pytest.fixture
def dit():
    return _tiny_dit()


def _positions(dit: MossImage, image: int, text: int, dtype=torch.bfloat16):
    dit.pe_embedder.clear()
    ids = joint_positions(3, 4, text)
    dit.pe_embedder.store("joint", ids, dtype=dtype)
    dit.pe_embedder.store("txt", ids[image:], dtype=dtype)


def test_forward_answers_in_the_canonical_latent(dit):
    height, width, text = 3, 4, 5
    _positions(dit, height * width, text)
    with _compute():
        txt = dit.text_states(_n(1, text, CTX), dit.pe_embedder["txt"])
        out = dit(_n(1, Z, height, width), torch.full((1,), 0.7), txt, dit.pe_embedder["joint"])

    assert out.shape == (1, Z, height, width)
    assert torch.isfinite(out.float()).all()


def test_every_token_shares_one_stream_and_only_the_image_predicts(dit):
    height, width, text = 3, 4, 5
    _positions(dit, height * width, text)
    seen = {"front_end": [], "joint": [], "head": []}
    dit.text_embedder.blocks[0].register_forward_pre_hook(
        lambda _m, a: seen["front_end"].append(a[0].shape[1])
    )
    dit.layers[0].register_forward_pre_hook(lambda _m, a: seen["joint"].append(a[0].shape[1]))
    dit.final_linear_video.register_forward_pre_hook(
        lambda _m, a: seen["head"].append(a[0].shape[1])
    )

    with _compute():
        txt = dit.text_states(_n(1, text, CTX), dit.pe_embedder["txt"])
        dit(_n(1, Z, height, width), torch.full((1,), 0.7), txt, dit.pe_embedder["joint"])

    assert seen["front_end"] == [text]
    assert seen["joint"] == [height * width + text]
    assert seen["head"] == [height * width]


def test_the_refined_text_is_wider_than_the_prompt_it_came_from(dit):
    """``text_states`` takes raw hidden states of ``text_dim``, ``forward`` takes its
    ``dim``-wide output: the two are not interchangeable arguments."""
    _positions(dit, 12, 5)
    with _compute():
        refined = dit.text_states(_n(1, 5, CTX), dit.pe_embedder["txt"])
    assert refined.shape == (1, 5, DIM)


def test_the_timestep_reaches_every_block_as_six_shared_rows(dit):
    height, width, text = 3, 4, 5
    _positions(dit, height * width, text)
    with _compute():
        txt = dit.text_states(_n(1, text, CTX), dit.pe_embedder["txt"])

    seen = []
    dit.shared_block_modulation.register_forward_pre_hook(
        lambda _m, a: seen.append(a[0].clone())
    )
    block_in = []
    dit.layers[0].register_forward_pre_hook(
        lambda _m, a: block_in.append([row.shape for row in a[1]])
    )
    with _compute():
        for step in (0.9, 0.2):
            dit(_n(1, Z, height, width), torch.full((1,), step), txt, dit.pe_embedder["joint"])

    assert len(seen) == 2
    assert seen[0].shape == (1, DIM)
    assert not torch.allclose(seen[0], seen[1])  # the timestep is what moved
    assert [tuple(shape) for shape in block_in[0]] == [(1, 1, DIM)] * 6


def test_a_skipped_joint_block_is_just_a_shorter_residual_chain(dit):
    """Dropping a block changes the answer without shifting any other block's input.

    The blocks share the one modulation row set, so leaving one out consumes no state:
    the run must equal the same run with that block replaced by its own residual.
    """
    height, width, text = 3, 4, 5
    _positions(dit, height * width, text)
    with _compute():
        txt = dit.text_states(_n(1, text, CTX), dit.pe_embedder["txt"])
    latent, t, pe = _n(1, Z, height, width), torch.full((1,), 0.7), dit.pe_embedder["joint"]

    ran, hooks = [], [
        block.register_forward_pre_hook(lambda _m, _a, i=i: ran.append(i))
        for i, block in enumerate(dit.layers)
    ]
    try:
        with _compute():
            skipped = dit(latent, t, txt, pe, [1])
        assert ran == [0]  # block 1 was never entered

        ran.clear()
        real = dit.layers[1].forward
        dit.layers[1].forward = lambda tokens, *rest: tokens  # == its own residual
        with _compute():
            stubbed = dit(latent, t, txt, pe)
        dit.layers[1].forward = real
    finally:
        for hook in hooks:
            hook.remove()

    assert torch.equal(skipped, stubbed)
    with _compute():
        full = dit(latent, t, txt, pe)
    assert not torch.equal(skipped, full)


def test_situ_is_the_soft_capped_gated_activation():
    torch.manual_seed(0)
    hidden = torch.randn(3, 4, dtype=torch.float64)
    gate = torch.randn(3, 4, dtype=torch.float64) * 50
    out = situ(hidden, gate)

    expected = (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (
        25.0 * torch.tanh(hidden / 25.0)
    )
    assert torch.allclose(out, expected)
    assert out.abs().max() <= 100.0 + 1e-9  # both factors are capped before gating


def test_the_joint_blocks_carry_the_situ_cap_on_their_gate():
    """One fewer pass over the ffn-width tensor, and legal only because the projection
    after the activation has no bias for the missing scale to disagree with."""
    torch.manual_seed(0)
    hidden = torch.randn(3, 4, dtype=torch.float64)
    gate = torch.randn(3, 4, dtype=torch.float64) * 50
    assert torch.allclose(situ(hidden, gate), SITU_CAP * situ_unscaled(hidden, gate))

    assert _tiny_dit().layers[0].proj["ffn_down"].bias is None


# ----------------------------------------------------------------------- sampling grid


def test_schedule_is_the_reference_shift_of_a_linear_grid():
    ts = get_schedule(4)
    assert len(ts) == 5
    assert ts[0] == 1.0 and ts[-1] == 0.0  # the shift maps its own endpoints exactly
    assert all(a > b for a, b in zip(ts, ts[1:]))

    raw = torch.linspace(1, 0, 5, dtype=torch.float64).tolist()
    shifted = [(s * FLOW_SHIFT) / (1 + (FLOW_SHIFT - 1) * s) for s in raw]
    assert ts == pytest.approx(shifted)


# -------------------------------------------------------------------------- the adapter


def test_init_latents_uses_the_vae_geometry_and_the_seed():
    model = _bare(vae=FakeVAE())
    latents = model.init_latents(_params(seed=7))

    assert latents.shape == (1, Z, 3, 2)  # one latent cell per 16 pixels
    assert torch.equal(latents, model.init_latents(_params(seed=7)))
    assert not torch.equal(latents, model.init_latents(_params(seed=8)))


def test_encode_prompt_runs_the_negative_branch_only_under_guidance():
    seen = []

    def encode(prompt):
        seen.append(prompt)
        return torch.zeros(1, 6, CTX)

    model = _bare(text_encoder=encode)
    cond = model.encode_prompt(
        EncodePromptArgs(prompt="cat", negative_prompt="blur", guidance_scale=3.0)
    )
    assert seen == ["cat", "blur"]
    assert cond.null is not None

    seen.clear()
    model.encode_prompt(EncodePromptArgs(prompt="cat", guidance_scale=1.0))
    assert seen == ["cat"]


def test_prepare_latent_keeps_the_latent_and_refines_the_text_once(dit):
    """The front-end runs once per branch, never in the loop."""
    model = _bare(dit=dit, vae=FakeVAE())
    cond = Conditioning(cond=torch.randn(1, 5, CTX), null=torch.randn(1, 3, CTX))

    calls = []
    real = dit.text_states
    dit.text_states = lambda *a, **k: calls.append(1) or real(*a, **k)

    x = model.prepare_latent(torch.randn(1, Z, 3, 2), cond, _params())
    assert x.shape == (1, Z, 3, 2)
    assert calls == [1, 1]  # both branches, once each

    model.dit = dit
    with torch.no_grad():
        for i in range(3):
            model.denoise_step(x, torch.tensor(0.7), cond, 3.0, i)
    assert len(calls) == 2


def test_each_branch_gets_its_own_positions_because_the_length_differs(dit):
    """A shorter negative prompt moves the whole image along ``f``, so the branches
    cannot share one table — and neither can be padded."""
    model = _bare(dit=dit, vae=FakeVAE())
    cond = Conditioning(cond=torch.randn(1, 7, CTX), null=torch.randn(1, 3, CTX))
    model.prepare_latent(torch.randn(1, Z, 3, 2), cond, _params())

    tables = dit.pe_embedder._cache
    assert set(tables) == {"joint", "txt", "joint_uncond", "txt_uncond"}
    assert tables["joint"][0].shape[0] == 3 * 2 + 7
    assert tables["joint_uncond"][0].shape[0] == 3 * 2 + 3

    # Same grid, put at a different ``f``: its rotary rows cannot be shared.
    assert not torch.equal(tables["joint"][0][:6], tables["joint_uncond"][0][:6])
    # The text positions start at f = 0 in both, so the short branch is a prefix.
    assert torch.equal(tables["txt"][0][:3], tables["txt_uncond"][0])


def test_denoise_step_does_plain_cfg_over_the_two_branches(dit):
    model = _bare(dit=dit, vae=FakeVAE())
    cond = Conditioning(cond=torch.randn(1, 7, CTX), null=torch.randn(1, 3, CTX))
    x = model.prepare_latent(torch.randn(1, Z, 3, 2), cond, _params())

    seen, outputs = [], []
    real_forward = dit.forward

    def spy(*args, **kwargs):
        seen.append(args)
        outputs.append(real_forward(*args, **kwargs))
        return outputs[-1]

    dit.forward = spy
    with _compute():
        v = model.denoise_step(x, torch.tensor(0.7), cond, 3.0, 1)

    assert len(seen) == 2
    assert seen[0][0] is seen[1][0]  # both branches, same latent
    assert torch.equal(seen[0][1], seen[1][1])  # and the same timestep
    assert [s[2].shape[1] for s in seen] == [7, 3]  # each branch's own text
    assert not torch.equal(seen[0][3][0], seen[1][3][0])  # each branch's own positions

    pos, neg = (out.float() for out in outputs)
    assert v.shape == x.shape
    assert torch.allclose(v.float(), neg + 3.0 * (pos - neg), rtol=2e-2, atol=1e-3)

    seen.clear()
    with _compute():
        model.denoise_step(x, torch.tensor(0.7), cond, 1.0, 1)
    assert len(seen) == 1  # guidance off: one forward


def test_the_negative_branch_runs_one_block_short(dit):
    """Only the uncond forward is shortened; guidance off runs the full stack."""
    # The recipe's own index is past the tiny stack, so the adapter is handed one.
    assert MossImageModel.UNCOND_SKIP_BLOCKS == frozenset({22})
    model = _bare(dit=dit, vae=FakeVAE(), UNCOND_SKIP_BLOCKS=frozenset({1}))
    cond = Conditioning(cond=torch.randn(1, 7, CTX), null=torch.randn(1, 3, CTX))
    x = model.prepare_latent(torch.randn(1, Z, 3, 2), cond, _params())

    seen, real_forward = [], dit.forward

    def spy(*args, **kwargs):
        seen.append(args[4] if len(args) > 4 else kwargs.get("skip_blocks", ()))
        return real_forward(*args, **kwargs)

    dit.forward = spy
    with _compute():
        model.denoise_step(x, torch.tensor(0.7), cond, 3.0, 1)
    assert [tuple(s) for s in seen] == [(), (1,)]  # the prompt keeps every block

    seen.clear()
    with _compute():
        model.denoise_step(x, torch.tensor(0.7), cond, 1.0, 1)
    assert [tuple(s) for s in seen] == [()]  # guidance off: nothing is skipped


def test_guidance_off_is_the_path_the_skip_cannot_reach(dit):
    """Bit-for-bit the same velocity with the skip in place: no uncond, no change."""
    cond = Conditioning(cond=torch.randn(1, 5, CTX), null=torch.randn(1, 3, CTX))
    latent = torch.randn(1, Z, 3, 2)
    outs = []
    # A block the tiny stack actually has, so a skip leaking into the cond branch
    # would show up here rather than hiding past ``len(dit.layers)``.
    for skip in (frozenset(), frozenset({1})):
        model = _bare(dit=dit, vae=FakeVAE(), UNCOND_SKIP_BLOCKS=skip)
        x = model.prepare_latent(latent, cond, _params())
        with _compute():
            outs.append(model.denoise_step(x, torch.tensor(0.7), cond, 1.0, 0))

    assert torch.equal(outs[0], outs[1])


def test_a_single_branch_model_needs_no_uncond_tables(dit):
    model = _bare(dit=dit, vae=FakeVAE())
    model.prepare_latent(torch.randn(1, Z, 3, 2), Conditioning(cond=torch.randn(1, 5, CTX)), _params())

    assert set(dit.pe_embedder._cache) == {"joint", "txt"}
    assert model._txt_states["uncond"] is None


# --------------------------------------------------------------- checkpoint loading


def _checkpoint_state(dit: MossImage) -> dict:
    """The DiT's tensors under the checkpoint's own names."""
    state = {}
    for key, value in dit.state_dict().items():
        key = key.replace(".qk_norm.query_norm.", ".q_norm.").replace(
            ".qk_norm.key_norm.", ".k_norm."
        )
        state[key] = value.clone()
    # Both of these are carried by the exports and belong to no module: they must be
    # dropped rather than matched.
    state["rope.bands"] = torch.ones(1, dtype=torch.float32)
    state["layers.0.comfy_attention.config"] = torch.tensor(list(b"{}"), dtype=torch.uint8)
    return state


def test_a_checkpoint_header_is_enough_to_read_the_geometry(tmp_path):
    path = write_safetensors(tmp_path / "tiny_moss.safetensors", _checkpoint_state(_tiny_dit()))

    params = detect_params(path)
    assert params == MossImageParams(**TINY)


def test_the_checkpoint_loads_back_exactly(tmp_path):
    dit = _tiny_dit()
    path = write_safetensors(tmp_path / "tiny_moss.safetensors", _checkpoint_state(dit))

    loaded = load_moss_image_dit(path, detect_params(path), device="cpu", dtype=torch.float32)

    reference = dit.state_dict()
    for key, value in loaded.state_dict().items():
        assert key in reference, key
        assert torch.allclose(value.float(), reference[key].float(), atol=1e-7), key


def test_an_export_with_the_projections_apart_is_refused(tmp_path):
    dit = _tiny_dit()
    state = _checkpoint_state(dit)
    fused = state.pop("layers.0.proj.qkv.weight")
    rows = fused.shape[0] // 3
    for i, proj in enumerate(("q", "k", "v")):
        state[f"layers.0.proj.{proj}.weight"] = fused[i * rows : (i + 1) * rows]
    path = write_safetensors(tmp_path / "split_moss.safetensors", state)

    with pytest.raises(ValueError, match="projections apart"):
        detect_params(path)


def test_a_modulation_head_that_disagrees_with_dim_is_refused(tmp_path):
    state = _checkpoint_state(_tiny_dit())
    state["video_head_modulation.weight"] = torch.zeros(3, state["video_head_modulation.weight"].shape[1])
    path = write_safetensors(tmp_path / "broken_moss.safetensors", state)

    with pytest.raises(ValueError, match="video_head_modulation"):
        detect_params(path)


def test_loras_naming_the_separate_projections_land_on_the_fused_one():
    """Stacked in the fused matrix's row order, which is the only layout it has."""
    rank, out, in_dim = 2, 4, 3
    parts, deltas = {}, {}
    for i, proj in enumerate(("q", "k", "v")):
        down = torch.arange(rank * in_dim, dtype=torch.float32).reshape(rank, in_dim) + i
        up = torch.arange(out * rank, dtype=torch.float32).reshape(out, rank) + 10 * i
        parts[f"layers.0.proj.{proj}.lora_A.weight"] = down
        parts[f"layers.0.proj.{proj}.lora_B.weight"] = up
        deltas[proj] = up @ down

    fused = _fuse_stacked(parts, tuple(MossImageModel.lora_fusions["qkv"]), "qkv")

    assert set(fused) == {
        "layers.0.proj.qkv.lora_A.weight",
        "layers.0.proj.qkv.lora_B.weight",
    }
    merged = (
        fused["layers.0.proj.qkv.lora_B.weight"]
        @ fused["layers.0.proj.qkv.lora_A.weight"]
    )
    assert torch.allclose(merged, torch.cat([deltas[p] for p in ("q", "k", "v")], dim=0))


# ----------------------------------------------------------------------- text encoder


class _FakeTokenizer:
    """Any text becomes ``length`` tokens; records how it was called."""

    def __init__(self, length=12):
        self.length = length
        self.seen = {}

    def __call__(self, text, add_special_tokens=True, truncation=False, max_length=None):
        self.seen = {
            "text": text,
            "add_special_tokens": add_special_tokens,
            "truncation": truncation,
            "max_length": max_length,
        }
        return {"input_ids": list(range(10, 10 + self.length))}


class _FakeBody:
    def __init__(self, lm):
        self.lm = lm

    def __call__(self, **inputs):
        self.lm.seen = inputs
        length = inputs["input_ids"].shape[1]
        return type(
            "Out", (), {"last_hidden_state": torch.full((1, length, 6), 0.25, dtype=torch.float32)}
        )()


class _FakeLM(torch.nn.Module):
    def __init__(self, hidden=6):
        super().__init__()
        self.probe = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))
        self.hidden, self.seen = hidden, {}
        self.model = _FakeBody(self)


def _fake_encoder(length=12):
    tokenizer = _FakeTokenizer(length)
    lm = _FakeLM()
    return MossImageTextEncoder(lm, tokenizer), tokenizer, lm


def test_the_template_is_a_user_turn_left_open_in_a_thinking_block():
    assert prompt_template("a cat") == (
        "<|im_start|>user\na cat<|im_end|>\n<|im_start|>assistant\n<think>\n"
    )


def test_the_prompt_is_neither_padded_nor_special_token_stripped():
    encoder, tokenizer, lm = _fake_encoder(17)
    with torch.no_grad():
        embeds = encoder.encode("a cat")

    assert tokenizer.seen["text"] == prompt_template("a cat")
    assert tokenizer.seen["add_special_tokens"] is False  # the template IS the sequence
    assert tokenizer.seen["truncation"] is True
    assert tokenizer.seen["max_length"] == MAX_SEQUENCE_LENGTH

    assert lm.seen["input_ids"].shape == (1, 17)
    assert torch.equal(lm.seen["attention_mask"], torch.ones(1, 17, dtype=torch.long))
    assert lm.seen["use_cache"] is False
    assert embeds.shape == (1, 17, 6)
    assert embeds.dtype is torch.bfloat16
