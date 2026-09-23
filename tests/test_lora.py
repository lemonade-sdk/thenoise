"""LoRA loading, naming, fusing, folding, applying and undoing.

All of it is pure, tiny and CPU-only — and it is the code whose failure mode is
"the LoRA silently does nothing", i.e. invisible to the user. Covers the
spec/path helpers on the adapter, the naming-convention resolution, the stacked
projections and ``DiffusionModel.switch_loras``.
"""
from __future__ import annotations

import os

import pytest
import torch
import torch.nn as nn

from conftest import StubModel, write_safetensors
from thenoise.dit.quantized import QuantizedLinear
from thenoise.models.flux_klein import FluxKleinModel
from thenoise.utils.lora import (
    FUSE_GATE_UP,
    FUSE_QKV,
    LoraMode,
    _fold,
    _fuse_stacked,
    _match_lora_keys,
    _normalize_lora_suffix,
    _projection_scale,
    apply_lora_to_model,
    undo_lora_on_model,
)

def _fuse(lora_sd, spec):
    """Apply a fusion spec (``{fused: parts}``) as ``_normalize_lora_sd`` does."""
    for fused, parts in spec.items():
        lora_sd = _fuse_stacked(lora_sd, tuple(parts), fused)
    return lora_sd


def _delta(down, up, scale: float = 1.0) -> torch.Tensor:
    """The weight delta of a factor pair carrying ``alpha / rank * strength``."""
    return (up * scale) @ down


# ------------------------------------------------- LoRA specs and the lora dir


@pytest.mark.parametrize(
    "spec,expected",
    [
        ("style", ("style.safetensors", 1.0)),          # suffix auto-appended
        ("style:0.8", ("style.safetensors", 0.8)),
        ("sub/style:0.7", ("sub/style.safetensors", 0.7)),
        ("already.safetensors", ("already.safetensors", 1.0)),
    ],
)
def test_parse_lora_spec(spec, expected):
    model = object.__new__(StubModel)
    model.lora_dir = "/tmp/loras"
    assert model._parse_lora_spec(spec) == expected


@pytest.mark.parametrize(
    "filename,ok",
    [
        ("style.safetensors", True),
        ("sub/style.safetensors", True),
        ("../etc/passwd", False),          # escapes the base directory
        ("sub/../../etc/passwd", False),
    ],
)
def test_resolve_lora_path_is_confined_to_lora_dir(tmp_path, filename, ok):
    model = object.__new__(StubModel)
    model.lora_dir = str(tmp_path)

    if ok:
        path = model._resolve_lora_path(filename)
        assert path == os.path.join(str(tmp_path), filename)
    else:
        with pytest.raises(ValueError, match="escapes base directory"):
            model._resolve_lora_path(filename)


def test_resolve_lora_path_rejects_absolute_paths(tmp_path):
    """The guard is on the resolved path, so an absolute path can't sneak in."""
    model = object.__new__(StubModel)
    model.lora_dir = str(tmp_path)

    with pytest.raises(ValueError, match="escapes base directory"):
        model._resolve_lora_path("/etc/passwd")


def test_resolve_lora_path_requires_a_lora_dir():
    model = object.__new__(StubModel)
    model.lora_dir = ""
    with pytest.raises(ValueError, match="base directory is not set"):
        model._resolve_lora_path("style.safetensors")


def test_list_loras_returns_sorted_short_names_recursive(tmp_path):
    import tempfile

    model = object.__new__(StubModel)

    for rel in [
        "12345_something.safetensors",
        "67890_other.safetensors",
        "sub/style.safetensors",
        "not_a_lora.txt",
    ]:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")

    model.lora_dir = str(tmp_path)
    assert model.list_loras() == ["12345_something", "67890_other", "sub/style"]

    model.lora_dir = ""
    assert model.list_loras() == []
    assert model._make_lora_spec_hash(None) == "__none__"


def test_lora_spec_hash_is_order_insensitive():
    model = object.__new__(StubModel)
    assert model._make_lora_spec_hash([]) == "__none__"
    assert model._make_lora_spec_hash(["a:0.5", "b:1.0"]) == model._make_lora_spec_hash(
        ["b:1.0", "a:0.5"]
    )
    assert model._make_lora_spec_hash(["a:0.5"]) != model._make_lora_spec_hash(["a:0.6"])


# ------------------------------------------------------------- suffix spellings


@pytest.mark.parametrize(
    "raw,canonical",
    [
        ("blocks.0.attn.lora_down.weight", "blocks.0.attn.lora_A.weight"),
        ("blocks.0.attn.lora_up.weight", "blocks.0.attn.lora_B.weight"),
        ("blocks.0.attn.lora.down.weight", "blocks.0.attn.lora_A.weight"),
        ("blocks.0.attn.lora.up.weight", "blocks.0.attn.lora_B.weight"),
        # PEFT puts the adapter name between the factor and the leaf, and it is
        # whatever the trainer called the adapter, not always "default".
        ("blocks.0.attn.to_q.lora_A.default.weight", "blocks.0.attn.to_q.lora_A.weight"),
        ("blocks.0.attn.to_q.lora_B.default.weight", "blocks.0.attn.to_q.lora_B.weight"),
        ("blocks.0.attn.to_q.lora_A.my_adapter.weight", "blocks.0.attn.to_q.lora_A.weight"),
        # Already canonical (diffusers) forms pass through untouched.
        ("blocks.0.attn.lora_A.weight", "blocks.0.attn.lora_A.weight"),
        ("blocks.0.attn.lora_B.weight", "blocks.0.attn.lora_B.weight"),
        # Alphas and unrelated leaves are left alone.
        ("blocks.0.attn.alpha", "blocks.0.attn.alpha"),
        ("blocks.0.attn.bias", "blocks.0.attn.bias"),
        ("blocks.0.attn.base_layer.weight", "blocks.0.attn.base_layer.weight"),
    ],
)
def test_normalize_lora_suffix(raw, canonical):
    out = _normalize_lora_suffix({raw: torch.ones(1)})
    assert list(out) == [canonical]


def test_normalize_lora_suffix_collapses_a_whole_peft_state_dict():
    """The PEFT form of a factor pair ends up matchable as one target."""
    out = _normalize_lora_suffix(
        {
            "blocks.0.attn.to_q.lora_A.default.weight": torch.ones(1),
            "blocks.0.attn.to_q.lora_B.default.weight": torch.ones(1),
        }
    )
    assert set(out) == {
        "blocks.0.attn.to_q.lora_A.weight",
        "blocks.0.attn.to_q.lora_B.weight",
    }
    assert _match_lora_keys("blocks.0.attn.to_q.weight", set(out)) is not None


def test_normalize_lora_suffix_rewrites_a_whole_state_dict():
    out = _normalize_lora_suffix(
        {
            "x.lora_down.weight": torch.ones(1),
            "x.lora_up.weight": torch.ones(1),
            "x.alpha": torch.ones(1),
        }
    )
    assert set(out) == {"x.lora_A.weight", "x.lora_B.weight", "x.alpha"}


# ---------------------------------------------------------- naming conventions

_TARGET = "blocks.0.attn.k_proj"

# Convention -> the LoRA target name used by the training tool.
NAMING_CONVENTIONS = {
    "sd-scripts": f"lora_unet_{_TARGET.replace('.', '_')}",
    "underscored-bare": _TARGET.replace(".", "_"),
    "dotted-bare": _TARGET,
    "diffusion_model-prefixed": f"diffusion_model.{_TARGET}",
    "transformer-prefixed": f"transformer.{_TARGET}",
}


@pytest.mark.parametrize("name", NAMING_CONVENTIONS.values(), ids=NAMING_CONVENTIONS)
def test_match_lora_keys_resolves_every_convention(name):
    keys = {f"{name}.lora_A.weight", f"{name}.lora_B.weight", f"{name}.alpha"}
    assert _match_lora_keys(f"{_TARGET}.weight", keys) == (
        f"{name}.lora_A.weight",
        f"{name}.lora_B.weight",
        f"{name}.alpha",
    )


@pytest.mark.parametrize(
    "key,keys",
    [
        # Only weights can carry a LoRA.
        (f"{_TARGET}.bias", {f"{_TARGET}.lora_A.weight", f"{_TARGET}.lora_B.weight"}),
        # Half a factor pair is not a LoRA.
        (f"{_TARGET}.weight", {f"{_TARGET}.lora_A.weight"}),
        # A LoRA for a different layer must not be borrowed.
        (f"{_TARGET}.weight", {"blocks.0.attn.q_proj.lora_A.weight",
                               "blocks.0.attn.q_proj.lora_B.weight"}),
    ],
    ids=["non-weight-key", "incomplete-pair", "unrelated-layer"],
)
def test_match_lora_keys_returns_none(key, keys):
    assert _match_lora_keys(key, keys) is None


# ------------------------------------------------------------ attention fusion


def _qkv_lora(
    rank: int = 2,
    in_f: int = 4,
    out_f: int = 6,
    prefix: str = "blocks.0.attn.",
    alphas: dict | None = None,
    present: tuple = ("q", "k", "v"),
) -> dict:
    """Separate to_q/to_k/to_v factors, optionally with per-projection alphas."""
    keys = {}
    for i, which in enumerate(("q", "k", "v")):
        if which not in present:
            continue
        keys[f"{prefix}to_{which}.lora_A.weight"] = torch.arange(
            rank * in_f, dtype=torch.float32
        ).reshape(rank, in_f) + i
        keys[f"{prefix}to_{which}.lora_B.weight"] = torch.arange(
            out_f * rank, dtype=torch.float32
        ).reshape(out_f, rank) + 10 * i
        alpha = None if alphas is None else alphas.get(which)
        if alpha is not None:
            keys[f"{prefix}to_{which}.alpha"] = torch.tensor(float(alpha))
    return keys


def test_fuse_qkv_builds_the_qkv_pair():
    sd = _qkv_lora()
    fused = _fuse(sd, FUSE_QKV)

    assert not any("to_q" in k or "to_k" in k or "to_v" in k for k in fused)
    a = fused["blocks.0.attn.qkv.lora_A.weight"]
    b = fused["blocks.0.attn.qkv.lora_B.weight"]
    # A = cat(dim=0) -> (3r, in); B = block_diag(B_q, B_k, B_v) -> (out, 3r) with
    # out = out_q + out_k + out_v, i.e. the fused projection's output width.
    assert a.shape == (6, 4)
    assert b.shape == (18, 6)

    for i, which in enumerate("qkv"):
        assert torch.equal(a[2 * i : 2 * i + 2], sd[f"blocks.0.attn.to_{which}.lora_A.weight"])
        assert torch.equal(
            b[6 * i : 6 * i + 6, 2 * i : 2 * i + 2],
            sd[f"blocks.0.attn.to_{which}.lora_B.weight"],
        )
        # Off-diagonal blocks are exactly zero (q must not leak into k/v rows).
        for j in range(3):
            if j != i:
                assert torch.equal(b[6 * i : 6 * i + 6, 2 * j : 2 * j + 2], torch.zeros(6, 2))


def test_fuse_qkv_fused_delta_equals_the_stack_of_projection_deltas():
    """The fused rank-3r delta is the stack of the three separate merges.

    That equality is the whole reason the fusion is correct: the three separate
    LoRAs, each applied with ``r/r == 1``, are reproduced by one application on
    the fused projection.
    """
    sd = _qkv_lora()
    fused = _fuse(sd, FUSE_QKV)
    per_projection = [
        _delta(sd[f"blocks.0.attn.to_{w}.lora_A.weight"],
               sd[f"blocks.0.attn.to_{w}.lora_B.weight"])
        for w in "qkv"
    ]
    fused_delta = _delta(
        fused["blocks.0.attn.qkv.lora_A.weight"],
        fused["blocks.0.attn.qkv.lora_B.weight"],
    )
    assert torch.allclose(fused_delta, torch.cat(per_projection, dim=0), rtol=1e-6)


def test_fuse_qkv_folds_each_projection_alpha():
    """Per-projection ``alpha/rank`` must survive the fusion (ComfyUI semantics).

    ComfyUI applies each projection separately with its own ``alpha/rank``; a
    fused pair carries a single rank (``3r``), so the scales are folded into the
    factors. Without that, a LoRA with ``alpha != rank`` silently comes out at
    the wrong strength (the fused default ``3r/3r`` is always 1).
    """
    alphas = {"q": 8.0, "k": 2.0, "v": None}  # rank 2 -> scales 4.0, 1.0, 1.0
    sd = _qkv_lora(alphas=alphas)
    fused = _fuse(sd, FUSE_QKV)

    per_projection = [
        _delta(
            sd[f"blocks.0.attn.to_{w}.lora_A.weight"],
            sd[f"blocks.0.attn.to_{w}.lora_B.weight"],
            (2.0 if alphas[w] is None else alphas[w]) / 2.0,
        )
        for w in "qkv"
    ]
    fused_delta = _delta(
        fused["blocks.0.attn.qkv.lora_A.weight"],
        fused["blocks.0.attn.qkv.lora_B.weight"],
    )
    assert torch.allclose(fused_delta, torch.cat(per_projection, dim=0), rtol=1e-6)

    # The consumed alphas must not resurface as "unused keys" warnings.
    assert not [k for k in fused if k.endswith(".alpha")]


def test_fuse_qkv_pads_missing_projections_with_zeros():
    """A q-only attention LoRA lands on the q rows instead of raising KeyError."""
    for present, alphas in ((("q",), None), (("k", "v"), {"k": 4.0, "v": 2.0})):
        sd = _qkv_lora(alphas=alphas, present=present)
        fused = _fuse(sd, FUSE_QKV)
        a = fused["blocks.0.attn.qkv.lora_A.weight"]
        b = fused["blocks.0.attn.qkv.lora_B.weight"]
        # A keeps only the trained ranks; B always spans the full fused qkv height.
        assert a.shape == (2 * len(present), 4)
        assert b.shape == (18, 2 * len(present))

        delta = _delta(a, b)
        assert delta.shape == (18, 4)
        for i, w in enumerate("qkv"):
            if w in present:
                alpha = 2.0 if alphas is None or alphas.get(w) is None else alphas[w]
                expected = _delta(
                    sd[f"blocks.0.attn.to_{w}.lora_A.weight"],
                    sd[f"blocks.0.attn.to_{w}.lora_B.weight"],
                    alpha / 2.0,
                )
                assert torch.allclose(delta[6 * i : 6 * i + 6], expected, rtol=1e-6)
            else:
                assert torch.equal(delta[6 * i : 6 * i + 6], torch.zeros(6, 4))


def test_fuse_qkv_skips_an_unlayable_subset(caplog):
    """q+k with different output widths: the missing v slice is unknowable.

    Better an explicit warning and untouched keys (reported unused) than a delta
    written to the wrong rows of the fused projection.
    """
    sd = _qkv_lora(present=("q", "k"))
    sd["blocks.0.attn.to_k.lora_B.weight"] = torch.ones(3, 2)
    fused = _fuse(sd, FUSE_QKV)

    assert set(fused) == set(sd)
    assert not any("qkv" in k for k in fused)
    assert "skipping qkv fusion" in caplog.text


@pytest.mark.parametrize(
    "alpha,rank,expected",
    [
        (None, 8, 1.0),   # no alpha key -> ComfyUI's hardcoded 1.0 == rank/rank
        (16.0, 8, 2.0),
        (4.0, 8, 0.5),
        (0.0, 8, 0.0),    # alpha 0 disables the LoRA in ComfyUI too
    ],
)
def test_projection_scale_matches_comfy_alpha_rule(alpha, rank, expected):
    sd = {} if alpha is None else {"x.alpha": torch.tensor(alpha)}
    assert _projection_scale(sd, "x.alpha", rank) == expected


def test_fuse_qkv_is_a_noop_without_qkv_and_does_not_mutate():
    sd = {"blocks.0.ff.net.0.lora_A.weight": torch.ones(2, 4), "blocks.0.ff.net.0.lora_B.weight": torch.ones(6, 2)}
    snapshot = dict(sd)
    assert _fuse(sd, FUSE_QKV) == snapshot
    assert sd == snapshot


# ------------------------------------------------------- SwiGLU gate/up fusion


def _gate_up_lora(
    rank: int = 2,
    in_f: int = 4,
    hidden: int = 6,
    prefix: str = "blocks.0.img_mlp.",
    alphas: dict | None = None,
    present: tuple = ("gate_layer", "proj"),
) -> dict:
    """Separate SwiGLU ``gate_layer``/``proj`` factors, optionally with alphas."""
    keys = {}
    for i, which in enumerate(("gate_layer", "proj")):
        if which not in present:
            continue
        keys[f"{prefix}{which}.lora_A.weight"] = torch.arange(
            rank * in_f, dtype=torch.float32
        ).reshape(rank, in_f) + i
        keys[f"{prefix}{which}.lora_B.weight"] = torch.arange(
            hidden * rank, dtype=torch.float32
        ).reshape(hidden, rank) + 10 * i
        alpha = None if alphas is None else alphas.get(which)
        if alpha is not None:
            keys[f"{prefix}{which}.alpha"] = torch.tensor(float(alpha))
    return keys


def test_fuse_gate_up_builds_the_gate_up_pair():
    sd = _gate_up_lora()
    fused = _fuse(sd, FUSE_GATE_UP)

    assert not any("gate_layer" in k or ".proj." in k for k in fused)
    a = fused["blocks.0.img_mlp.gate_up.lora_A.weight"]
    b = fused["blocks.0.img_mlp.gate_up.lora_B.weight"]
    # The fused matrix is [gate; up], so A is the rank stack and B the block
    # diagonal at each half's row offset.
    assert a.shape == (4, 4)
    assert b.shape == (12, 4)
    for i, which in enumerate(("gate_layer", "proj")):
        assert torch.equal(a[2 * i : 2 * i + 2], sd[f"blocks.0.img_mlp.{which}.lora_A.weight"])
        assert torch.equal(
            b[6 * i : 6 * i + 6, 2 * i : 2 * i + 2],
            sd[f"blocks.0.img_mlp.{which}.lora_B.weight"],
        )
        for j in range(2):
            if j != i:
                assert torch.equal(b[6 * i : 6 * i + 6, 2 * j : 2 * j + 2], torch.zeros(6, 2))


def test_fuse_gate_up_fused_delta_is_the_stack_of_the_two_merges():
    """Applying the fused pair once == applying gate and up on their own halves."""
    sd = _gate_up_lora()
    fused = _fuse(sd, FUSE_GATE_UP)

    per_part = [
        _delta(sd[f"blocks.0.img_mlp.{which}.lora_A.weight"],
               sd[f"blocks.0.img_mlp.{which}.lora_B.weight"])
        for which in ("gate_layer", "proj")
    ]
    fused_delta = _delta(
        fused["blocks.0.img_mlp.gate_up.lora_A.weight"],
        fused["blocks.0.img_mlp.gate_up.lora_B.weight"],
    )
    assert torch.allclose(fused_delta, torch.cat(per_part, dim=0), rtol=1e-6)


def test_fuse_gate_up_folds_each_part_alpha():
    """The fused rank-4 pair must keep the two parts' own ``alpha/rank`` scales."""
    alphas = {"gate_layer": 8.0, "proj": 1.0}  # rank 2 -> scales 4.0 and 0.5
    sd = _gate_up_lora(alphas=alphas)
    fused = _fuse(sd, FUSE_GATE_UP)

    per_part = [
        _delta(
            sd[f"blocks.0.img_mlp.{which}.lora_A.weight"],
            sd[f"blocks.0.img_mlp.{which}.lora_B.weight"],
            alphas[which] / 2.0,
        )
        for which in ("gate_layer", "proj")
    ]
    fused_delta = _delta(
        fused["blocks.0.img_mlp.gate_up.lora_A.weight"],
        fused["blocks.0.img_mlp.gate_up.lora_B.weight"],
    )
    assert torch.allclose(fused_delta, torch.cat(per_part, dim=0), rtol=1e-6)
    assert not [k for k in fused if k.endswith(".alpha")]


def test_fuse_gate_up_pads_a_missing_part_with_zeros():
    """A gate-only LoRA lands on the gate half instead of the whole matrix."""
    sd = _gate_up_lora(present=("gate_layer",))
    fused = _fuse(sd, FUSE_GATE_UP)

    a = fused["blocks.0.img_mlp.gate_up.lora_A.weight"]
    b = fused["blocks.0.img_mlp.gate_up.lora_B.weight"]
    assert a.shape == (2, 4)
    assert b.shape == (12, 2)

    delta = _delta(a, b)
    assert torch.equal(delta[6:], torch.zeros(6, 4))
    assert torch.allclose(
        delta[:6],
        _delta(
            sd["blocks.0.img_mlp.gate_layer.lora_A.weight"],
            sd["blocks.0.img_mlp.gate_layer.lora_B.weight"],
        ),
        rtol=1e-6,
    )


def test_fuse_gate_up_skips_an_unlayable_subset(caplog):
    """Parts disagreeing on the input dim cannot share one fused ``A``."""
    sd = _gate_up_lora()
    sd["blocks.0.img_mlp.proj.lora_A.weight"] = torch.ones(2, 5)
    fused = _fuse(sd, FUSE_GATE_UP)

    assert set(fused) == set(sd)
    assert not any("gate_up" in k for k in fused)
    assert "skipping gate_up fusion" in caplog.text


def test_fuse_gate_up_is_a_noop_for_qkv_factors_and_vice_versa():
    """The two stackings never touch each other's keys."""
    qkv, gate = _qkv_lora(), _gate_up_lora()
    assert _fuse(qkv, FUSE_GATE_UP) == qkv
    assert _fuse(gate, FUSE_QKV) == gate


# ----------------------------------------------------------------- apply/undo


class _TinyNet(nn.Module):
    """Two LoRA targets: a projection and an attention layer, both BF16."""

    def __init__(self):
        super().__init__()
        self.proj = QuantizedLinear(4, 6, bias=False)
        self.blocks = nn.ModuleList([nn.Module()])
        self.blocks[0].attn = QuantizedLinear(4, 6, bias=False)
        # Integer-valued weights keep the bf16 add/subtract round-trip exact.
        with torch.no_grad():
            self.proj.weight.fill_(1.0)
            self.blocks[0].attn.weight.fill_(2.0)


def _int_lora(target: str, rank: int = 2, in_f: int = 4, out_f: int = 6) -> dict:
    """A LoRA with integer factors and ``alpha == rank`` (scale exactly 1)."""
    return {
        f"{target}.lora_down.weight": torch.ones(rank, in_f),
        f"{target}.lora_up.weight": torch.ones(out_f, rank),
        f"{target}.alpha": torch.tensor(float(rank)),
    }


def test_bf16_apply_then_undo_restores_weights_bit_exactly():
    model = _TinyNet()
    original = {k: v.clone() for k, v in model.state_dict().items()}

    result = apply_lora_to_model(model, [_int_lora("proj")], [1.0])
    assert result["targets"]["proj"].mode is LoraMode.BAKED
    assert not torch.equal(model.proj.weight, original["proj.weight"])

    undo_lora_on_model(model, result)
    for key, tensor in model.state_dict().items():
        assert torch.equal(tensor, original[key]), f"{key} not restored exactly"


def test_two_loras_on_one_weight_accumulate_into_a_single_delta():
    model = _TinyNet()
    original = model.proj.weight.clone()

    first, second = _int_lora("proj"), _int_lora("proj")
    result = apply_lora_to_model(model, [first, second], [1.0, 1.0])
    stacked = model.proj.weight.clone()

    # One LoRA with the doubled multiplier is the same delta (both are 1s).
    single = _TinyNet()
    apply_lora_to_model(single, [_int_lora("proj")], [2.0])
    assert torch.equal(stacked, single.proj.weight)

    undo_lora_on_model(model, result)
    assert torch.equal(model.proj.weight, original)


def test_apply_reports_and_skips_unused_keys(caplog):
    model = _TinyNet()
    sd = _int_lora("proj")
    sd["nowhere.at.all.lora_A.weight"] = torch.ones(2, 4)
    sd["nowhere.at.all.lora_B.weight"] = torch.ones(6, 2)

    result = apply_lora_to_model(model, [sd], [1.0])
    assert set(result["targets"]) == {"proj"}
    assert "unused keys" in caplog.text


# -------------------------------------------------- the model's fusion spec


class _SwigluNet(nn.Module):
    """One fused ``gate_up`` row — the Qwen-Image 2.1 SwiGLU layout."""

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Module()])
        self.blocks[0].img_mlp = nn.Module()
        self.blocks[0].img_mlp.gate_up = QuantizedLinear(4, 12, bias=False)
        with torch.no_grad():
            self.blocks[0].img_mlp.gate_up.weight.fill_(1.0)


def _peft_gate_up_lora() -> dict:
    """A PEFT-named LoRA on the split SwiGLU halves (gate delta 2, up delta 8)."""
    return {
        "blocks.0.img_mlp.gate_layer.lora_A.default.weight": torch.ones(2, 4),
        "blocks.0.img_mlp.gate_layer.lora_B.default.weight": torch.ones(6, 2),
        "blocks.0.img_mlp.proj.lora_A.default.weight": 2 * torch.ones(2, 4),
        "blocks.0.img_mlp.proj.lora_B.default.weight": 2 * torch.ones(6, 2),
    }


def test_a_declared_fusion_lands_a_split_lora_on_the_fused_weight():
    """PEFT naming + the model's ``gate_up`` spec == the stack of the two halves."""
    model = _SwigluNet()
    original = model.blocks[0].img_mlp.gate_up.weight.clone()

    result = apply_lora_to_model(
        model, [_peft_gate_up_lora()], [1.0], fusions=FUSE_GATE_UP
    )
    assert set(result["targets"]) == {"blocks.0.img_mlp.gate_up"}
    expected = torch.cat([2 * torch.ones(6, 4), 8 * torch.ones(6, 4)], dim=0)
    assert torch.equal(model.blocks[0].img_mlp.gate_up.weight, original + expected)

    undo_lora_on_model(model, result)
    assert torch.equal(model.blocks[0].img_mlp.gate_up.weight, original)


def test_an_undeclared_fusion_is_not_applied(caplog):
    """A model that runs the split projections must never get its LoRA stacked.

    The spec is opt-in per adapter, so here the factors stay two unmatched names:
    reported, and nothing written to the wrong rows.
    """
    model = _SwigluNet()
    original = model.blocks[0].img_mlp.gate_up.weight.clone()

    result = apply_lora_to_model(model, [_peft_gate_up_lora()], [1.0])
    assert result["targets"] == {}
    assert torch.equal(model.blocks[0].img_mlp.gate_up.weight, original)
    assert "unused keys" in caplog.text


def test_switch_loras_uses_the_model_fusion_spec(tmp_path):
    """The adapter's ``lora_fusions`` is what reaches ``apply_lora_to_model``."""

    class _FusedModel(StubModel):
        lora_fusions = FUSE_GATE_UP

    model = _FusedModel(lora_dir=str(tmp_path))
    write_safetensors(tmp_path / "peft.safetensors", _peft_gate_up_lora())
    dit = _SwigluNet()
    original = dit.blocks[0].img_mlp.gate_up.weight.clone()

    model.switch_loras(["peft.safetensors:1.0"], dit)
    assert not torch.equal(dit.blocks[0].img_mlp.gate_up.weight, original)

    model.switch_loras(None, dit)
    assert torch.equal(dit.blocks[0].img_mlp.gate_up.weight, original)


def test_apply_with_no_loras_returns_an_empty_result():
    result = apply_lora_to_model(_TinyNet(), [], [])
    assert result["targets"] == {}
    # Undoing it is a no-op rather than an error.
    undo_lora_on_model(_TinyNet(), result)


# ----------------------------------------------------------- switch_loras()


@pytest.fixture
def switch_model(tmp_path, monkeypatch):
    """A stub adapter over a spy-instrumented apply/undo pair."""
    import thenoise.models.base as model_base

    events = []
    real_apply, real_undo = model_base.apply_lora_to_model, model_base.undo_lora_on_model

    def spy_apply(model, lora_sds, multipliers, **kwargs):
        events.append(("apply", len(lora_sds)))
        return real_apply(model, lora_sds, multipliers, **kwargs)

    def spy_undo(model, result):
        events.append(("undo", None))
        return real_undo(model, result)

    monkeypatch.setattr(model_base, "apply_lora_to_model", spy_apply)
    monkeypatch.setattr(model_base, "undo_lora_on_model", spy_undo)

    model = StubModel(lora_dir=str(tmp_path))
    write_safetensors(tmp_path / "style.safetensors", _int_lora("proj"))
    write_safetensors(tmp_path / "pose.safetensors", _int_lora("blocks.0.attn"))
    return model, events


def test_switch_loras_is_a_noop_for_the_same_spec(switch_model):
    model, events = switch_model
    dit = _TinyNet()

    model.switch_loras(["style.safetensors:1.0"], dit)
    assert [e[0] for e in events] == ["apply"]
    model.switch_loras(["style.safetensors:1.0"], dit)
    assert [e[0] for e in events] == ["apply"]  # not re-applied


def test_switch_loras_undoes_the_previous_spec_first(switch_model):
    model, events = switch_model
    dit = _TinyNet()

    model.switch_loras(["style.safetensors:1.0"], dit)
    model.switch_loras(["pose.safetensors:1.0"], dit)
    assert [e[0] for e in events] == ["apply", "undo", "apply"]

    # Switching back to base undoes and applies nothing.
    model.switch_loras(None, dit)
    assert [e[0] for e in events] == ["apply", "undo", "apply", "undo"]


def test_switch_loras_restores_the_base_weights(switch_model):
    model, _ = switch_model
    dit = _TinyNet()
    original = dit.proj.weight.clone()

    model.switch_loras(["style.safetensors:1.0"], dit)
    assert not torch.equal(dit.proj.weight, original)
    model.switch_loras(None, dit)
    assert torch.equal(dit.proj.weight, original)


def test_switch_loras_honours_the_model_key_map(tmp_path):
    """A ComfyUI-named Flux.2 LoRA lands on this repo's module names."""

    class _FluxKeyMapModel(StubModel):
        # The real Flux Klein schema-rename table.
        _lora_key_map = FluxKleinModel._lora_key_map

    model = _FluxKeyMapModel(lora_dir=str(tmp_path))

    class _Fluxish(nn.Module):
        def __init__(self):
            super().__init__()
            self.single_blocks = nn.ModuleList([nn.Module()])
            self.single_blocks[0].linear1 = QuantizedLinear(4, 6, bias=False)
            with torch.no_grad():
                self.single_blocks[0].linear1.weight.fill_(1.0)

    dit = _Fluxish()
    base = torch.ones(6, 4)
    # Saved with the ComfyUI names (single_transformer_blocks/attn.to_qkv_mlp_proj).
    write_safetensors(
        tmp_path / "comfy.safetensors",
        _int_lora("single_transformer_blocks.0.attn.to_qkv_mlp_proj"),
    )

    model.switch_loras(["comfy.safetensors:1.0"], dit)
    assert not torch.equal(dit.single_blocks[0].linear1.weight, base)

    model.switch_loras(None, dit)
    assert torch.equal(dit.single_blocks[0].linear1.weight, base)


def test_switch_loras_without_a_lora_dir_does_not_apply(tmp_path):
    model = StubModel(lora_dir=None)
    dit = _TinyNet()
    original = dit.proj.weight.clone()

    model.switch_loras(["style.safetensors:1.0"], dit)
    assert torch.equal(dit.proj.weight, original)


# ------------------------------------------------- folding several LoRAs


def test_fold_sums_several_loras_on_one_target():
    """The folded pair's delta is the sum of the individual scaled deltas."""
    torch.manual_seed(0)
    pairs = [
        (torch.randn(2, 8), torch.randn(6, 2), 1.0),
        (torch.randn(4, 8), torch.randn(6, 4), 0.25),
    ]
    folded = _fold(pairs)
    assert folded.down.shape == (6, 8)
    assert folded.up.shape == (6, 6)
    assert torch.allclose(
        folded.delta(), sum(_delta(d, u, s) for d, u, s in pairs), rtol=2e-2, atol=2e-2
    )


def test_fold_passes_a_single_unscaled_pair_through():
    """No alpha, full strength: the factors reach the layer bit-identically."""
    down, up = torch.randn(3, 8), torch.randn(6, 3)
    assert _fold([(down, up, 1.0)]).down is down
    assert _fold([(down, up, 1.0)]).up is up
