"""LoRA loading, naming, fusing, applying and undoing.

All of it is pure, tiny and CPU-only — and it is the code whose failure mode is
"the LoRA silently does nothing", i.e. invisible to the user. Covers the
spec/path helpers on the adapter, the naming-convention resolution and attention
fusion in ``thenoise.utils.lora``, the bf16 apply→undo round-trip and
``DiffusionModel.switch_loras``.
"""
from __future__ import annotations

import os

import pytest
import torch
import torch.nn as nn

from conftest import StubModel, write_safetensors
from thenoise.models.flux_klein import FluxKleinModel
from thenoise.utils.lora import (
    _fuse_attention,
    _match_lora_keys,
    _normalize_lora_suffix,
    apply_lora_to_model,
    compute_lora_delta,
    undo_lora_on_model,
)

CPU = torch.device("cpu")


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
        # Already canonical (diffusers) forms pass through untouched.
        ("blocks.0.attn.lora_A.weight", "blocks.0.attn.lora_A.weight"),
        ("blocks.0.attn.lora_B.weight", "blocks.0.attn.lora_B.weight"),
        # Alphas and unrelated leaves are left alone.
        ("blocks.0.attn.alpha", "blocks.0.attn.alpha"),
        ("blocks.0.attn.bias", "blocks.0.attn.bias"),
    ],
)
def test_normalize_lora_suffix(raw, canonical):
    out = _normalize_lora_suffix({raw: torch.ones(1)})
    assert list(out) == [canonical]


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


def _qkv_lora(rank=2, in_f=4, out_f=6, prefix="blocks.0.attn."):
    keys = {}
    for i, which in enumerate(("q", "k", "v")):
        keys[f"{prefix}to_{which}.lora_A.weight"] = torch.arange(
            rank * in_f, dtype=torch.float32
        ).reshape(rank, in_f) + i
        keys[f"{prefix}to_{which}.lora_B.weight"] = torch.arange(
            out_f * rank, dtype=torch.float32
        ).reshape(out_f, rank) + 10 * i
    return keys


def test_fuse_attention_builds_the_qkv_pair():
    sd = _qkv_lora()
    fused = _fuse_attention(sd)

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


def test_fuse_attention_fused_delta_equals_the_stack_of_projection_deltas():
    """The fused rank-3r delta with ``alpha = 3r`` scales to 1 and is the stack.

    That equality is the whole reason the fusion is correct: the three separate
    LoRAs, each applied with ``r/r == 1``, are reproduced by one application on
    the fused projection (whose ``alpha`` defaults to ``down.size(0) == 3r``, also
    a scale of 1).
    """
    sd = _qkv_lora()
    fused = _fuse_attention(sd)
    per_projection = [
        compute_lora_delta(
            sd[f"blocks.0.attn.to_{w}.lora_A.weight"],
            sd[f"blocks.0.attn.to_{w}.lora_B.weight"],
            2.0,  # per-projection alpha == rank
            1.0,
            CPU,
        )
        for w in "qkv"
    ]
    fused_delta = compute_lora_delta(
        fused["blocks.0.attn.qkv.lora_A.weight"],
        fused["blocks.0.attn.qkv.lora_B.weight"],
        6.0,  # == down.size(0) == 3r -> scale 1
        1.0,
        CPU,
    )
    assert torch.equal(fused_delta, torch.cat(per_projection, dim=0))


def test_fuse_attention_is_a_noop_without_qkv_and_does_not_mutate():
    sd = {"blocks.0.ff.net.0.lora_A.weight": torch.ones(2, 4), "blocks.0.ff.net.0.lora_B.weight": torch.ones(6, 2)}
    snapshot = dict(sd)
    assert _fuse_attention(sd) == snapshot
    assert sd == snapshot


# ----------------------------------------------------------------- apply/undo


class _TinyNet(nn.Module):
    """Two LoRA targets: a plain linear and an attention projection."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 6, bias=False)
        self.blocks = nn.ModuleList([nn.Module()])
        self.blocks[0].attn = nn.Linear(4, 6, bias=False)
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

    result = apply_lora_to_model(model, [_int_lora("proj")], [1.0], CPU)
    assert result["affected_keys"] == ("proj.weight",)
    assert not torch.equal(model.proj.weight, original["proj.weight"])

    undo_lora_on_model(model, result, CPU)
    for key, tensor in model.state_dict().items():
        assert torch.equal(tensor, original[key]), f"{key} not restored exactly"


def test_two_loras_on_one_weight_accumulate_into_a_single_delta():
    model = _TinyNet()
    original = model.proj.weight.clone()

    first, second = _int_lora("proj"), _int_lora("proj")
    result = apply_lora_to_model(model, [first, second], [1.0, 1.0], CPU)
    stacked = model.proj.weight.clone()

    # One LoRA with the doubled multiplier is the same delta (both are 1s).
    single = _TinyNet()
    apply_lora_to_model(single, [_int_lora("proj")], [2.0], CPU)
    assert torch.equal(stacked, single.proj.weight)

    undo_lora_on_model(model, result, CPU)
    assert torch.equal(model.proj.weight, original)


def test_apply_reports_and_skips_unused_keys(caplog):
    model = _TinyNet()
    sd = _int_lora("proj")
    sd["nowhere.at.all.lora_A.weight"] = torch.ones(2, 4)
    sd["nowhere.at.all.lora_B.weight"] = torch.ones(6, 2)

    result = apply_lora_to_model(model, [sd], [1.0], CPU)
    assert result["affected_keys"] == ("proj.weight",)
    assert "unused keys" in caplog.text


def test_apply_with_no_loras_returns_an_empty_result():
    result = apply_lora_to_model(_TinyNet(), [], [], CPU)
    assert result["affected_keys"] == ()
    assert result["lora_sds"] == []
    # Undoing it is a no-op rather than an error.
    undo_lora_on_model(_TinyNet(), result, CPU)


# ----------------------------------------------------------- switch_loras()


@pytest.fixture
def switch_model(tmp_path, monkeypatch):
    """A stub adapter over a spy-instrumented apply/undo pair."""
    import thenoise.models.base as model_base

    events = []
    real_apply, real_undo = model_base.apply_lora_to_model, model_base.undo_lora_on_model

    def spy_apply(model, lora_sds, multipliers, device, **kwargs):
        events.append(("apply", len(lora_sds)))
        return real_apply(model, lora_sds, multipliers, device, **kwargs)

    def spy_undo(model, result, device):
        events.append(("undo", None))
        return real_undo(model, result, device)

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
            self.single_blocks[0].linear1 = nn.Linear(4, 6, bias=False)
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
