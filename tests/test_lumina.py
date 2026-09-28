"""The shared Lumina/S3-DiT core, and the two padding conventions it parameterises.

Weight-free and GPU-free: a tiny transformer (2 heads, head_dim 128 = the 32+48+48
axis split the real models use) is enough to pin down the things a wrong build would
only reveal as a bad picture — where the alignment pads go, who is allowed to attend
to them, where the image's temporal position starts, and where a second conditioning
tensor lands. Plus the checkpoint-name maps both loaders share.
"""
from __future__ import annotations

import pytest
import torch

from conftest import write_safetensors

# dim 256 / 2 heads -> head_dim 128, which is what the (32, 48, 48) axis split needs.
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


@pytest.fixture(autouse=True)
def _blocks_run_eagerly(monkeypatch):
    """Call the transformer blocks un-compiled.

    ``conftest`` disables dynamo for the whole suite (the tests check model math,
    not Inductor), and a ``fullgraph=True`` block *raises* rather than falling back
    to eager when dynamo is off. Unwrap to the pre-compile function so these tests
    can actually run a forward.
    """
    from thenoise.dit.lumina.models import LuminaTransformerBlock

    monkeypatch.setattr(
        LuminaTransformerBlock,
        "forward",
        LuminaTransformerBlock.forward._torchdynamo_orig_callable,
        raising=False,
    )


@pytest.fixture(params=["learned", "zero_masked"])
def core(request):
    """One tiny Lumina transformer per padding convention."""
    from thenoise.dit.lumina.models import LuminaTransformer2DModel

    torch.manual_seed(0)
    # float64 end to end: the asserts below are about placement, and bf16 rounding
    # would only give them a worse signal.
    model = LuminaTransformer2DModel(pad_mode=request.param, **TINY_CONFIG)
    return model.double().eval().requires_grad_(False)


def _inputs(cap_len=37, height=8, width=8, extra_len=0, dtype=torch.float64):
    latent = torch.randn(16, 1, height, width, dtype=dtype)
    cap = torch.randn(cap_len, TINY_CONFIG["cap_feat_dim"], dtype=dtype)
    extra = (
        torch.randn(extra_len, TINY_CONFIG["dim"], dtype=dtype) if extra_len else None
    )
    return latent, cap, extra


# ------------------------------------------------------------- pad-token existence


def test_learned_mode_owns_pad_tokens_and_zero_masked_does_not():
    """The pad tokens are the ONE parameter that tells the two family members apart.

    Z-Image ships them and Ming-Image does not, so a ``zero_masked`` model that
    declared them could never load its own checkpoint (the loader is strict), and a
    ``learned`` model without them would pad with garbage it then attends to.
    """
    from thenoise.dit.lumina.models import LuminaTransformer2DModel

    learned = LuminaTransformer2DModel(pad_mode="learned", **TINY_CONFIG)
    assert isinstance(learned.x_pad_token, torch.nn.Parameter)
    assert isinstance(learned.cap_pad_token, torch.nn.Parameter)

    masked = LuminaTransformer2DModel(pad_mode="zero_masked", **TINY_CONFIG)
    assert masked.x_pad_token is None and masked.cap_pad_token is None
    assert not {"x_pad_token", "cap_pad_token"} & set(masked.state_dict())


def test_zimage_subclass_is_the_learned_mode():
    from thenoise.dit.lumina.models import LuminaTransformer2DModel
    from thenoise.dit.zimage.models import ZImageTransformer2DModel

    zimage = ZImageTransformer2DModel(**TINY_CONFIG)
    assert zimage.pad_mode == "learned"
    assert isinstance(zimage, LuminaTransformer2DModel)
    # An explicit pad_mode still wins, so the subclass is a default and not a cage.
    assert ZImageTransformer2DModel(pad_mode="zero_masked", **TINY_CONFIG).pad_mode == "zero_masked"


def test_unknown_pad_mode_is_an_error():
    from thenoise.dit.lumina.models import LuminaTransformer2DModel

    with pytest.raises(ValueError, match="unknown pad_mode"):
        LuminaTransformer2DModel(pad_mode="mask_the_moon", **TINY_CONFIG)


# ---------------------------------------------------------------------- stream math


def test_padded_lengths_round_up_both_streams(core):
    latent, cap, extra = _inputs(cap_len=37)
    x_stream, cap_stream = core.patchify_and_embed([latent], [cap], 2, 1, None)

    assert x_stream.valid == [16]          # 8x8 latent, 2x2 patch -> 16 tokens
    assert x_stream.padded == [32]
    assert cap_stream.valid == [37]
    assert cap_stream.padded == [64]       # 37 -> next multiple of 32


def test_extra_conditioning_extends_the_caption_block(core):
    """``cap_extra`` is caption-block tail: it counts in the length AND the positions.

    The image's ``t`` axis starts after the whole padded caption, so growing the
    caption by any amount must move the image positions with it.
    """
    latent, cap, extra = _inputs(cap_len=40, extra_len=24)
    x_stream, cap_stream = core.patchify_and_embed([latent], [cap], 2, 1, [extra])

    assert cap_stream.valid == [64]        # 40 prompt + 24 extra
    assert cap_stream.padded == [64]
    # Caption positions are 1..len, and the image starts one past the PADDED length.
    assert cap_stream.pos_ids[0][0].tolist() == [1, 0, 0]
    assert cap_stream.pos_ids[0][-1].tolist() == [64, 0, 0]
    assert x_stream.pos_ids[0][0].tolist() == [65, 0, 0]


def test_extra_conditioning_moves_the_image_positions(core):
    """The rule the whole caption-block layout exists to enforce."""
    latent, cap, _ = _inputs(cap_len=30)
    x_without, cap_without = core.patchify_and_embed([latent], [cap], 2, 1, None)

    extra = torch.zeros(10, TINY_CONFIG["dim"], dtype=torch.float64)
    x_with, cap_with = core.patchify_and_embed([latent], [cap], 2, 1, [extra])

    # 30 valid padded to 32, but 30+10 valid padded to 64. The image's own token
    # count is unchanged; its START moves with the caption's padded length, which
    # is all the RoPE geometry can key on.
    assert cap_without.padded == [32] and cap_with.padded == [64]
    assert x_without.padded == x_with.padded == [32]
    assert x_without.pos_ids[0][0, 0] == 33
    assert x_with.pos_ids[0][0, 0] == 65


def test_extra_length_mismatch_is_an_error(core):
    latent, cap, _ = _inputs()
    with pytest.raises(ValueError, match="cap_extra"):
        core.patchify_and_embed([latent], [cap], 2, 1, [torch.zeros(3, 256, dtype=torch.float64), torch.zeros(3, 256, dtype=torch.float64)])


def test_position_ids_cover_the_padded_length_and_pads_sit_at_the_origin(core):
    latent, cap, _ = _inputs(cap_len=37)
    x_stream, cap_stream = core.patchify_and_embed([latent], [cap], 2, 1, None)

    for stream in (x_stream, cap_stream):
        assert len(stream.pos_ids[0]) == stream.padded[0]
    # The image is 16 tokens padded to 32: the last 16 positions are the (0,0,0) pad.
    assert (x_stream.pos_ids[0][16:] == 0).all()
    assert (cap_stream.pos_ids[0][37:] == 0).all()


# ------------------------------------------------------------------------- forward


@pytest.mark.parametrize("extra_len", [0, 24])
def test_forward_shapes_and_finiteness(core, extra_len):
    latent, cap, extra = _inputs(cap_len=37, height=8, width=8, extra_len=extra_len)
    extras = [extra] if extra is not None else None
    t = torch.tensor([0.42], dtype=torch.float64)

    core.prepare_rope([latent], [cap], extras)
    out = core([latent], t, [cap], extras)[0]
    assert out.shape == (16, 1, 8, 8)
    assert torch.isfinite(out).all()


def test_forward_matches_preparing_rope_per_branch(core):
    """The ``rope_key`` split behaves like a fresh preparation of the same caption."""
    latent, cap, _ = _inputs(cap_len=30)
    t = torch.tensor([0.75], dtype=torch.float64)

    core.prepare_rope([latent], [cap], key="")
    direct = core([latent], t, [cap])[0]

    core.prepare_rope([latent], [cap], key="_neg", clear=False)
    keyed = core([latent], t, [cap], rope_key="_neg")[0]
    assert torch.allclose(direct, keyed)


def test_pad_slots_hold_the_pad_token_in_learned_mode():
    """A learned pad slot is the model's own embedding, not a zero."""
    from thenoise.dit.lumina.models import LuminaTransformer2DModel
    from thenoise.utils.sequence import pad_to_length

    core = LuminaTransformer2DModel(pad_mode="learned", **TINY_CONFIG)
    feats = [torch.ones(3, 4, dtype=torch.float64)]
    pad_token = torch.full((1, 4), 7.0, dtype=torch.float64)
    padded = pad_to_length(feats, [5], pad_token=pad_token)
    assert torch.equal(padded[0][:3], feats[0])
    assert torch.equal(padded[0][3:], pad_token.expand(2, 4))


def test_cap_extra_reaches_the_network_and_changes_the_output():
    """``cap_extra=None`` and an all-zero extra are NOT the same forward.

    Zero-padding the caption tail still lengthens the stream and shifts the image
    positions, so if a forward ever ignored ``cap_extra`` this would fail silently
    in the direction that matters (the conditioning being dropped).
    """
    from thenoise.dit.lumina.models import LuminaTransformer2DModel

    torch.manual_seed(0)
    model = LuminaTransformer2DModel(pad_mode="zero_masked", **TINY_CONFIG).double().eval()
    latent, cap, _ = _inputs(cap_len=40)
    t = torch.tensor([0.5], dtype=torch.float64)

    model.prepare_rope([latent], [cap])
    base = model([latent], t, [cap])[0]

    zero_extra = torch.zeros(16, TINY_CONFIG["dim"], dtype=torch.float64)
    model.prepare_rope([latent], [cap], [zero_extra])
    with_extra = model([latent], t, [cap], [zero_extra])[0]
    assert not torch.allclose(base, with_extra)

    real_extra = torch.randn(16, TINY_CONFIG["dim"], dtype=torch.float64)
    assert not torch.allclose(with_extra, model([latent], t, [cap], [real_extra])[0])


def test_batch_of_two_different_caption_lengths(core):
    """Unequal captions pad to the batch max; every item still returns its own grid."""
    lat_a, cap_a, extra_a = _inputs(cap_len=31, height=8, width=8)
    lat_b, cap_b, _ = _inputs(cap_len=17, height=4, width=4)
    t = torch.full((2,), 0.3, dtype=torch.float64)

    core.prepare_rope([lat_a, lat_b], [cap_a, cap_b], None)
    out = core([lat_a, lat_b], t, [cap_a, cap_b], None)
    assert out[0].shape == (16, 1, 8, 8) and out[1].shape == (16, 1, 4, 4)
    assert all(torch.isfinite(o).all() for o in out)


# ----------------------------------------------------------------- the mask itself


def test_zero_masked_punches_holes_and_learned_does_not():
    """The whole point of ``pad_mode``, written out as the masks attention sees.

    Unified stream: image 4 valid padded to 8, caption 3 valid padded to 8.
    ``zero_masked`` must attend to 7 of the 16 slots and drop BOTH pad runs;
    ``learned`` attends to all 16 (its pads are real tokens).
    """
    from thenoise.dit.lumina.models import LuminaTransformer2DModel

    segments = [[(4, 8), (3, 8)]]
    masked = LuminaTransformer2DModel(pad_mode="zero_masked", **TINY_CONFIG)
    learned = LuminaTransformer2DModel(pad_mode="learned", **TINY_CONFIG)

    mask = masked._attention_mask(segments, torch.device("cpu"))
    assert mask.shape == (1, 16)
    assert mask[0].tolist() == [True] * 4 + [False] * 4 + [True] * 3 + [False] * 5

    # No pads at all -> the mask-free fast path, for both modes.
    none = [[(8, 8)]]
    assert masked._attention_mask(none, torch.device("cpu")) is None
    assert learned._attention_mask(none, torch.device("cpu")) is None

    # Learned: the pads are attended, so a single stream is all-True and vanishes.
    assert learned._attention_mask(segments, torch.device("cpu")) is None
    # ...but the batch padding past the item's own length still is masked.
    ragged = [[(4, 8)], [(2, 4)]]
    assert learned._attention_mask(ragged, torch.device("cpu"))[0].tolist() == [True] * 8
    assert masked._attention_mask(ragged, torch.device("cpu"))[1].tolist() == (
        [True, True] + [False] * 6
    )


def test_single_stream_masks_only_its_own_pads(core):
    """A refiner stream is one segment: valid, then the alignment pad run."""
    from thenoise.dit.lumina.models import LuminaTransformer2DModel

    model = LuminaTransformer2DModel(pad_mode="zero_masked", **TINY_CONFIG)
    mask = model._attention_mask([[(5, 8)]], torch.device("cpu"))
    assert mask[0].tolist() == [True] * 5 + [False] * 3


# --------------------------------------------------------------- checkpoint naming


LEGACY_SD = {
    "all_x_embedder.2-1.weight": torch.arange(64, dtype=torch.float32).reshape(8, 8),
    "all_final_layer.2-1.linear.weight": torch.ones(8, 8),
    "layers.0.attention.to_q.weight": torch.full((4, 8), 1.0),
    "layers.0.attention.to_k.weight": torch.full((4, 8), 2.0),
    "layers.0.attention.to_v.weight": torch.full((4, 8), 3.0),
    "layers.0.attention.to_out.0.weight": torch.ones(8, 8),
    "layers.0.attention.norm_q.weight": torch.ones(4),
    "layers.0.attention.norm_k.weight": torch.ones(4),
    "layers.0.attention.qkv.weight": torch.ones(8, 8),  # already-fused module untouched
}


def test_lumina_key_map_renames_the_legacy_layout():
    from thenoise.dit.lumina.keys import lumina_key_map

    assert lumina_key_map("all_x_embedder.2-1.weight") == "x_embedder.weight"
    assert lumina_key_map("all_final_layer.2-1.linear.bias") == "final_layer.linear.bias"
    assert (
        lumina_key_map("layers.0.attention.to_out.0.weight")
        == "layers.0.attention.out.weight"
    )
    assert (
        lumina_key_map("layers.0.attention.norm_q.weight")
        == "layers.0.attention.qk_norm.query_norm.weight"
    )
    assert (
        lumina_key_map("layers.0.attention.norm_k.weight")
        == "layers.0.attention.qk_norm.key_norm.weight"
    )
    # The int8 export's spelling works too, and fused keys pass through untouched.
    assert (
        lumina_key_map("layers.3.attention.q_norm.weight")
        == "layers.3.attention.qk_norm.query_norm.weight"
    )
    assert lumina_key_map("x_embedder.weight") == "x_embedder.weight"


def test_fuse_qkv_stacks_in_qkv_row_order():
    from thenoise.dit.lumina.keys import fuse_qkv

    fused = fuse_qkv({k: v.clone() for k, v in LEGACY_SD.items()})
    qkv = fused["layers.0.attention.qkv.weight"]
    assert qkv.shape == (12, 8)
    assert qkv[:4].eq(1.0).all() and qkv[4:8].eq(2.0).all() and qkv[8:].eq(3.0).all()
    for part in ("to_q", "to_k", "to_v"):
        assert f"layers.0.attention.{part}.weight" not in fused


def test_fuse_qkv_is_a_noop_for_an_already_fused_checkpoint():
    from thenoise.dit.lumina.keys import fuse_qkv

    sd = {"layers.0.attention.qkv.weight": torch.ones(24, 8)}
    assert fuse_qkv(sd) is sd


def test_lumina_state_map_loads_a_legacy_checkpoint(tmp_path):
    """A legacy-named file lands on the fused module tree, strictly.

    The state dict is built FROM the model so the shapes line up; what is under
    test is the naming, not the numbers.
    """
    from accelerate import init_empty_weights

    from thenoise.dit.lumina.keys import lumina_state_map
    from thenoise.dit.lumina.models import LuminaTransformer2DModel
    from thenoise.utils.loader import load_dit

    torch.manual_seed(0)
    ref = LuminaTransformer2DModel(pad_mode="zero_masked", **TINY_CONFIG)
    sd = {k: v.detach().clone() for k, v in ref.state_dict().items()}

    legacy = {}
    for key, value in sd.items():
        if key.endswith("attention.qkv.weight"):
            head = key[: -len("qkv.weight")]
            rows = value.shape[0] // 3
            for i, part in enumerate(("to_q", "to_k", "to_v")):
                legacy[f"{head}{part}.weight"] = value[i * rows : (i + 1) * rows]
        else:
            legacy[key] = value
    legacy = {
        k.replace("x_embedder.", "all_x_embedder.2-1.")
        .replace("final_layer.", "all_final_layer.2-1.")
        .replace("attention.out.", "attention.to_out.0.")
        .replace("qk_norm.query_norm.", "norm_q.")
        .replace("qk_norm.key_norm.", "norm_k."): v
        for k, v in legacy.items()
    }
    path = write_safetensors(tmp_path / "legacy.safetensors", legacy)

    with init_empty_weights():
        model = LuminaTransformer2DModel(pad_mode="zero_masked", **TINY_CONFIG)
    load_dit(model, path, device="cpu", dtype=torch.float32, state_map=lumina_state_map)

    for key, value in ref.state_dict().items():
        assert torch.equal(model.state_dict()[key], value), key


# ----------------------------------------------------------------- Z-Image wiring


def test_zimage_still_pads_with_its_learned_tokens_and_attends():
    """End-to-end Z-Image behaviour through the shared core (the regression guard).

    A caption that needs padding must produce (a) pad slots holding the learned
    ``cap_pad_token`` and (b) an attention mask that keeps them, both of which are
    Z-Image's documented behaviour and both of which the refactor could silently
    have changed.
    """
    from thenoise.dit.zimage.models import ZImageTransformer2DModel
    from thenoise.utils.sequence import pad_to_length

    torch.manual_seed(0)
    model = ZImageTransformer2DModel(**TINY_CONFIG).eval()
    latent, cap, _ = _inputs(cap_len=33, height=4, width=4, dtype=torch.float64)
    model.double()

    x_stream, cap_stream = model.patchify_and_embed([latent], [cap], 2, 1, None)
    assert cap_stream.padded == [64]

    embedded = model.cap_embedder(cap)
    padded = pad_to_length([embedded], cap_stream.padded, pad_token=model.cap_pad_token)
    assert torch.equal(padded[0][33:], model.cap_pad_token.to(torch.float64).expand(31, 256))
    # Real tokens untouched by the pad fill.
    assert torch.equal(padded[0][:33], embedded)

    mask = model._attention_mask([cap_stream.segments], torch.device("cpu"))
    assert mask is None  # batch of one, pads attended => no mask needed at all
