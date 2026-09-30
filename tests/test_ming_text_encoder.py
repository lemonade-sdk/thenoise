"""Ming-Image's conditioner: what it captures, where it positions, and how int8 lands.

Weight-free and GPU-free throughout. Tiny ``BailingMoeV2`` / connector configs stand in
for the 18 B thinker (``head_dim`` stays the released 128 because ``video_rope`` sizes
its mrope sections against the resulting 64-wide rotary band), a ten-row ``tokenizers``
build carries the checkpoint's REAL special-token ids, and both released text-encoder
exports are replayed from synthetic safetensors files written out of a tiny conditioner.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer, models, pre_tokenizers

from conftest import comfy_quant, int8_qt, write_safetensors
from thenoise.text_encoders.bailing_moe import (
    MROPE_SECTION,
    BailingMoeV2Config,
    ExpertBank,
    SparseMoeBlock,
    expert_dispatch,
    video_rope,
)
from thenoise.text_encoders.ming_image import (
    IMAGE_END_TOKEN,
    IMAGE_PATCH_TOKEN,
    IMAGE_TOKEN,
    MingConnectorConfig,
    MingImageConditioner,
    MingTokenizerError,
    QUERY_TOKENS,
    T2I_PROMPT_TEMPLATE,
    TEXT_ENCODER_DROP_KEYS,
    build_prompt,
    build_prompt_ids,
    check_query_block,
    encode_ming_prompt,
    load_ming_text_encoder,
    load_ming_tokenizer,
)

ROPE_THETA = 600000.0
ROTARY_DIM = 64  # head_dim 128 * partial_rotary_factor 0.5 == 2 * sum(MROPE_SECTION)
#: The interleaved height/width slots of the rotary band; the rest carry ``t``.
SPATIAL_SLOTS = sum(MROPE_SECTION[1:])

TINY_THINKER = BailingMoeV2Config(
    vocab_size=IMAGE_PATCH_TOKEN + 8,  # the query block is addressed by its real id
    hidden_size=32,
    intermediate_size=48,
    moe_intermediate_size=16,
    num_hidden_layers=4,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=128,
    rope_theta=ROPE_THETA,
    num_experts=8,
    num_experts_per_tok=2,
    n_group=4,
    topk_group=2,
    first_k_dense_replace=1,
)

TINY_CONNECTOR = MingConnectorConfig(
    hidden_size=24,
    intermediate_size=40,
    num_hidden_layers=1,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=128,
)

#: One dict describes the tiny conditioner to the loader AND to the direct builder.
TINY_CONDITIONER = dict(
    thinker=TINY_THINKER,
    connector=TINY_CONNECTOR,
    selected_layers=(2, 3, 4),  # ends at the tiny thinker's layer count
    num_queries=4,
    query_grid=(1, 4),
    cap_feat_dim=12,
    direct_dim=20,
)

NUM_QUERIES = TINY_CONDITIONER["num_queries"]
CAP_DIM = TINY_CONDITIONER["cap_feat_dim"]
DIRECT_DIM = TINY_CONDITIONER["direct_dim"]

#: Three prompt tokens, then ``<image> <imagePatch> </image>``: the shape the released
#: template really ends on (the query block is the last thing the prompt sends).
IDS = [1, 2, 3, IMAGE_TOKEN, IMAGE_PATCH_TOKEN, IMAGE_END_TOKEN]
QUERY_AT = IDS.index(IMAGE_PATCH_TOKEN)
#: What the thinker sees: the marker row is replaced by the query run.
SEQUENCE_LEN = len(IDS) - 1 + NUM_QUERIES


def tiny_conditioner(**overrides) -> MingImageConditioner:
    """A seeded, frozen conditioner at test scale (the same numbers every call)."""
    torch.manual_seed(0)
    return MingImageConditioner(**{**TINY_CONDITIONER, **overrides}).eval().requires_grad_(False)


# ------------------------------------------------------------- a stand-in tokenizer

#: The checkpoint's own ids for the markup the template spells out. ``tokenizers``
#: takes a sparse WordLevel vocab, so a ten-row table can carry the real ones.
SPECIAL_IDS = {
    "<role>": 157151,
    "</role>": 157152,
    "<|role_end|>": 156895,
    "<imagePatch>": IMAGE_PATCH_TOKEN,
    "<image>": IMAGE_TOKEN,
    "</image>": IMAGE_END_TOKEN,
}


def ming_tokenizer() -> Tokenizer:
    """A whitespace WordLevel tokenizer that knows the template's markup.

    Not the released tokenizer (a 12 MB BPE payload inside the TE file), but an
    honest stand-in: every markup string is a special token with its released id, so
    ``build_prompt_ids``/``check_query_block`` see exactly the marker geometry they
    see in production. Words the tiny vocab lacks become ``<unk>``, which is what the
    Chinese system turn does to a tokenizer that lacks it too.
    """
    vocab = {"<unk>": 0, "a": 1, "cat": 2, "flask": 3, **SPECIAL_IDS}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.add_special_tokens(list(SPECIAL_IDS))
    return tokenizer


# ------------------------------------------------------- synthetic checkpoint files

# The int8 export keeps these full precision (measured from the released header): an
# embedding gathers rows instead of running a GEMM, and the routers score in fp32.
_KEPT_FULL_PRECISION = ("embed_tokens.", "gate.proj.", "image_gate.proj.", "expert_bias")


def _as_int8(state_dict: dict) -> dict:
    """Re-export a conditioner state dict the way the int8-convrot file does.

    Per-row scales for every projection — and the same treatment for the 3-D expert
    banks, whose stored scale is ``[experts, out, 1]`` in the released file: one step
    per expert per output row.
    """
    out = {}
    for key, value in state_dict.items():
        quantizable = (
            key.endswith(".weight")
            and value.dim() >= 2
            and not any(part in key for part in _KEPT_FULL_PRECISION)
        )
        if not quantizable:
            out[key] = value.to(torch.bfloat16)
            continue
        rows = value.reshape(-1, value.shape[-1]).float()
        scale = (rows.abs().amax(dim=1, keepdim=True) / 127).clamp_min(1e-8)
        out[key] = (rows / scale).round().clamp(-127, 127).to(torch.int8).view_as(value)
        out[f"{key}_scale"] = scale.view(*value.shape[:-1], 1).to(torch.float32)
        out[key[: -len("weight")] + "comfy_quant"] = comfy_quant(convrot=False)
    return out


def _unbuilt_tensors() -> dict:
    """The prefixes the conditioner does not build, as the released file carries them."""
    junk = {
        "vision.patch_embed.weight": torch.zeros(4, 4, dtype=torch.bfloat16),
        "vision.blocks.0.attn.qkv.weight": torch.zeros(4, 4, dtype=torch.bfloat16),
        "linear_proj.0.weight": torch.zeros(4, 4, dtype=torch.bfloat16),
        "linear_proj.0.bias": torch.zeros(4, dtype=torch.bfloat16),
        "thinker.lm_head.weight": torch.zeros(8, 32, dtype=torch.bfloat16),
    }
    assert all(key.startswith(TEXT_ENCODER_DROP_KEYS) for key in junk), (
        "the fixture must exercise the prefixes the loader really drops"
    )
    return junk


# --------------------------------------------------------------- the conditioner


def test_tiny_conditioner_returns_both_conditioning_tensors():
    cap, direct = tiny_conditioner()(torch.tensor([IDS]))

    assert cap.shape == (1, NUM_QUERIES, CAP_DIM)
    # ``:q-1``: the prompt span, with the ``<image>`` that opens the block left out.
    assert direct.shape == (1, QUERY_AT - 1, DIRECT_DIM)
    assert torch.isfinite(cap).all() and torch.isfinite(direct).all()


def test_the_query_block_replaces_the_imagepatch_embedding():
    """Spliced OVER the marker, not appended after it — which is what makes the
    prompt span ``q-1`` and the rope counter spend one position on the whole block.
    """
    cond = tiny_conditioner()
    embeddings = cond.thinker.embed_tokens(torch.tensor([IDS]))
    spliced = cond.query_block(embeddings, QUERY_AT)

    assert spliced.shape == (1, SEQUENCE_LEN, embeddings.shape[-1])
    assert torch.equal(spliced[0, :QUERY_AT], embeddings[0, :QUERY_AT])
    assert torch.equal(
        spliced[0, QUERY_AT : QUERY_AT + NUM_QUERIES],
        cond.query_tokens.to(embeddings.dtype),
    )
    assert torch.equal(spliced[0, QUERY_AT + NUM_QUERIES :], embeddings[0, QUERY_AT + 1 :])


def test_capture_returns_the_layer_inputs_and_the_post_norm_state():
    """Index ``k`` below the layer count is the INPUT of layer ``k`` (index 0 being
    the embeddings); the last entry is the state AFTER the thinker's own norm, which
    is what the reference's ``hidden_states[-1]`` holds.
    """
    cond = tiny_conditioner()
    seen: dict[int, torch.Tensor] = {}
    for index, layer in enumerate(cond.thinker.layers):
        layer.register_forward_pre_hook(
            lambda _module, args, index=index: seen.__setitem__(index, args[0])
        )

    embeddings = cond.query_block(cond.thinker.embed_tokens(torch.tensor([IDS])), QUERY_AT)
    hidden, captured = cond.thinker(
        embeddings, blocks=[(QUERY_AT, 1, NUM_QUERIES)], capture_pre_layers=(0, 2)
    )

    assert cond.capture_pre_layers == (2, 3)  # ``selected_layers`` minus the last
    assert len(captured) == 3  # the two requested inputs + the post-norm state
    assert captured[0] is embeddings
    assert captured[1] is seen[2]
    assert captured[2] is hidden


@pytest.mark.parametrize("selected_layers", [(2, 3), (2, 3, 5), ()])
def test_selected_layers_must_end_at_the_thinker_layer_count(selected_layers):
    """``(2, 3)`` would ask for a state that is not the post-norm one and leave the
    directVLM concat short of the third tensor the released config selects.
    """
    with pytest.raises(ValueError, match="layer count"):
        tiny_conditioner(selected_layers=selected_layers)


def test_the_query_grid_has_to_hold_the_query_tokens():
    with pytest.raises(ValueError, match="do not fill"):
        tiny_conditioner(num_queries=NUM_QUERIES, query_grid=(1, 2))


@pytest.mark.parametrize(
    "ids",
    [
        [1, 2, 3, 4, 5],  # no marker at all
        [1, IMAGE_PATCH_TOKEN, 2, IMAGE_PATCH_TOKEN, 3],  # two blocks: which is it?
        IMAGE_PATCH_TOKEN,  # ... and a prompt that is nothing but the marker
    ],
)
def test_a_conditioner_without_exactly_one_marked_block_refuses(ids):
    """Each of these would otherwise draw a picture out of the markup alone."""
    with pytest.raises(ValueError, match="imagePatch"):
        tiny_conditioner()(torch.tensor([ids]))


def test_the_query_tokens_are_the_ones_marked_as_image_tokens():
    """``router_type: MultiRouter`` means the block's tokens answer to
    ``mlp.image_gate`` and the prompt's to ``mlp.gate``; a wrong mask routes the
    picture through the wrong experts and nothing complains.
    """
    cond = tiny_conditioner()
    masks: list[torch.Tensor] = []
    real_forward = SparseMoeBlock.forward

    def spy(self, x, image_mask):
        masks.append(image_mask)
        return real_forward(self, x, image_mask)

    SparseMoeBlock.forward = spy
    try:
        cond(torch.tensor([IDS]))
    finally:
        SparseMoeBlock.forward = real_forward

    assert masks, "the tiny thinker has no MoE layer to route through"
    expected = torch.zeros((1, SEQUENCE_LEN), dtype=torch.bool)
    expected[0, QUERY_AT : QUERY_AT + NUM_QUERIES] = True
    for mask in masks:
        assert mask.shape == (1, SEQUENCE_LEN)
        assert torch.equal(mask, expected)


def test_the_image_router_actually_decides_the_experts():
    torch.manual_seed(0)
    block = SparseMoeBlock(TINY_THINKER).eval().requires_grad_(False)
    x = torch.randn(1, 5, TINY_THINKER.hidden_size)
    all_image = torch.ones(1, 5, dtype=torch.bool)
    all_text = torch.zeros(1, 5, dtype=torch.bool)

    routed_image = block(x, all_image)
    routed_text = block(x, all_text)
    unmasked = block(x, None)

    assert not torch.allclose(routed_image, routed_text)
    assert torch.allclose(routed_text, unmasked), "a text-only mask must be a no-op"


# --------------------------------------------------------------------- video_rope


def _axis(cos: torch.Tensor, sin: torch.Tensor, slot: int) -> float:
    """The position one rotary slot rotates by: ``atan2`` inverts ``pos * inv_freq``.

    Only valid while the angle stays inside one turn, which is why every position in
    these tests is small.
    """
    inv_freq = 1.0 / (
        ROPE_THETA ** (torch.arange(0, ROTARY_DIM, 2, dtype=torch.float32) / ROTARY_DIM)
    )
    angle = torch.atan2(sin[..., slot], cos[..., slot]).remainder(2 * math.pi)
    return float(angle / inv_freq[slot])


@pytest.mark.parametrize(
    "seq_len,blocks,expected",
    [
        # No block: a plain 0..L-1 counter on all three axes.
        (4, (), [(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3)]),
        # One 1x4 block at index 2: ONE temporal/height position for all four tokens,
        # ``w`` centred on it, and the text after the block back down by n-1.
        (
            6,
            [(2, 1, 4)],
            [(0, 0, 0), (1, 1, 1), (2, 2, 1), (2, 2, 2), (2, 2, 3), (2, 2, 4)],
        ),
        # A 2x2 block: height steps per row, width per column, both centred on the
        # position the block occupies in the text.
        (
            6,
            [(1, 2, 2)],
            [(0, 0, 0), (1, 1, 1), (1, 1, 2), (1, 2, 1), (1, 2, 2), (2, 2, 2)],
        ),
    ],
)
def test_video_rope_positions(seq_len, blocks, expected):
    cos, sin = video_rope(seq_len, blocks, ROTARY_DIM, ROPE_THETA, torch.device("cpu"))

    for index, (t, h, w) in enumerate(expected):
        # The slow slots past the spatial band carry ``t``; inside it, even slots
        # carry ``h`` and odd ones ``w``.
        assert round(_axis(cos[index], sin[index], SPATIAL_SLOTS)) == t, index
        assert round(_axis(cos[index], sin[index], 0)) == h, index
        assert round(_axis(cos[index], sin[index], 1)) == w, index


def test_video_rope_duplicates_the_rotary_band():
    """``rotary_half=True``: the split-half helper rotates the first half of the head
    dim against the second, so every frequency has to appear twice.
    """
    cos, sin = video_rope(5, [(1, 1, 2)], ROTARY_DIM, ROPE_THETA, torch.device("cpu"))

    assert cos.shape == sin.shape == (5, 1, 1, ROTARY_DIM)
    half = ROTARY_DIM // 2
    assert torch.equal(cos[..., :half], cos[..., half:])
    assert torch.equal(sin[..., :half], sin[..., half:])


# ------------------------------------------------------------------ expert banks


def _reference_dispatch(x, topk_idx, topk_weight, gate_up, down):
    """The dense reading of the same maths: every token, every one of its experts."""
    out = torch.zeros_like(x)
    inter = down.in_features
    for token in range(x.shape[0]):
        for k in range(topk_idx.shape[1]):
            expert = int(topk_idx[token, k])
            gated = F.linear(x[token], gate_up.weight[expert])
            out[token] += (
                F.linear(F.silu(gated[:inter]) * gated[inter:], down.weight[expert])
                * topk_weight[token, k]
            )
    return out


def test_expert_dispatch_matches_a_dense_reference():
    """Sort-and-loop versus a plain per-token loop over the same weights: the scatter
    back has to make the sum over a token's experts exact.
    """
    torch.manual_seed(0)
    block = SparseMoeBlock(TINY_THINKER).eval().requires_grad_(False)
    x = torch.randn(6, TINY_THINKER.hidden_size)

    idx, weight = block.gate(x)
    gate_up, down = block.experts.gate_up_proj, block.experts.down_proj
    got = expert_dispatch(x, idx, weight, gate_up, down)
    want = _reference_dispatch(x, idx, weight, gate_up, down)

    assert torch.allclose(got, want, atol=1e-5)
    assert got.shape == x.shape


def _int8_bank(bank: ExpertBank):
    """Quantize a dense bank per expert per output row, as the file stores it."""
    dense = bank.weight.detach()
    rows = dense.reshape(-1, bank.in_features).float()
    scale = (rows.abs().amax(dim=1, keepdim=True) / 127).clamp_min(1e-9)
    qweight = (rows / scale).round().clamp(-127, 127).to(torch.int8).view_as(dense)
    return int8_qt(
        qweight, scale.view(bank.num_experts, bank.out_features, 1), convrot=False
    )


def test_an_int8_bank_stays_int8():
    """The point of the ComfyUI approach: the stored form is int8, so a bank costs one
    byte per weight instead of two. ``load_quantized`` back to bf16 would double it and
    buy nothing.
    """
    torch.manual_seed(0)
    bank = ExpertBank(4, 32, 64).eval().requires_grad_(False)
    elements = bank.weight.numel()

    bank.load_quantized(_int8_bank(bank))

    assert not dict(bank.named_parameters()), "a quantized bank must not keep a Parameter"
    assert bank.weight.storage_dtype == torch.int8
    assert bank.weight.nbytes == elements, "the bank came back heavier than one byte/weight"
    assert bank.expert_weight(0).shape == (32, 64)


def test_the_expert_view_is_a_view_and_only_loses_one_step():
    """``expert_weight`` must never materialise the bank: it shares its storage, so a
    per-expert GEMM costs nothing extra however big the bank is.
    """
    torch.manual_seed(0)
    bank = ExpertBank(4, 32, 64).eval().requires_grad_(False)
    dense = bank.weight.detach().clone()
    bank.load_quantized(_int8_bank(bank))

    stored, scale = bank.weight._qdata, bank.weight.params.scale
    for expert in range(bank.num_experts):
        view = bank.expert_weight(expert)
        assert view._qdata.data_ptr() == stored.data_ptr() + expert * view._qdata.numel()
        step = float(scale[expert].max())
        assert (view.dequantize().float() - dense[expert].float()).abs().max() <= step


def test_int8_expert_dispatch_stays_within_the_quantization_step():
    torch.manual_seed(0)
    block = SparseMoeBlock(TINY_THINKER).eval().requires_grad_(False).to(torch.bfloat16)
    x = torch.randn(6, TINY_THINKER.hidden_size, dtype=torch.bfloat16)
    gate_up, down = block.experts.gate_up_proj, block.experts.down_proj
    idx, weight = block.gate(x)
    idx, weight = idx.to(torch.long), weight.to(torch.bfloat16)

    dense = expert_dispatch(x, idx, weight, gate_up, down)
    gate_up.load_quantized(_int8_bank(gate_up))
    down.load_quantized(_int8_bank(down))
    quantized = expert_dispatch(x, idx, weight, gate_up, down)

    # A quantized GEMM's error is the weight step times the summed input magnitude,
    # plus the bf16 the layout emits in.
    step = float(gate_up.weight.params.scale.max()) * math.sqrt(TINY_THINKER.hidden_size)
    error = float((quantized.float() - dense.float()).abs().max())
    assert error <= 4 * step + float(dense.abs().max()) * 2**-7


# --------------------------------------------------------- both checkpoint files


def test_bf16_file_loads_strictly_and_drops_what_the_tree_does_not_build(tmp_path):
    reference = tiny_conditioner().to(torch.bfloat16)
    tensors = {k: v.to(torch.bfloat16) for k, v in reference.state_dict().items()}
    tensors.update(_unbuilt_tensors())
    tensors["tokenizer_json"] = torch.tensor(list(b'{"model": {}}'), dtype=torch.uint8)
    path = write_safetensors(tmp_path / "ming_te_bf16.safetensors", tensors)

    model, _ = load_ming_text_encoder(
        path,
        device="cpu",
        dtype=torch.bfloat16,
        config=TINY_CONDITIONER,
        with_tokenizer=False,
    )

    loaded = model.state_dict()
    assert set(loaded) == set(reference.state_dict())
    for key, value in reference.state_dict().items():
        assert torch.equal(loaded[key].to(value.dtype), value), key


def test_a_genuinely_missing_weight_still_raises(tmp_path):
    """Dropping the image tower must not turn the load into a tolerate-everything one."""
    tensors = {k: v.to(torch.bfloat16) for k, v in tiny_conditioner().state_dict().items()}
    tensors.update(_unbuilt_tensors())
    del tensors["proj_out.weight"]
    path = write_safetensors(tmp_path / "ming_te_partial.safetensors", tensors)

    with pytest.raises(Exception, match="proj_out"):
        load_ming_text_encoder(
            path,
            device="cpu",
            dtype=torch.bfloat16,
            config=TINY_CONDITIONER,
            with_tokenizer=False,
        )


def test_int8_file_loads_low_bit_and_draws_the_same_picture(tmp_path):
    reference = tiny_conditioner().to(torch.bfloat16)
    tensors = _as_int8(reference.state_dict())
    banks = {
        key: value
        for key, value in tensors.items()
        if key.endswith("experts.gate_up_proj.weight")
    }
    assert banks and all(value.dtype == torch.int8 for value in banks.values())
    path = write_safetensors(tmp_path / "ming_te_int8.safetensors", tensors)

    model, _ = load_ming_text_encoder(
        path,
        device="cpu",
        dtype=torch.bfloat16,
        config=TINY_CONDITIONER,
        with_tokenizer=False,
    )
    # The loader leaves grad mode / weight state to ``MemoryManager.register``; a test
    # that forwards without an adapter has to freeze what it loaded.
    model.eval().requires_grad_(False)

    loaded_banks = [m for m in model.modules() if isinstance(m, ExpertBank)]
    assert loaded_banks, "the tiny thinker has no expert banks"
    for bank in loaded_banks:
        assert bank.weight.storage_dtype == torch.int8, "the bank was dequantized at load"
        assert bank.weight.nbytes == bank.weight.numel()

    ids = torch.tensor([IDS])
    cap_ref, direct_ref = reference(ids)
    cap, direct = model(ids)

    for got, want, name in ((cap, cap_ref, "cap_feats"), (direct, direct_ref, "direct")):
        assert got.shape == want.shape
        assert torch.isfinite(got).all()
        # Several quantized GEMMs deep in a 4-layer stack: a few steps, not a drift.
        step = float(want.abs().max()) * 2**-5
        assert (got.float() - want.float()).abs().max() <= step, name


# ------------------------------------------------------------------ prompt side


def test_the_template_carries_the_prompt_and_ends_on_the_query_block():
    text = build_prompt("a cat")

    assert "a cat" in text
    assert text.endswith("<image><imagePatch></image>")
    assert text.count("<imagePatch>") == 1
    assert "{" not in T2I_PROMPT_TEMPLATE.replace("{prompt}", "")


@pytest.mark.parametrize("prompt", ["a cat", "", "a {} flask, <|role_end|> / a <image>"])
def test_the_prompt_is_only_substituted_never_scanned(prompt):
    """Only the template is ``format``-scanned: braces and non-block markup in the
    caption cannot break it or shift the query block.
    """
    ids = build_prompt_ids(ming_tokenizer(), prompt)

    assert ids[-3:] == [IMAGE_TOKEN, IMAGE_PATCH_TOKEN, IMAGE_END_TOKEN]
    assert check_query_block(ids) == len(ids) - 2


def test_a_caption_that_carries_the_literal_marker_is_refused():
    """Special tokens match anywhere, so a typed ``<imagePatch>`` really is a second
    block on the wire. Better a hard error than a picture of the wrong thing.
    """
    ids = build_prompt_ids(ming_tokenizer(), "a <imagePatch> flask")

    assert ids.count(IMAGE_PATCH_TOKEN) == 2
    with pytest.raises(ValueError, match="exactly one"):
        check_query_block(ids)


def test_check_query_block_accepts_the_released_marker_and_rejects_the_rest():
    good = [1, 2, IMAGE_TOKEN, IMAGE_PATCH_TOKEN, IMAGE_END_TOKEN, 3]
    assert check_query_block(good) == 3

    for broken in (
        [1, 2, 3],  # no marker
        good + [IMAGE_PATCH_TOKEN, 4],  # two markers
        [1, IMAGE_PATCH_TOKEN, IMAGE_END_TOKEN],  # no <image> in front
        [1, IMAGE_TOKEN, IMAGE_PATCH_TOKEN],  # no </image> behind
        [IMAGE_PATCH_TOKEN, IMAGE_END_TOKEN],  # marker at index 0: no prompt at all
    ):
        with pytest.raises(ValueError, match="imagePatch"):
            check_query_block(broken)


def test_the_tokenizer_is_read_out_of_the_checkpoint(tmp_path):
    """``tokenizer_json`` is a U8 payload in the TE file, not a module weight."""
    payload = torch.tensor(list(ming_tokenizer().to_str().encode("utf-8")), dtype=torch.uint8)
    path = write_safetensors(tmp_path / "te_with_tokenizer.safetensors", {"tokenizer_json": payload})

    tokenizer = load_ming_tokenizer(path)

    assert tokenizer.encode("<imagePatch>", add_special_tokens=False).ids == [
        IMAGE_PATCH_TOKEN
    ]


def test_the_tokenizer_falls_back_to_a_directory_and_else_raises(tmp_path):
    stripped = write_safetensors(
        tmp_path / "te_no_tokenizer.safetensors",
        {"thinker.norm.weight": torch.ones(4, dtype=torch.bfloat16)},
    )
    (tmp_path / "tokenizer_dir").mkdir()

    with pytest.raises(MingTokenizerError, match="tokenizer_json"):
        load_ming_tokenizer(stripped)

    # A re-export that moved the payload out still works when told where it went.
    with open(tmp_path / "tokenizer_dir" / "tokenizer.json", "w", encoding="utf-8") as handle:
        handle.write(ming_tokenizer().to_str())
    assert load_ming_tokenizer(stripped, tokenizer_dir=str(tmp_path / "tokenizer_dir"))


def test_encode_ming_prompt_threads_the_tokenizer_and_the_dtype():
    cond = tiny_conditioner().to(torch.bfloat16)
    cond.query_tokens.data.fill_(0.5)  # any distinctive value; the shape is the point

    cap, direct = encode_ming_prompt(cond, ming_tokenizer(), "a flask", dtype=torch.bfloat16)

    assert cap.dtype == torch.bfloat16 and direct.dtype == torch.bfloat16
    assert cap.shape == (1, NUM_QUERIES, CAP_DIM)
    # The prompt really reached the thinker: ``<imagePatch>`` at id 157157 is 4 tokens
    # from the end of the template, and the span in front of it is the whole prompt.
    assert direct.shape[1] == check_query_block(build_prompt_ids(ming_tokenizer(), "a flask")) - 1
