"""The H3-Image DiT: its packed layout, its head bank, and the geometry read off a file.

Run on a reduced-width instance built here, because the whole contract is structure —
which rows carry which timestep, which head answers which interval, which shape comes
back out.
"""
from __future__ import annotations

import json
from itertools import groupby

import pytest
import torch
from safetensors.torch import save_file

from conftest import write_safetensors
from thenoise.dit.h3_image import sampling
from thenoise.dit.h3_image.encoder import (
    _truncate_qwen3_vl,
    layer_count,
    prompt_with_pictures,
    vision_token_tags,
)
from thenoise.dit.h3_image.layout import Layout, frame_grid, pad_to_patch
from thenoise.dit.h3_image.models import (
    COND_TIMESTEP,
    H3ImageDiT,
    H3ImageParams,
    detect_params,
    load_h3_image_dit,
)

TINY_SIGMAS = (1.0, 0.5, 0.0)


def _tiny_params(**over) -> H3ImageParams:
    base = dict(
        hidden_size=48,
        num_layers=1,
        token_refiner_layers=1,
        num_heads=2,
        head_dim=16,
        ffn_hidden_size=32,
        latents_dim=4,
        text_dim=16,
        rope_inv_freq=2,
        adaln_grid=5,
        adaln_dim=3,
        pdd_steps=2,
        pdd_sigmas=TINY_SIGMAS,
    )
    return H3ImageParams(**{**base, **over})


def _tiny_dit(params: H3ImageParams | None = None) -> H3ImageDiT:
    """A built instance with its two buffers filled in.

    ``adaln_t_table`` and ``rope.inv_freq`` are ``torch.empty`` until a checkpoint
    loads over them, so an unfilled instance computes with whatever the allocator
    left behind — usually NaN, which would fail every test for the wrong reason.
    """
    torch.manual_seed(0)
    dit = H3ImageDiT(params or _tiny_params()).eval().requires_grad_(False)
    dit.adaln_t_table.normal_()
    dit.rope.inv_freq.copy_(torch.tensor([1.0, 0.25]))
    return dit


# ------------------------------------------------------------------ geometry


def test_patchify_round_trips_through_the_head():
    dit = _tiny_dit()
    latent = torch.randn(1, 4, 6, 8)
    rows = dit.patchify(latent)
    assert rows.shape == (3 * 4, 4 * 4)  # (h/2 * w/2) rows of C*2*2
    assert torch.allclose(dit.unpatchify(rows, 6, 8), latent)


def test_patch_rows_are_area_normalised():
    """A tall and a wide lattice of one area span the same coordinate stretch."""
    tall, _ = frame_grid(2, 8)
    wide, _ = frame_grid(8, 2)
    assert torch.allclose(tall[:, 0], wide[:, 1])
    assert torch.allclose(tall[:, 1], wide[:, 0])


def test_pad_to_patch_repeats_the_edge_and_reports_the_original():
    padded, (h, w) = pad_to_patch(torch.ones(1, 2, 3, 4), (1, 2, 2))
    assert (h, w) == (3, 4)
    assert padded.shape == (1, 2, 4, 4)
    assert torch.equal(padded[..., 2, :], padded[..., 3, :])


# --------------------------------------------------------------------- layout


def test_layout_segments_cover_the_sequence_in_order():
    layout = Layout(text_len=4, latent_h=4, latent_w=6, refs=[(2, 4), (4, 4)])
    kinds = [kind for _, _, kind in layout.segments]
    assert kinds == ["text", "ref", "ref", "target"]

    assert layout.segments[0][0] == 0
    for (_, end, _), (start, _, _) in zip(layout.segments, layout.segments[1:]):
        assert end == start
    assert layout.segments[-1][1] == layout.seq_len
    assert layout.target_span == layout.segments[-1][:2]

    # (2,4) is one row of patches x two; (4,4) is 2x2; the target (4,6) is 2x3.
    assert [end - start for start, end, _ in layout.segments] == [4, 2, 4, 6]
    assert layout.codec_rows == 12


def test_layout_puts_references_one_time_unit_apart():
    """The reference rows sit on the same grid as the target but a unit earlier.

    That offset, not a different projection, is what tells the DiT which rows are the
    picture to keep and which is the picture to make — the timestep tags the modality
    tag alone cannot, since both are visual.
    """
    layout = Layout(text_len=3, latent_h=2, latent_w=2, refs=[(2, 2), (2, 2)])
    (text_span, ref_a, ref_b, target) = [
        layout.position_ids[start:stop] for start, stop, _ in layout.segments
    ]
    assert text_span[:, 0].tolist() == [0.0, 1.0, 2.0]
    assert torch.equal(text_span[:, 1:], torch.zeros(3, 2))
    assert ref_a[:, 0].unique().tolist() == [3.0]
    assert ref_b[:, 0].unique().tolist() == [4.0]
    assert target[:, 0].unique().tolist() == [5.0]


def test_layout_without_references_is_text_then_target():
    layout = Layout(text_len=2, latent_h=4, latent_w=4)
    assert [kind for _, _, kind in layout.segments] == ["text", "target"]
    assert layout.refs == ()


def test_modulation_rows_follow_the_timestep_and_the_tags():
    layout = Layout(text_len=4, latent_h=2, latent_w=2)
    rows = layout.modulation_rows({"text": 1, "ref": 1, "target": 1})
    assert rows == [(0, 4, 4), (4, 5, 3)]  # t-row 1: text tag 1 -> 4, visual tag 0 -> 3

    tags = torch.tensor([1, 0, 0, 1])
    split = layout.modulation_rows({"text": 0, "ref": 0, "target": 0}, tags=tags)
    assert split == [(0, 1, 1), (1, 3, 0), (3, 4, 1), (4, 5, 0)]


def test_a_reference_is_modulated_at_its_own_timestep():
    """A reference's pinned timestep is a different row than the target's.

    The run's distinct timesteps are sorted, so at any sigma above 0.001 the target is
    row 0 and the reference row 1: reading the target's row for both would condition a
    reference as picture-to-be-cleaned, which is the one thing the packing exists to avoid.
    """
    layout = Layout(text_len=3, latent_h=2, latent_w=2, refs=[(2, 2)])
    rows = layout.modulation_rows({"text": 0, "ref": 1, "target": 0})
    assert rows == [(0, 3, 1), (3, 4, 3), (4, 5, 0)]


# ---------------------------------------------------------------- head bank


def test_pdd_head_bank_uses_one_head_per_interval_of_its_own_schedule():
    dit = _tiny_dit()
    final = dit.final_layer
    weight = final.video_out.weight.reshape(2, -1, final.video_out.weight.shape[1])
    bias = final.video_out.bias.reshape(2, -1)

    head_first = final._bank(1.0, torch.tensor(TINY_SIGMAS))
    assert torch.equal(head_first[0], weight[0])
    assert torch.equal(head_first[1], bias[0])

    head_second = final._bank(0.5, torch.tensor(TINY_SIGMAS))
    assert torch.allclose(head_second[0], weight[0] + weight[1])
    assert torch.allclose(head_second[1], bias[0] + bias[1])


def test_pdd_head_bank_averages_the_intervals_a_wider_step_spans():
    """One step over the whole grid takes the dt-weighted mean of both heads.

    The offset row is half-applied because it covers half the distance: the bank's row
    0 is a complete head, and every later row is a delta from it.
    """
    dit = _tiny_dit()
    final = dit.final_layer
    weight = final.video_out.weight.reshape(2, -1, final.video_out.weight.shape[1])
    folded = final._bank(1.0, torch.tensor([1.0, 0.0]))
    assert torch.allclose(folded[0], weight[0] + 0.5 * weight[1])


def test_head_bank_demands_the_schedule_it_is_stepping_through():
    dit = _tiny_dit()
    with pytest.raises(ValueError, match="needs the sigma schedule"):
        dit.final_layer._bank(1.0, None)


def test_forward_needs_no_schedule_for_a_single_head():
    params = _tiny_params(pdd_steps=1, pdd_sigmas=(1.0, 0.0))
    dit = _tiny_dit(params)
    out = dit(torch.randn(1, 4, 4, 4), 1.0, torch.randn(1, 3, params.hidden_size),
              Layout(3, 4, 4))
    assert out.shape == (1, 4, 4, 4)


# ------------------------------------------------------------------ forward


def test_forward_returns_the_targets_velocity():
    dit = _tiny_dit()
    params = dit.params
    latent = torch.randn(1, params.latents_dim, 6, 6)
    context = torch.randn(1, 5, params.hidden_size)
    out = dit(latent, 0.75, context, Layout(5, 6, 6), sample_sigmas=torch.tensor(TINY_SIGMAS))

    assert out.shape == latent.shape
    assert torch.isfinite(out).all()
    # The head predicts the flow towards noise, the adapter integrates the other way.
    assert out.dtype == latent.dtype


def test_forward_cuts_the_padding_back_off_an_odd_latent():
    dit = _tiny_dit()
    params = dit.params
    latent = torch.randn(1, params.latents_dim, 5, 7)
    layout = Layout(2, 6, 8)  # the padded shape
    out = dit(latent, 1.0, torch.randn(1, 2, params.hidden_size), layout,
              sample_sigmas=torch.tensor(TINY_SIGMAS))
    assert out.shape == (1, params.latents_dim, 5, 7)


def test_forward_rejects_a_layout_built_for_another_shape():
    dit = _tiny_dit()
    with pytest.raises(ValueError, match="layout was built for"):
        dit(torch.randn(1, 4, 6, 6), 1.0, torch.randn(1, 2, 48), Layout(2, 4, 4),
            sample_sigmas=torch.tensor(TINY_SIGMAS))


def test_references_join_the_sequence_without_reaching_the_output():
    dit = _tiny_dit()
    params = dit.params
    latent = torch.randn(1, params.latents_dim, 4, 4)
    context = torch.randn(1, 3, params.hidden_size)
    ref = torch.randn(1, params.latents_dim, 2, 2)

    plain = dit(latent, 1.0, context, Layout(3, 4, 4),
                sample_sigmas=torch.tensor(TINY_SIGMAS))
    with_ref = dit(latent, 1.0, context, Layout(3, 4, 4, [(2, 2)]), refs=[ref],
                   sample_sigmas=torch.tensor(TINY_SIGMAS))
    assert with_ref.shape == plain.shape
    assert not torch.allclose(with_ref, plain)


def test_reference_rows_are_mixed_with_the_noise_their_timestep_promises():
    """Same seed, same rows: the augmentation is per-reference, not per-run.

    A reference is labelled :data:`COND_TIMESTEP`, so it is 99.9% signal — and the
    noise is drawn from a stream that restarts for every reference, which is why a
    second reference does not inherit the first one's draws.
    """
    dit = _tiny_dit()
    ref = torch.ones(1, 4, 2, 2)
    target = torch.zeros(1, 4, 2, 2)
    rows = dit._codec_rows(target, [ref, ref], seed=7)
    first, second = rows[:1], rows[1:2]

    # The sliver of noise is the one its label promises, from the stream that restarts
    # for each reference: 16 values for the one patch row a (2,2) latent makes.
    noise = torch.randn(1, 16, generator=torch.Generator("cpu").manual_seed(7))
    assert torch.allclose(first, COND_TIMESTEP * ref.reshape(1, -1) + (1 - COND_TIMESTEP) * noise)

    # The target's own rows are fed as they are.
    assert torch.equal(rows[2:], target.reshape(1, -1))

    assert torch.equal(first, second)
    assert not torch.equal(first, dit._codec_rows(target, [ref], seed=8)[:1])


def test_attention_normalizes_each_heads_query_and_key():
    """The norms exist as weights in the file, so they have to be on the path.

    A declared-but-unapplied norm loads cleanly and predicts plausible noise, so the
    assertion is that moving the weights moves the output.
    """
    dit = _tiny_dit()
    attn = dit.blocks[0].attn
    x = torch.randn(6, dit.params.hidden_size)
    base = attn(x)

    for name in ("q_norm", "k_norm"):
        norm = getattr(attn, name)
        with torch.no_grad():
            norm.weight.mul_(1.5)
        perturbed = attn(x)
        with torch.no_grad():
            norm.weight.div_(1.5)
        assert not torch.allclose(perturbed, base), f"{name} is not on the path"
        assert torch.allclose(attn(x), base)


def test_refine_text_projects_the_prompt_to_the_trunk():
    dit = _tiny_dit()
    assert dit.refine_text(torch.randn(1, 6, 16)).shape == (1, 6, 48)


def test_rope_rotates_part_of_the_head_and_passes_the_rest():
    dit = _tiny_dit()
    cos, sin = dit.rope_angles(torch.rand(4, 3, dtype=torch.float64), torch.float32)
    assert cos.shape == (4, 1, 1, dit.params.rope_dim)
    assert dit.params.rope_dim < dit.params.head_dim  # a partial rotation
    assert torch.allclose(cos.pow(2) + sin.pow(2), torch.ones_like(cos), atol=1e-5)


def test_time_embedding_interpolates_the_curve_and_clamps():
    dit = _tiny_dit()
    dit.adaln_t_table.copy_(torch.arange(15, dtype=torch.float32).reshape(5, 3))
    grid = dit.params.adaln_grid
    at_half = dit.time_embedding([0.5], torch.device("cpu"))
    row = (grid - 1) * 0.5
    assert torch.allclose(at_half[0], torch.lerp(
        torch.arange(15, dtype=torch.float32).reshape(5, 3)[int(row)],
        torch.arange(15, dtype=torch.float32).reshape(5, 3)[int(row) + 1],
        torch.tensor(row - int(row)),
    ))
    beyond = dit.time_embedding([1.0, 4.0], torch.device("cpu"))
    assert torch.equal(beyond[0], beyond[1])  # both clamp to the curve's top row


# ----------------------------------------------------------- geometry from a file


def _write_dit(tmp_path, dit, params, sigmas=TINY_SIGMAS) -> str:
    path = tmp_path / "h3-dit.safetensors"
    save_file(
        {k: v.contiguous() for k, v in dit.state_dict().items()},
        str(path),
        metadata={"config": json.dumps(
            {"transformer": {"pdd_steps": params.pdd_steps, "pdd_sigmas": list(sigmas)}}
        )},
    )
    return str(path)


def test_detect_params_reads_the_head_bank_and_its_grid(tmp_path):
    dit = _tiny_dit()
    detected = detect_params(_write_dit(tmp_path, dit, dit.params))
    assert detected == dit.params


def test_loader_restores_every_weight(tmp_path):
    dit = _tiny_dit()
    path = _write_dit(tmp_path, dit, dit.params)
    loaded = load_h3_image_dit(path, detect_params(path), device="cpu", dtype=torch.float32)

    loaded_state = loaded.state_dict()
    assert loaded_state.keys() == dit.state_dict().keys()
    for key, tensor in dit.state_dict().items():
        assert torch.equal(loaded_state[key], tensor), key


def test_load_keeps_the_fp32_islands_at_fp32(tmp_path):
    """A run in BF16 moves the trunk and leaves the modulation curve and head bank alone.

    The checkpoint stores the modulation heads at lower precision than they run at, and
    the curve is read by interpolation: both are pinned by module, not by the load call.
    """
    dit = _tiny_dit()
    path = _write_dit(tmp_path, dit, dit.params)
    loaded = load_h3_image_dit(path, detect_params(path), device="cpu", dtype=torch.bfloat16)

    assert loaded.final_layer.video_out.weight.dtype == torch.float32
    assert loaded.video_patch_proj.weight.dtype == torch.float32
    assert loaded.adaln_t_table.dtype == torch.float32
    assert loaded.blocks[0].adaln_proj.linear.weight.dtype == torch.float32
    assert loaded.blocks[0].norm1.weight.dtype == torch.bfloat16
    assert loaded.blocks[0].attn.qkv_proj.weight.dtype == torch.bfloat16


def test_detect_refuses_a_head_bank_that_disagrees_with_its_grid(tmp_path):
    dit = _tiny_dit()
    dit.video_patch_proj = torch.nn.Linear(8, 48)  # an 8-wide patch -> a 4-row bank
    path = _write_dit(tmp_path, dit, dit.params, sigmas=(1.0, 0.0))  # ... with 2 points
    with pytest.raises(ValueError, match="head bank is 4 rows"):
        detect_params(path)


def test_detect_defaults_to_one_head_without_the_metadata(tmp_path):
    dit = _tiny_dit(_tiny_params(pdd_steps=1, pdd_sigmas=(1.0, 0.0)))
    path = tmp_path / "plain.safetensors"
    write_safetensors(path, {k: v.contiguous() for k, v in dit.state_dict().items()})
    assert detect_params(str(path)).pdd_steps == 1


# ------------------------------------------------------------------ schedule


def test_the_schedule_is_the_grid_itself_at_the_distilled_step_count():
    ladder = sampling.sigma_ladder(TINY_SIGMAS, 2)
    assert ladder == pytest.approx([1.0, 0.5, 0.0])


def test_a_coarser_schedule_walks_the_grid_index_not_the_sigma_axis():
    ladder = sampling.sigma_ladder(TINY_SIGMAS, 1)
    assert ladder == pytest.approx([1.0, 0.0])

    grid = (1.0, 0.9, 0.5, 0.0)
    # Four steps over three intervals: the index lands between points 1 and 2 at step 2.
    ladder = sampling.sigma_ladder(grid, 4)
    assert ladder[0] == pytest.approx(1.0)
    assert ladder[2] == pytest.approx(0.7)
    assert ladder[-1] == 0.0
    assert all(a > b for a, b in zip(ladder, ladder[1:]))


def test_percent_to_sigma_spans_the_endpoints():
    assert sampling.percent_to_sigma(TINY_SIGMAS, 0.0) == 1.0
    assert sampling.percent_to_sigma(TINY_SIGMAS, 1.0) == 0.0
    assert sampling.percent_to_sigma(TINY_SIGMAS, 0.5) == pytest.approx(0.5)
    assert sampling.percent_to_sigma(TINY_SIGMAS, 0.25) == pytest.approx(0.75)


# --------------------------------------------------------------- presentation


def test_references_are_labelled_in_prompt_order():
    text = prompt_with_pictures("swap the hat", 2)
    assert text.index("Picture 1: ") < text.index("Picture 2: ") < text.index("swap the hat")
    assert prompt_with_pictures("a fox", 0) == "a fox"


def test_vision_token_tags_cover_the_markers_framing_the_pictures():
    """A picture's span is its patches *and* the two markers around them."""
    image_pad, frame = 151655, 151652
    ids = torch.tensor([[7, frame, image_pad, image_pad, image_pad, frame, 9, 9]])
    tags = vision_token_tags(ids[0])
    assert tags.tolist() == [1, 0, 0, 0, 0, 0, 1, 1]


def test_a_prompt_without_pictures_is_all_text():
    assert vision_token_tags(torch.tensor([1, 2, 3])).tolist() == [1, 1, 1]


# --------------------------------------------------------------- the conditioner
#
# ``encode`` is the one part of the release that is neither shape nor arithmetic: the
# processor has to hand back the token types the truncated model asks for, and the
# sequence the DiT modulates has to be the sequence that was embedded. A reduced-width
# Qwen3-VL, built from the vendored config with the same call the loader makes, is the
# cheapest thing that can say so.

TINY_HIDDEN = 64
TINY_LAYERS = 5


def _tiny_qwen_config():
    """The vendored 32B conditioner config at a width a CPU test can afford.

    Reduced rather than written out: the truncation and the presentation read the
    release's own fields (its image-token id, its vision merge size), so a hand-made
    small config could drift from them unnoticed.
    """
    import copy

    from transformers import Qwen3VLConfig

    from thenoise.utils.qwen_configs import QWEN3_VL_32B_H3_CONFIG

    config = Qwen3VLConfig.from_dict(copy.deepcopy(QWEN3_VL_32B_H3_CONFIG))
    for name, value in {
        "hidden_size": TINY_HIDDEN, "num_hidden_layers": TINY_LAYERS,
        "num_attention_heads": 4, "num_key_value_heads": 2,
        "intermediate_size": 128, "max_position_embeddings": 4096,
    }.items():
        setattr(config.text_config, name, value)
    for name, value in {
        "depth": 2, "hidden_size": 32, "intermediate_size": 64, "num_heads": 2,
        "out_hidden_size": TINY_HIDDEN,
    }.items():
        setattr(config.vision_config, name, value)
    return config


@pytest.fixture(scope="module")
def tiny_conditioner():
    from PIL import Image

    from thenoise.dit.h3_image.encoder import H3ImageTextEncoder
    from thenoise.utils.text_encoder import (
        QWEN25_TOKENIZER_CONFIG_DIR,
        QWEN3_VL_TOKENIZER_OVERRIDES,
        load_qwen3_vl_processor,
        load_qwen3_vl_tokenizer,
    )

    model = _truncate_qwen3_vl(_tiny_qwen_config(), 2).eval().requires_grad_(False)
    tokenizer, _ = load_qwen3_vl_tokenizer(
        QWEN25_TOKENIZER_CONFIG_DIR,
        max_length=4096,
        overrides=QWEN3_VL_TOKENIZER_OVERRIDES,
    )
    encoder = H3ImageTextEncoder(model, tokenizer, load_qwen3_vl_processor(tokenizer))
    return encoder.eval(), Image.new("RGB", (64, 64), "white")


def test_truncation_sets_the_depth_and_replaces_the_norm():
    """The released file is a 32B cut to 50 layers, and the final RMSNorm went with the cut."""
    config = _tiny_qwen_config()
    model = _truncate_qwen3_vl(config, 3)

    assert len(model.model.language_model.layers) == 3
    assert config.text_config.num_hidden_layers == 3
    assert isinstance(model.model.language_model.norm, torch.nn.Identity)
    assert not hasattr(model, "lm_head")


def test_layer_count_reads_the_depth_off_a_file(tmp_path):
    path = write_safetensors(
        tmp_path / "te.safetensors",
        {f"model.layers.{i}.self_attn.q_proj.weight": torch.zeros(1) for i in range(5)},
    )
    assert layer_count(path) == 5

    with pytest.raises(ValueError, match="no Qwen3-VL decoder layers"):
        layer_count(write_safetensors(tmp_path / "empty.safetensors", {"a.b": torch.zeros(1)}))


def test_the_conditioner_embeds_the_prompt_it_assembled(tiny_conditioner):
    encoder, picture = tiny_conditioner
    embeds, mask, tags = encoder.encode("make it red", [picture])

    assert embeds.shape[:1] == (1,) and embeds.shape[2] == TINY_HIDDEN
    assert mask.shape == tags.shape == embeds.shape[:2]
    assert int(mask.sum()) == mask.numel()  # one sequence, nothing padded
    assert torch.isfinite(embeds).all()

    # The picture and its two markers are one visual run, with the prompt around it
    # still text. (How many tokens a picture is worth is the processor's business, and
    # is pinned in test_vision_token_tags_cover_the_markers_framing_the_pictures.)
    visual = (tags[0] == 0).tolist()
    assert [k for k, _ in groupby(visual)] == [False, True, False]
    assert sum(visual) > 0
    assert visual.count(False) > 2  # "Picture 1: " and the prompt on both sides
    assert int((tags == 1).sum()) > 0

    # Asking again with the same prompt and picture gives the same conditioning.
    again, _, again_tags = encoder.encode("make it red", [picture])
    assert torch.equal(again, embeds) and torch.equal(again_tags, tags)


def test_an_empty_prompt_is_one_token_not_none(tiny_conditioner):
    """Nothing to tokenize is a request the pipeline can make; an empty sequence is not."""
    encoder, _ = tiny_conditioner
    embeds, _, tags = encoder.encode("", [])
    assert embeds.shape[:2] == (1, 1)
    assert tags.shape == (1, 1)
