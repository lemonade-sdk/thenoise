import pytest
import torch
from PIL import Image
from conftest import write_safetensors

from thenoise.dit.hunyuan3d import (
    CODEC_PREFIX,
    DIT_PREFIX,
    Hunyuan3DCodecParams,
    Hunyuan3DShapeCodec,
    Hunyuan3DShapeDiT,
    Hunyuan3DShapeParams,
    checkpoint_shapes,
    codec_key_map,
    codec_params,
    detect_params,
    is_hunyuan3d_shape,
    load_hunyuan3d_shape_codec,
    load_hunyuan3d_shape_dit,
    shape_key_map,
    shape_params,
    take_prefix,
)
from thenoise.dit.hunyuan3d.codec import _heads
from thenoise.dit.hunyuan3d.models import MoE
from thenoise.utils.fourier import fourier_dim, fourier_features, fourier_frequencies

TINY = dict(
    hidden_size=32,
    depth=5,
    num_heads=4,
    head_dim=8,
    in_channels=4,
    out_channels=4,
    context_dim=16,
    mlp_ratio=2,
    moe_layers=2,
    num_experts=3,
    top_k=2,
)


def tiny_params(**overrides) -> Hunyuan3DShapeParams:
    return Hunyuan3DShapeParams(**{**TINY, **overrides})


def tiny_dit(**overrides) -> Hunyuan3DShapeDiT:
    model = Hunyuan3DShapeDiT(tiny_params(**overrides))
    with torch.no_grad():
        for parameter in model.parameters():
            torch.nn.init.normal_(parameter, std=0.02)
    return model.eval()


def inputs(tokens=6, context=3):
    return (torch.zeros(1, tokens, TINY["in_channels"]),
            torch.zeros(1),
            torch.zeros(1, context, TINY["context_dim"]))


def test_one_output_token_per_latent_token():
    dit = tiny_dit()
    x, t, context = inputs(tokens=7)
    with torch.no_grad():
        out = dit(x, t, context)
    assert out.shape == (1, 7, TINY["out_channels"])


def test_the_timestep_token_leads_the_sequence_and_is_dropped_at_the_end():
    dit = tiny_dit()
    seen = []
    dit.final_layer.register_forward_hook(
        lambda _m, args, out: seen.append((args[0].shape[1], out.shape[1]))
    )
    with torch.no_grad():
        dit(*inputs(tokens=6))
    assert seen == [(7, 6)]


def test_skips_pair_the_shallow_half_with_the_deep_half_last_in_first_out():
    dit = tiny_dit()
    outputs = {}
    for layer, block in enumerate(dit.blocks):
        block.register_forward_hook(lambda _m, _a, out, layer=layer: outputs.__setitem__(layer, out))
    with torch.no_grad():
        dit(*inputs())
    # Blocks 0 and 1 save; block 3 folds the last save (1) and block 4 the one before.
    folded = {}
    for layer in (3, 4):
        dit.blocks[layer].skip_linear.register_forward_pre_hook(
            lambda _m, args, layer=layer: folded.__setitem__(layer, args[0])
        )
    with torch.no_grad():
        dit(*inputs())
    assert torch.equal(folded[3][..., :TINY["hidden_size"]], outputs[1])
    assert torch.equal(folded[4][..., :TINY["hidden_size"]], outputs[0])


def test_moe_sums_the_kept_experts_unrenormalized_over_an_ungated_shared_expert():
    torch.manual_seed(0)
    moe = MoE(dim=8, num_experts=3, top_k=2, mult=2).to(torch.float64).eval()
    tokens = torch.randn(4, 8, dtype=torch.float64)
    with torch.no_grad():
        weights, chosen = moe.gate(tokens).softmax(-1).topk(2, dim=-1)
        expected = moe.shared_experts(tokens)
        for slot, expert in enumerate(moe.experts):
            picked = chosen == slot
            expected = expected + expert(tokens) * (weights * picked).sum(1, keepdim=True)
        got = moe(tokens.unsqueeze(0))
    assert torch.allclose(got.reshape(4, 8), expected, atol=1e-10)
    assert (weights.sum(-1) < 1).all(), "two of three experts keep less than the whole mass"


def test_params_are_read_out_of_the_checkpoint_shapes():
    shapes = {
        "model.x_embedder.weight": (32, 4),
        "model.final_layer.linear.weight": (4, 32),
        "model.blocks.0.attn1.q_norm.weight": (8,),
        "model.blocks.0.attn2.to_k.weight": (32, 16),
        "model.blocks.0.mlp.fc1.weight": (64, 32),
        "model.blocks.1.mlp.fc1.weight": (64, 32),
        "model.blocks.2.mlp.fc1.weight": (64, 32),
        "model.blocks.3.moe.gate.weight": (3, 32),
        "model.blocks.3.moe.experts.0.net.0.proj.weight": (64, 32),
        "model.blocks.4.moe.gate.weight": (3, 32),
    }
    params = shape_params(shapes)
    assert (params.hidden_size, params.depth, params.num_heads, params.head_dim) == (32, 5, 4, 8)
    assert (params.in_channels, params.out_channels, params.context_dim) == (4, 4, 16)
    assert (params.mlp_ratio, params.moe_layers, params.num_experts) == (2, 2, 3)
    assert params.qkv_bias is False
    assert [params.uses_moe(layer) for layer in range(5)] == [False, False, False, True, True]
    assert [params.uses_skip(layer) for layer in range(5)] == [False, False, False, True, True]


def test_the_fingerprint_needs_the_timestep_mlp_and_qk_norm():
    assert is_hunyuan3d_shape([
        "model.t_embedder.mlp.2.weight",
        "model.blocks.0.attn1.k_norm.weight",
    ])
    assert not is_hunyuan3d_shape(["model.t_embedder.mlp.2.weight"])
    assert not is_hunyuan3d_shape(["latent_in.weight", "guidance_in.in_layer.weight"])


def test_the_plain_mlp_moves_onto_the_shared_feed_forward_slots():
    assert shape_key_map("model.blocks.0.mlp.fc1.weight").endswith("mlp.net.0.proj.weight")
    assert shape_key_map("model.blocks.0.mlp.fc2.bias").endswith("mlp.net.2.bias")
    assert shape_key_map("model.blocks.0.attn1.q_norm.weight").endswith(
        "attn1.qk_norm.query_norm.weight"
    )
    expert = "model.blocks.3.moe.experts.1.net.0.proj.weight"
    assert shape_key_map(expert) == expert


def test_take_prefix_keeps_one_submodule_of_a_shared_file():
    state = {"model.x": 1, "vae.y": 2, "conditioner.main_image_encoder.model.z": 3}
    assert take_prefix("vae.")(state) == {"y": 2}


def test_a_shared_file_loads_its_dit_and_nothing_else(tmp_path):
    dit = tiny_dit()
    tensors = {}
    for key, value in dit.state_dict().items():
        key = key.replace(".mlp.net.0.proj.", ".mlp.fc1.")
        key = key.replace(".mlp.net.2.", ".mlp.fc2.")
        key = key.replace(".qk_norm.query_norm.", ".q_norm.")
        key = key.replace(".qk_norm.key_norm.", ".k_norm.")
        tensors[DIT_PREFIX + key] = value
    tensors["vae.post_kl.weight"] = torch.ones(4, 4)
    tensors["conditioner.main_image_encoder.model.layernorm.weight"] = torch.ones(32)
    path = write_safetensors(tmp_path / "hunyuan3d.safetensors", tensors)

    loaded = load_hunyuan3d_shape_dit(path, detect_params(path), device="cpu",
                                      dtype=torch.float32)

    assert torch.equal(loaded.x_embedder.weight, dit.x_embedder.weight)
    assert torch.equal(loaded.blocks[0].mlp.net[0].proj.weight, dit.blocks[0].mlp.net[0].proj.weight)
    assert torch.equal(loaded.blocks[3].moe.experts[0].net[0].proj.weight,
                       dit.blocks[3].moe.experts[0].net[0].proj.weight)
    assert not hasattr(loaded, "vae")


# ---------------------------------------------------------------------- shape codec

TINY_CODEC = dict(
    width=32,
    embed_dim=4,
    num_heads=4,
    head_dim=8,
    layers=2,
    mlp_ratio=2,
    geo_mlp_ratio=2,
    num_freqs=2,
    qkv_bias=False,
    qk_norm=True,
    geo_ln_post=True,
)


def tiny_codec(**overrides) -> Hunyuan3DShapeCodec:
    params = Hunyuan3DCodecParams(**{**TINY_CODEC, **overrides})
    model = Hunyuan3DShapeCodec(params)
    with torch.no_grad():
        for parameter in model.parameters():
            torch.nn.init.normal_(parameter, std=0.02)
    return model.eval()


def test_fourier_features_put_the_coordinate_first():
    coords = torch.tensor([[0.5, -0.25, 1.0]])
    out = fourier_features(coords, num_freqs=3)
    assert out.shape == (1, 21) == (1, fourier_dim(3, 3))
    assert torch.equal(out[:, :3], coords)
    sin, cos = out[:, 3:12], out[:, 12:21]
    angles = coords[:, :, None] * fourier_frequencies(3, device=coords.device)
    assert torch.allclose(sin, angles.sin().flatten(1), atol=1e-6)
    assert torch.allclose(cos, angles.cos().flatten(1), atol=1e-6)
    assert fourier_features(coords, 3, include_input=False).shape == (1, 18)


def test_fourier_frequencies_double_and_optionally_scale_by_pi():
    assert torch.equal(fourier_frequencies(3, include_pi=False), torch.tensor([1.0, 2.0, 4.0]))
    assert torch.allclose(fourier_frequencies(3, include_pi=True) / torch.pi,
                          torch.tensor([1.0, 2.0, 4.0]))


def test_qkv_is_interleaved_inside_each_head():
    heads, per = 4, 8
    q, k, v = (torch.randn(2, 5, heads * per) for _ in range(3))
    fused = torch.stack((q.view(2, 5, heads, per), k.view(2, 5, heads, per),
                         v.view(2, 5, heads, per)), dim=-2).reshape(2, 5, 3 * heads * per)
    got_q, got_k, got_v = _heads(fused, heads, 3)
    assert got_q.shape == (2, heads, 5, per)
    assert torch.equal(got_q.transpose(1, 2).reshape(2, 5, heads * per), q)
    assert torch.equal(got_v.transpose(1, 2).reshape(2, 5, heads * per), v)


def test_the_codec_returns_one_logit_per_queried_point():
    codec = tiny_codec()
    latents = torch.zeros(2, 5, TINY_CODEC["embed_dim"])
    queries = torch.zeros(2, 7, 3)
    with torch.no_grad():
        out = codec(latents, queries)
    assert out.shape == (2, 7)
    assert torch.equal(out, codec.sdf(codec.tokens(latents), queries))


def test_the_codec_drives_the_octree_through_one_point_to_one_sdf():
    from thenoise.mesh.octree import sample_sdf

    codec = tiny_codec()
    with torch.no_grad():
        tokens = codec.tokens(torch.randn(1, 6, TINY_CODEC["embed_dim"]))
        volume = sample_sdf(
            lambda points: codec.sdf(tokens, points[None])[0],
            32, device="cpu", bounds=1.0, start=4, chunk=97)
    assert volume.queries > 0
    assert volume.values.shape == (33, 33, 33)
    assert volume.cells.shape == (32, 32, 32)


def test_the_codec_norms_heads_with_layernorm_not_rmsnorm():
    norm = tiny_codec().transformer.resblocks[0].attn.attention.query_norm
    assert isinstance(norm, torch.nn.LayerNorm) and norm.bias is not None


def test_codec_params_are_read_out_of_the_checkpoint_shapes():
    width, embed, heads_dim, freqs = 32, 4, 8, 2
    shapes = {
        "vae.post_kl.weight": (width, embed),
        "vae.transformer.resblocks.0.attn.attention.q_norm.weight": (heads_dim,),
        "vae.transformer.resblocks.0.mlp.c_fc.weight": (width * 2, width),
        "vae.geo_decoder.query_proj.weight": (width, fourier_dim(3, freqs)),
        "vae.geo_decoder.cross_attn_decoder.mlp.c_fc.weight": (width * 3, width),
    }
    params = codec_params(shapes)
    assert (params.width, params.embed_dim, params.num_heads, params.head_dim) == (32, 4, 4, 8)
    assert (params.layers, params.mlp_ratio, params.geo_mlp_ratio) == (1, 2, 3)
    assert params.num_freqs == 2
    assert params.qkv_bias is False and params.qk_norm is True
    assert params.geo_ln_post is False


def test_codec_key_map_finds_the_shared_slots():
    assert codec_key_map("vae.transformer.resblocks.0.mlp.c_fc.weight").endswith(
        "mlp.net.0.proj.weight"
    )
    assert codec_key_map("vae.transformer.resblocks.0.mlp.c_proj.weight").endswith(
        "mlp.net.2.weight"
    )
    assert codec_key_map("vae.geo_decoder.cross_attn_decoder.attn.attention.k_norm.bias"
                         ).endswith("attn.attention.key_norm.bias")


def test_the_encode_half_is_left_where_it_is(tmp_path):
    codec = tiny_codec()
    tensors = {}
    for key, value in codec.state_dict().items():
        key = key.replace(".mlp.net.0.proj.", ".mlp.c_fc.")
        key = key.replace(".mlp.net.2.", ".mlp.c_proj.")
        key = key.replace(".attention.query_norm.", ".attention.q_norm.")
        key = key.replace(".attention.key_norm.", ".attention.k_norm.")
        tensors[CODEC_PREFIX + key] = value
    tensors[CODEC_PREFIX + "encoder.input_proj.weight"] = torch.ones(32, 15)
    tensors[CODEC_PREFIX + "pre_kl.weight"] = torch.ones(8, 32)
    tensors["model.x_embedder.weight"] = torch.ones(4, 4)
    path = write_safetensors(tmp_path / "codec.safetensors", tensors)

    loaded = load_hunyuan3d_shape_codec(path, codec_params(checkpoint_shapes(path)),
                                        device="cpu", dtype=torch.float32)

    assert torch.equal(loaded.post_kl.weight, codec.post_kl.weight)
    assert torch.equal(loaded.geo_decoder.output_proj.weight,
                       codec.geo_decoder.output_proj.weight)
    assert not hasattr(loaded, "encoder")


# ---------------------------------------------------------------------- conditioner

from thenoise.dit.hunyuan3d import CONDITIONER_PREFIX, Hunyuan3DConditionerParams
from thenoise.dit.hunyuan3d import conditioner_key_map, conditioner_params
from thenoise.dit.hunyuan3d.conditioner import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    Hunyuan3DImageEncoder,
    dino_normalize,
    load_hunyuan3d_image_encoder,
)

TINY_DINO = dict(hidden_size=64, layers=2, num_heads=1, mlp_dim=128, patch_size=14,
                 image_size=56)


def tiny_dino(**overrides) -> Hunyuan3DImageEncoder:
    params = Hunyuan3DConditionerParams(**{**TINY_DINO, **overrides})
    model = Hunyuan3DImageEncoder(params)
    with torch.no_grad():
        for parameter in model.parameters():
            torch.nn.init.normal_(parameter, std=0.02)
    return model.eval()


def picture(width=56, height=56, grey=128) -> Image.Image:
    return Image.new("RGB", (width, height), (grey, grey, grey))


def test_the_tower_geometry_is_read_out_of_the_header():
    shapes = {
        "conditioner.main_image_encoder.model.embeddings.patch_embeddings.projection.weight":
            (1024, 3, 14, 14),
        "conditioner.main_image_encoder.model.embeddings.position_embeddings": (1, 1370, 1024),
        "conditioner.main_image_encoder.model.encoder.layer.0.mlp.fc1.weight": (4096, 1024),
    }
    params = conditioner_params(shapes)
    assert params.hidden_size == 1024 and params.layers == 1
    assert params.patch_size == 14 and params.image_size == 518
    assert params.mlp_dim == 4096 and params.num_heads == 16
    assert params.num_tokens == 1370
    assert params.qkv_bias is False


def test_a_position_count_that_is_no_grid_is_refused():
    shapes = {
        "conditioner.main_image_encoder.model.embeddings.patch_embeddings.projection.weight":
            (1024, 3, 14, 14),
        "conditioner.main_image_encoder.model.embeddings.position_embeddings": (1, 1000, 1024),
        "conditioner.main_image_encoder.model.encoder.layer.0.mlp.fc1.weight": (4096, 1024),
    }
    with pytest.raises(ValueError, match="square grid"):
        conditioner_params(shapes)


def test_the_saved_attention_names_find_the_transformers_slots():
    assert conditioner_key_map(
        "encoder.layer.0.attention.attention.value.bias").endswith("attention.v_proj.bias")
    assert conditioner_key_map(
        "encoder.layer.0.attention.output.dense.weight").endswith("attention.o_proj.weight")


def test_dino_normalize_uses_the_pretraining_statistics():
    out = dino_normalize(torch.zeros(3, 2, 2))
    assert torch.allclose(out.mean(dim=(1, 2)),
                          (torch.tensor(0.5) - torch.tensor(IMAGENET_MEAN))
                          / torch.tensor(IMAGENET_STD))
    assert dino_normalize(torch.zeros(2, 3, 4, 4)).shape == (2, 3, 4, 4)


def test_the_tower_gives_one_token_per_patch_plus_the_cls():
    with torch.no_grad():
        assert tiny_dino()([picture()], torch.float32).shape == (1, 17, TINY_DINO["hidden_size"])
        no_cls = tiny_dino(use_cls_token=False)([picture()], torch.float32)
        assert no_cls.shape == (1, 16, TINY_DINO["hidden_size"])


def test_the_tower_takes_any_whole_number_of_patches():
    # The engine fits a reference by area and never crops, so the tower interpolates positions.
    encoder = tiny_dino()
    with torch.no_grad():
        assert encoder([picture(70, 42)], torch.float32).shape == (1, 16, 64)
        assert encoder([picture(), picture()], torch.float32).shape == (2, 17, 64)
    with pytest.raises(ValueError, match="one picture size"):
        encoder.pixels([picture(), picture(70, 42)], torch.float32)


def test_the_unconditional_context_is_zeros_like_the_conditioned_one():
    like = torch.ones(2, 17, 64)
    null = tiny_dino().unconditional(like)
    assert null.shape == like.shape and torch.equal(null, torch.zeros_like(like))


def test_the_conditioner_loads_under_its_own_prefix(tmp_path):
    encoder = tiny_dino()
    tensors = {}
    for key, value in encoder.state_dict().items():
        for ours, saved in ((".attention.q_proj.", ".attention.attention.query."),
                            (".attention.k_proj.", ".attention.attention.key."),
                            (".attention.v_proj.", ".attention.attention.value."),
                            (".attention.o_proj.", ".attention.output.dense.")):
            key = key.replace(ours, saved)
        tensors[CONDITIONER_PREFIX + key] = value
    tensors["model.x_embedder.weight"] = torch.ones(4, 4)
    path = write_safetensors(tmp_path / "cond.safetensors", tensors)

    loaded = load_hunyuan3d_image_encoder(path, conditioner_params(checkpoint_shapes(path)),
                                          device="cpu", dtype=torch.float32)

    assert torch.equal(loaded.embeddings.cls_token, encoder.embeddings.cls_token)
    assert torch.equal(loaded.encoder.layer[1].mlp.fc2.weight,
                       encoder.encoder.layer[1].mlp.fc2.weight)
    assert not any(p.requires_grad for p in loaded.parameters())
    assert not loaded.training
