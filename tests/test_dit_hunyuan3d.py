import pytest
import torch
from conftest import write_safetensors

from thenoise.dit.hunyuan3d import (
    DIT_PREFIX,
    Hunyuan3DShapeDiT,
    Hunyuan3DShapeParams,
    detect_params,
    is_hunyuan3d_shape,
    load_hunyuan3d_shape_dit,
    shape_key_map,
    shape_params,
    take_prefix,
)
from thenoise.dit.hunyuan3d.models import MoE

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
