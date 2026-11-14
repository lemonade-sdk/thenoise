"""Wan 2.2: what the released variants decide for themselves -- checkpoint key
layout, transformer and text-encoder shapes, the causal codec, the schedule, the
adapter lifecycle and the two-expert A14B path.
"""

from __future__ import annotations

import io
import re
import types
from pathlib import Path

import pytest
import sentencepiece as spm
import torch
from conftest import stub_wan_components, wan_adapter, write_wan_dit, write_wan_vae
from PIL import Image
from safetensors.torch import save_file

from thenoise.dit.quantized import QuantizedLinear
from thenoise.dit.wan_video.checkpoint import load_wan_native_transformer
from thenoise.dit.wan_video.model import (
    Wan22Params,
    WanNativeDiT,
    WanNativeRoPE,
    detect_params,
)
from thenoise.dit.wan_video.sampling import FLOW_SHIFT, get_sigmas
from thenoise.errors import ModelReloadNeeded
from thenoise.models import resolve
from thenoise.models.base import DiffusionModel
from thenoise.models.config import ModelConfig, VideoSamplingParams
from thenoise.models.wan_video import VARIANTS, Wan22Model, variant_for
from thenoise.runtime import ModelPaths, Runtime, Settings
from thenoise.samplers import Step, create_sampler
from thenoise.text_encoders.wan_video import (
    COMPUTE_DTYPE,
    WanUMT5,
    _T5Attention,
    _T5GatedFFN,
    load_wan_text_encoder,
)
from thenoise.utils.wan_weights import wan_dit_key, wan_vae21_key, wan_vae_key
from thenoise.vae.qwen_image import (
    _WAN21_FAMILY_ARCH,
    QWEN_IMAGE_LATENTS_MEAN,
    QWEN_IMAGE_LATENTS_STD,
    AutoencoderKLQwenImage,
)
from thenoise.vae.wan_common import patchify, unpatchify
from thenoise.vae.wan_video import (
    _RELEASES,
    AutoencoderKLWanVideo,
    _AttentionBlock,
    _CausalConv3D,
    _comfy_vae_config,
    _DupUp3D,
    _ResidualDownBlock,
    _ResidualUpBlock,
    _vae_release,
    _wan21_vae_config,
    detect_latent_width,
    load_wan_video_vae,
)
from thenoise.video.controller import VideoPipelineController
from thenoise.video.jobs import Cancellation, VideoCancelled
from thenoise.video.spec import VideoSpec

# ----------------------------------------------------------------- Comfy-Org checkpoint key layout

def _original_vae_key(key: str, depth: int = 1) -> str:
    k = key
    if k.startswith("quant_conv."):
        return "conv1." + k[11:]
    if k.startswith("post_quant_conv."):
        return "conv2." + k[16:]
    for part in ("encoder", "decoder"):
        swaps = {
            f"{part}.conv_in.": f"{part}.conv1.",
            f"{part}.norm_out.": f"{part}.head.0.",
            f"{part}.conv_out.": f"{part}.head.2.",
            f"{part}.mid_block.resnets.0.": f"{part}.middle.0.",
            f"{part}.mid_block.attentions.0.": f"{part}.middle.1.",
            f"{part}.mid_block.resnets.1.": f"{part}.middle.2.",
        }
        for prefix, dest in swaps.items():
            if k.startswith(prefix):
                k = dest + k[len(prefix):]
                break
    match = re.match(r"^(encoder)\.down_blocks\.(\d+)\.(resnets|downsampler)\.?([\d]*)(?:\.)?(.*)$", k)
    if match:
        part, i, typ, j, rest = match.groups()
        index = int(j) if typ == "resnets" else depth
        k = f"encoder.downsamples.{i}.downsamples.{index}.{rest}"
    match = re.match(r"^(decoder)\.up_blocks\.(\d+)\.(resnets|upsampler)\.?([\d]*)(?:\.)?(.*)$", k)
    if match:
        part, i, typ, j, rest = match.groups()
        index = int(j) if typ == "resnets" else depth + 1
        k = f"decoder.upsamples.{i}.upsamples.{index}.{rest}"
    k = k.replace(".norm1.", ".residual.0.")
    k = k.replace(".norm2.", ".residual.3.")
    k = k.replace(".conv1.", ".residual.2.") if ".downsamples." in k or ".upsamples." in k or ".middle." in k else k
    k = k.replace(".conv2.", ".residual.6.")
    k = k.replace(".conv_shortcut.", ".shortcut.")
    return k


def test_vae_comfy_key_map_full_topology():
    cfg = _comfy_vae_config(dict(base_dim=4, decoder_base_dim=4, num_res_blocks=1))
    with torch.device("meta"):
        module = AutoencoderKLWanVideo(cfg)
    names = set(module.state_dict())
    raw = {_original_vae_key(k) for k in names}
    mapped = {wan_vae_key(k, num_res_blocks=1) for k in raw}
    if mapped != names:
        print("missing", sorted(names - mapped)[:25]); print("extra", sorted(mapped-names)[:25])
    assert len(raw) == len(names)
    assert mapped == names


def test_dit_comfy_names():
    expected = {
        "text_embedding.0.weight": "condition_embedder.text_linear_1.weight",
        "time_embedding.2.bias": "condition_embedder.time_linear_2.bias",
        "time_projection.1.bias": "condition_embedder.time_proj.bias",
        "head.modulation": "scale_shift_table",
        "blocks.0.norm2.weight": "blocks.0.norm3.weight",
        "blocks.0.norm3.weight": "blocks.0.norm2.weight",
        "blocks.10.self_attn.q.weight": "blocks.10.attn1.to_q.weight",
        "blocks.5.cross_attn.o.bias": "blocks.5.attn2.to_out.bias",
        "blocks.2.ffn.0.weight": "blocks.2.ffn_in.weight",
        "blocks.2.ffn.2.weight": "blocks.2.ffn_out.weight",
        "blocks.7.self_attn.o.weight": "blocks.7.attn1.to_out.weight",
    }
    assert {k: wan_dit_key(k) for k in expected} == expected


# ----------------------------------------------------------------- Transformer checkpoint

#: Small enough to build in a test, and still legal geometry for `detect_params`:
#: one 128-wide head, two blocks.
DIT = Wan22Params(in_channels=48, out_channels=48, dim=128, num_layers=2,
                   ffn_dim=256, freq_dim=32, text_dim=32, patch_size=(1, 2, 2))


def _raw_key(key: str) -> str:
    """The inverse of `wan_dit_key`, to write a checkpoint the way Comfy-Org does."""
    roots = {
        "condition_embedder.text_linear_1.": "text_embedding.0.",
        "condition_embedder.text_linear_2.": "text_embedding.2.",
        "condition_embedder.time_linear_1.": "time_embedding.0.",
        "condition_embedder.time_linear_2.": "time_embedding.2.",
        "condition_embedder.time_proj.": "time_projection.1.",
        "proj_out.": "head.head.",
        "scale_shift_table": "head.modulation",
    }
    for dst, src in roots.items():
        if key.startswith(dst):
            return src + key[len(dst):]
    if not key.startswith("blocks."):
        return key
    parts = key.split(".", 2)
    prefix, suffix = ".".join(parts[:2]) + ".", parts[2]
    if suffix == "scale_shift_table":
        return prefix + "modulation"
    # ComfyUI norm2 is the FFN norm, norm3 the cross-attention one.
    if suffix.startswith("norm2."):
        return prefix + "norm3." + suffix[6:]
    for dst, src in (
        ("attn1.to_q.", "self_attn.q."), ("attn1.to_k.", "self_attn.k."),
        ("attn1.to_v.", "self_attn.v."), ("attn1.to_out.", "self_attn.o."),
        ("attn1.norm_q.", "self_attn.norm_q."), ("attn1.norm_k.", "self_attn.norm_k."),
        ("attn2.to_q.", "cross_attn.q."), ("attn2.to_k.", "cross_attn.k."),
        ("attn2.to_v.", "cross_attn.v."), ("attn2.to_out.", "cross_attn.o."),
        ("attn2.norm_q.", "cross_attn.norm_q."), ("attn2.norm_k.", "cross_attn.norm_k."),
        ("ffn_in.", "ffn.0."), ("ffn_out.", "ffn.2."),
    ):
        if suffix.startswith(dst):
            return prefix + src + suffix[len(dst):]
    return key


_MATMUL_WEIGHTS = re.compile(r"^blocks\.\d+\.(?:self|cross)_attn\.[qkvo]\.weight$"
                             r"|^blocks\.\d+\.ffn\.[02]\.weight$")


def _write_dit(root, *, name="wan2.2_ti2v_5B_fp16.safetensors", missing=None, wrong_shape=None,
           mixed=False):
    checkpoint = root / "split_files" / "diffusion_models" / name
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with torch.random.fork_rng():
        torch.manual_seed(123)
        source = WanNativeDiT(DIT)
    with torch.no_grad():
        for param in source.parameters():
            param.uniform_(-0.1, 0.1)
    state = {}
    for module_name, tensor in source.state_dict().items():
        raw = _raw_key(module_name)
        assert wan_dit_key(raw) == module_name, (module_name, raw, wan_dit_key(raw))
        precision = torch.float16
        if mixed:
            precision = torch.float16 if _MATMUL_WEIGHTS.match(raw) else torch.float32
        state[raw] = tensor.detach().to(precision).contiguous()
    if missing:
        state.pop(missing)
    if wrong_shape:
        state[wrong_shape] = torch.zeros(2, 2, dtype=torch.float16)
    save_file(state, str(checkpoint))
    return state, checkpoint


def test_params_are_read_from_the_header(tmp_path):
    _, path = _write_dit(tmp_path)
    assert detect_params(str(path)) == DIT


def test_a_wrong_checkpoint_is_named_rather_than_guessed(tmp_path):
    state, path = _write_dit(tmp_path)
    state["patch_embedding.weight"] = state["patch_embedding.weight"][:, :3].contiguous()
    save_file(state, str(path))
    with pytest.raises(ValueError, match="no Wan variant"):
        detect_params(str(path))


def test_a_file_that_is_not_a_wan_transformer_is_refused(tmp_path):
    save_file({"layers.0.weight": torch.zeros(2, 2)}, str(tmp_path / "other.safetensors"))
    with pytest.raises(ValueError, match="patch_embedding"):
        detect_params(str(tmp_path / "other.safetensors"))


def test_wan_rope_uses_shared_builder_and_caches_by_geometry():
    from thenoise.utils.rope import RopeCache

    rope = WanNativeRoPE(DIT.head_dim, DIT.patch_size)
    assert isinstance(rope._frequencies, RopeCache)

    input_video = torch.zeros(1, 48, 3, 4, 6)
    first = rope(input_video)
    assert rope(input_video) is first
    assert first.shape == (1, 18, DIT.head_dim // 2, 2, 2)

    head_dim = DIT.head_dim
    hw = 2 * (head_dim // 6)
    cos_axes, sin_axes = [], []
    for part in (head_dim - 2 * hw, hw, hw):
        inv_freq = 1 / 10000 ** (torch.arange(0, part, 2, dtype=torch.float64) / part)
        angles = torch.outer(torch.arange(64, dtype=torch.float64), inv_freq)
        cos_axes.append(angles.cos().float())
        sin_axes.append(angles.sin().float())
    cos = torch.cat([cos_axes[0][:3, None, None].expand(3, 2, 3, -1),
                     cos_axes[1][None, :2, None].expand(3, 2, 3, -1),
                     cos_axes[2][None, None, :3].expand(3, 2, 3, -1)], dim=-1)
    sin = torch.cat([sin_axes[0][:3, None, None].expand(3, 2, 3, -1),
                     sin_axes[1][None, :2, None].expand(3, 2, 3, -1),
                     sin_axes[2][None, None, :3].expand(3, 2, 3, -1)], dim=-1)
    expected = torch.stack((cos, -sin, sin, cos), dim=-1).reshape(first.shape)
    torch.testing.assert_close(first, expected, rtol=1e-5, atol=1e-5)

    rope.to("cpu")
    second = rope(input_video)
    assert second is not first
    torch.testing.assert_close(second, first, rtol=0, atol=0)
    assert rope(torch.zeros(1, 48, 1, 4, 4)).shape == (1, 4, DIT.head_dim // 2, 2, 2)


def test_comfy_checkpoint_dtype_and_forward(tmp_path):
    raw, path = _write_dit(tmp_path)
    model = load_wan_native_transformer(str(path), "cpu", dtype=torch.float32)
    assert isinstance(model, WanNativeDiT)
    assert not any(p.is_meta for p in model.parameters())
    assert model.patch_embedding.weight.dtype == torch.float32
    assert torch.equal(model.proj_out.weight, raw["head.head.weight"].float())
    x = torch.randn(1, 48, 3, 4, 6)
    ctx = torch.randn(1, 5, DIT.text_dim)
    with torch.no_grad():
        a = model(x, torch.tensor([500.0]), ctx)
        b = model(x, torch.full((1, 18), 500.0), ctx)
    assert a.shape == b.shape == x.shape
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    # One timestep per token, all equal, conditions the model exactly like one per batch.
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)


def test_bf16_comfy_precision_exceptions(tmp_path):
    _, path = _write_dit(tmp_path)
    model = load_wan_native_transformer(str(path), "cpu", dtype=torch.bfloat16)
    assert model.proj_out.weight.dtype == torch.bfloat16
    assert model.patch_embedding.weight.dtype == torch.float32
    assert model.condition_embedder.time_linear_1.weight.dtype == torch.float32
    assert model.blocks[0].scale_shift_table.dtype == torch.float32
    x = torch.randn(1, 48, 1, 4, 4, dtype=torch.bfloat16)
    context = torch.randn(1, 4, DIT.text_dim, dtype=torch.bfloat16)
    with torch.no_grad():
        output = model(x, torch.tensor([500.0]), context)
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()


def test_fp16_compute_dtype_preserves_fp32_patch_embedding(tmp_path):
    _, path = _write_dit(tmp_path)
    model = load_wan_native_transformer(str(path), "cpu", dtype=torch.float16)
    assert model.patch_embedding.weight.dtype == torch.float32
    assert model.condition_embedder.time_linear_1.weight.dtype == torch.float32
    assert model.condition_embedder.time_proj.weight.dtype == torch.float16
    assert model.dtype == torch.float16

    x = torch.randn(1, 48, 1, 4, 4, dtype=torch.float32).to(model.dtype)
    context = torch.randn(1, 4, DIT.text_dim, dtype=torch.float32).to(model.dtype)
    with torch.no_grad():
        output = model(x, torch.full((1, 4), 500.0), context)
    assert output.dtype == torch.float16
    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_a_plain_checkpoint_runs_at_its_own_precision(tmp_path):
    """D9: the file's dtype is the compute dtype unless the caller overrides it."""
    _, path = _write_dit(tmp_path)
    assert load_wan_native_transformer(str(path), "cpu").dtype == torch.float16


def test_a_mixed_checkpoint_loads_at_one_trunk_dtype(tmp_path):
    """A Linear that is Half in its weight and Float in its bias cannot run.

    ``Wan2_2-TI2V-5B-Turbo_fp16.safetensors`` is written like this.
    """
    _, path = _write_dit(tmp_path, mixed=True)
    model = load_wan_native_transformer(str(path), "cpu")
    assert model.dtype == torch.float16
    for module in model.modules():
        weight, bias = getattr(module, "weight", None), getattr(module, "bias", None)
        if torch.is_tensor(weight) and torch.is_tensor(bias):
            assert bias.dtype == weight.dtype, module
    assert model.patch_embedding.weight.dtype == torch.float32
    assert model.blocks[0].norm2.weight.dtype == torch.float32

    x = torch.randn(1, 48, 1, 4, 4, dtype=torch.float16)
    context = torch.randn(1, 4, DIT.text_dim, dtype=torch.float16)
    with torch.no_grad():
        output = model(x, torch.tensor([500.0]), context)
    assert output.dtype == torch.float16
    assert torch.isfinite(output).all()


def test_the_file_it_is_given_is_the_file_it_reads(tmp_path):
    """Weights may be renamed freely; nothing is looked up by filename."""
    _, path = _write_dit(tmp_path)
    renamed = tmp_path / "whatever-i-called-this.safetensors"
    path.rename(renamed)
    model = load_wan_native_transformer(str(renamed), "cpu")
    assert model.params == DIT


def test_checkpoint_rejects_missing_and_wrong_shapes(tmp_path):
    # Not patch_embedding: a structural key is refused earlier, by detect_params.
    _write_dit(tmp_path, missing="blocks.0.self_attn.q.weight")
    with pytest.raises(RuntimeError, match="Missing key"):
        load_wan_native_transformer(str(_checkpoint(tmp_path)), "cpu")
    _write_dit(tmp_path, wrong_shape="blocks.0.self_attn.q.weight")
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_wan_native_transformer(str(_checkpoint(tmp_path)), "cpu")


def _checkpoint(root):
    return root / "split_files" / "diffusion_models" / "wan2.2_ti2v_5B_fp16.safetensors"


def test_official_5b_topology_is_constructible_on_meta():
    params = Wan22Params(in_channels=48, out_channels=48, dim=3072, num_layers=30,
                         ffn_dim=14336, freq_dim=256, text_dim=4096, patch_size=(1, 2, 2))
    with torch.device("meta"):
        model = WanNativeDiT(params)
    assert params.heads == 24
    assert model.patch_embedding.weight.shape == (3072, 48, 1, 2, 2)
    assert len(model.blocks) == 30
    assert model.blocks[0].ffn_in.weight.shape == (14336, 3072)
    assert all(p.is_meta for p in model.parameters())


# ----------------------------------------------------------------- UMT5 text encoder

CFG = dict(width=16, ff_dim=32, heads=4, layers=2, vocab_size=64)


def small_checkpoint(root: Path, *, missing=False):
    dest = root / "split_files" / "text_encoders"
    dest.mkdir(parents=True)
    training = root / "sentences.txt"
    training.write_text(("A fox walking in snow. Another video of a fox.\n" * 20), encoding="utf-8")
    spm.SentencePieceTrainer.train(input=str(training), model_prefix=str(root / "sp"),
                                   vocab_size=64, hard_vocab_limit=False,
                                   pad_id=0, eos_id=1, unk_id=2, bos_id=-1,
                                   max_sentence_length=1000)
    with torch.random.fork_rng():
        torch.manual_seed(13)
        model = WanUMT5(**CFG)
    tensors = {key: value.detach().contiguous().half() for key, value in model.state_dict().items()}
    if missing:
        tensors.pop(next(iter(tensors)))
    tensors["spiece_model"] = torch.tensor(list((root / "sp.model").read_bytes()), dtype=torch.uint8)
    path = dest / "umt5_xxl_fp16.safetensors"
    save_file(tensors, str(path))
    return model, path



def test_t5_shared_attention_preserves_unscaled_scores_and_mask():
    model = _T5Attention(width=16, heads=4).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.uniform_(-0.1, 0.1)
        x = torch.randn(1, 6, 16)
        mask = torch.tensor([[True, True, True, False, False, False]])
        q, k, v = (layer(x).reshape(1, 6, 4, 4).transpose(1, 2)
                   for layer in (model.q, model.k, model.v))
        bias = model.relative_attention_bias(model._buckets(6, x.device))
        bias = bias.permute(2, 0, 1)[None].to(x.dtype)
        bias = bias.masked_fill(~mask[:, None, None, :], torch.finfo(x.dtype).min)
        expected = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, scale=1.0
        ).transpose(1, 2).reshape(1, 6, 16)
        torch.testing.assert_close(model(x, mask), model.o(expected), rtol=1e-5, atol=1e-5)

def test_t5_forward_shape_mask_and_no_transformers():
    model = WanUMT5(**CFG).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.uniform_(-0.05, 0.05)
    ids = torch.randint(0, 64, (1, 8))
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.bool)
    with torch.no_grad():
        out = model(ids, mask)
    assert out.shape == (1, 8, 16)
    assert torch.isfinite(out).all()
    assert not out[:, -3:].any()


def test_text_encoder_comfy_checkpoint_and_sentencepiece(tmp_path):
    source, path = small_checkpoint(tmp_path)
    model, tokenizer = load_wan_text_encoder(str(path), "cpu", dtype=torch.float32, params=CFG)
    assert not any(p.is_meta for p in model.parameters())
    assert torch.equal(model.shared.weight, source.shared.weight.half().float())
    ids, mask = tokenizer.encode("A fox", max_length=16)
    assert ids.shape == mask.shape == (1, 16)
    assert ids[0, mask[0]][-1] == 1
    assert not mask[0, -1]
    with torch.no_grad():
        out = model(ids, mask)
    assert out.shape == (1, 16, 16)


def test_bf16_umt5_gated_ffn_avoids_fp16_overflow():
    """Gated products can exceed FP16's range before the down projection.

    UMT5 is loaded in BF16; the DiT still uses FP16 after the final T5 norm.
    """
    assert COMPUTE_DTYPE is torch.bfloat16
    ffn = _T5GatedFFN(width=2, ff_dim=2).to(torch.bfloat16).eval()
    with torch.no_grad():
        ffn.wi_0.weight.fill_(200)
        ffn.wi_1.weight.fill_(200)
        ffn.wo.weight.fill_(1e-5)
        x = torch.ones((1, 1, 2), dtype=torch.bfloat16)
        out = ffn(x)
    assert out.dtype is torch.bfloat16
    assert bool(torch.isfinite(out).all())
    assert float(out.abs().max()) > 0


def test_text_missing_weights_rejected(tmp_path):
    _, path = small_checkpoint(tmp_path, missing=True)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_wan_text_encoder(str(path), "cpu", params=CFG)


# ----------------------------------------------------------------- Causal 3D codec

VAE_GEOMETRY = dict(base_dim=4, decoder_base_dim=4, num_res_blocks=1)


def _vae():
    with torch.random.fork_rng():
        torch.manual_seed(9)
        return AutoencoderKLWanVideo(_comfy_vae_config(VAE_GEOMETRY)).eval()


def _write_vae(root, *, missing=False, wrong_shape=False):
    path = root / "split_files" / "vae" / "wan2.2_vae.safetensors"
    path.parent.mkdir(parents=True)
    source = _vae()
    raw = {_original_vae_key(key): tensor.detach().contiguous().half()
           for key, tensor in source.state_dict().items()}
    if missing:
        raw.pop(next(iter(raw)))
    if wrong_shape:
        raw[next(iter(raw))] = torch.zeros(2, 2, dtype=torch.float16)
    save_file(raw, str(path))
    return path, source



@pytest.mark.parametrize("channels", [16, 320])
def test_spatial_attention_uses_shared_wan22_kernel(monkeypatch, channels):
    from thenoise.vae.wan22 import Wan22AttentionBlock

    block = _AttentionBlock(channels).eval()
    assert isinstance(block, Wan22AttentionBlock)
    assert set(block.state_dict()) == {"norm.gamma", "to_qkv.weight", "to_qkv.bias",
                                       "proj.weight", "proj.bias"}
    x = torch.randn(1, channels, 2, 4, 4)
    with torch.no_grad():
        frames = x.permute(0, 2, 1, 3, 4).reshape(2, channels, 4, 4)
        normalized = block.norm(frames)
        q, k, v = block.to_qkv(normalized).chunk(3, dim=1)
        q, k, v = (value.reshape(2, channels, 16).transpose(1, 2) for value in (q, k, v))
        reference = torch.nn.functional.scaled_dot_product_attention(
            q[:, None], k[:, None], v[:, None]
        ).squeeze(1).transpose(1, 2).reshape(2, channels, 4, 4)
        reference = (block.proj(reference) + frames).reshape(1, 2, channels, 4, 4).permute(0, 2, 1, 3, 4)

        def forbidden(*args, **kwargs):
            raise AssertionError("Video VAE must not use scaled_dot_product_attention")

        monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", forbidden)
        result = block(x)
    torch.testing.assert_close(result, reference, rtol=1e-5, atol=1e-5)


def test_patchify_exact():
    x = torch.arange(1 * 3 * 5 * 16 * 32).float().reshape(1, 3, 5, 16, 32)
    assert torch.equal(unpatchify(patchify(x, 2), 2), x)


def test_causal_conv_cache_equivalent_to_whole_video():
    conv = _CausalConv3D(3, 4, 3, padding=1).eval()
    x = torch.randn(1, 3, 5, 4, 4)
    full = conv(x)
    pieces = [conv(x[:, :, :1]), conv(x[:, :, 1:], cache_x=x[:, :, :1])]
    torch.testing.assert_close(torch.cat(pieces, dim=2), full, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("frames", [1, 5, 9])
def test_causal_video_shape_and_cache_isolation(frames):
    vae = _vae()
    video = torch.randn(1, 3, frames, 16, 16)
    with torch.no_grad():
        z = vae.encode(video)[0]
        assert z.shape == (1, 48, (frames - 1) // 4 + 1, 1, 1)
        a = vae.decode(z)
        b = vae.decode(z)
    assert a.shape == b.shape == video.shape
    assert torch.equal(a, b)
    assert all(entry is None for entry in vae._feat_map)


def test_decode_frames_are_the_whole_clip_in_order():
    """Streaming the decode out one latent frame at a time must be the same
    computation the one-call decode does, caches and all.
    """
    vae = _vae()
    z = torch.randn(1, 48, 3, 1, 1)
    with torch.no_grad():
        streamed = torch.cat(list(vae.decode_to_pixel_frames(z)), dim=2)
        whole = vae.decode_to_pixels(z)
        # The canonical interface is the old call with the normalisation around it.
        means = vae.decode(vae._denormalize(z))
    torch.testing.assert_close(streamed, whole, rtol=0, atol=0)
    torch.testing.assert_close(streamed, means, rtol=0, atol=0)
    assert all(entry is None for entry in vae._feat_map)


def test_pixels_to_latents_and_back_reaches_the_same_pixels():
    """The codec, not the adapter, owns the normalisation both ways."""
    vae = _vae()
    video = torch.randn(1, 3, 5, 16, 16)
    with torch.no_grad():
        canonical = vae.encode_pixels_to_latents(video)
        assert canonical.shape == (1, 48, 2, 1, 1)
        assert not torch.equal(canonical, vae.encode(video)[0])
        back = vae.decode_to_pixels(canonical)
    assert back.shape == video.shape


def test_a_precision_nobody_measured_this_codec_at_is_refused():
    """FP16 is where this family overflows; FP32 only costs, so it is allowed."""
    with torch.no_grad():
        with pytest.raises(RuntimeError, match="runs in torch.bfloat16"):
            _vae().half().decode(torch.randn(1, 48, 1, 1, 1))


def test_comfy_fp16_checkpoint_loads_at_the_compute_dtype(tmp_path):
    path, source = _write_vae(tmp_path)
    vae = load_wan_video_vae(str(path), "cpu", config=_comfy_vae_config(VAE_GEOMETRY))
    assert vae.dtype is vae.compute_dtype is torch.bfloat16
    assert not any(p.is_meta for p in vae.parameters())
    assert torch.equal(vae.encoder.conv_in.weight,
                       source.encoder.conv_in.weight.half().bfloat16())
    with torch.no_grad():
        result = vae.decode(torch.randn(1, 48, 2, 1, 1))
    assert result.shape == (1, 3, 5, 16, 16)


@pytest.mark.parametrize("damage,raises", [("missing", "Missing key"), ("shape", "size mismatch")])
def test_comfy_checkpoint_rejects_invalid_weights(tmp_path, damage, raises):
    path, _ = _write_vae(tmp_path, missing=damage == "missing", wrong_shape=damage == "shape")
    with pytest.raises(RuntimeError, match=raises):
        load_wan_video_vae(str(path), "cpu", config=_comfy_vae_config(VAE_GEOMETRY))


def test_the_release_is_read_off_the_file_not_from_the_model(tmp_path):
    """Which codec a file is comes from its header, since a request cannot say it."""
    path, _ = _write_vae(tmp_path)
    assert detect_latent_width(str(path)) == 48


def test_the_compression_properties_are_the_codecs_own():
    vae = _vae()
    assert vae.z_dim == 48 and vae.pixel_channels == 3
    # VAE_GEOMETRY keeps the released stage count and patch, so so does its compression.
    assert vae.spatial_compression == 16 and vae.temporal_compression == 4


def test_official_comfy_vae_topology_meta():
    with torch.device("meta"):
        vae = AutoencoderKLWanVideo(_comfy_vae_config())
    assert vae.config.z_dim == 48
    assert vae.config.base_dim == 160
    assert all(p.is_meta for p in vae.parameters())


@pytest.mark.parametrize("in_c,out_c,factor_t",
                         [(8, 4, 1), (8, 4, 2), (4, 4, 2), (8, 8, 2), (4, 8, 2)])
@pytest.mark.parametrize("t,first_chunk",
                         [(1, True), (1, False), (2, False), (3, True), (3, False)])
def test_dupup_add_to_is_the_duplicate_added(in_c, out_c, factor_t, t, first_chunk):
    """The fused shortcut adds what building the duplicate and adding it would have.

    ``(4, 8, 2)`` is the ratio that cannot decompose into a strided slice, so it covers
    the fallback alongside the walk.
    """
    up = _DupUp3D(in_c, out_c, factor_t, factor_s=2)
    x = torch.randn(1, in_c, t, 3, 5)
    with torch.no_grad():
        expected = up(x, first_chunk=first_chunk)
        got = up.add_to(x.clone(), torch.zeros_like(expected), first_chunk=first_chunk)
    assert torch.equal(got, expected)


def test_dupup_add_to_refuses_a_target_of_the_wrong_shape():
    up = _DupUp3D(8, 4, 2, factor_s=2)
    x = torch.randn(1, 8, 2, 3, 5)
    with pytest.raises(ValueError, match="must have shape"):
        up.add_to(x, torch.zeros(1, 4, 2, 6, 10))


@pytest.mark.parametrize("temporal", [False, True])
def test_an_up_block_leaves_the_tensor_it_was_handed_alone(temporal):
    """The shortcut an up block adds is its input, not a copy of it, which holds only
    because no stage in this codec writes into what it was given.
    """
    block = _ResidualUpBlock(4, 4, 1, 0.0, temporal=temporal, up=True)
    # A temporal block only sees one latent frame per call, with the feature cache that
    # carries the causal state across calls, so that is the shape fed to it here.
    x = torch.randn(1, 4, 1 if temporal else 2, 4, 4)
    before = x.clone()
    with torch.no_grad():
        block(x, [None] * 16, [0], first_chunk=temporal)
    assert torch.equal(x, before)


def test_a_down_block_leaves_the_tensor_it_was_handed_alone():
    block = _ResidualDownBlock(4, 4, 0.0, 1, temporal=False, down=True)
    x = torch.randn(1, 4, 2, 4, 4)
    before = x.clone()
    with torch.no_grad():
        block(x, None, None)
    assert torch.equal(x, before)


# ----------------------------------------------------------------- Schedule

def test_wan_fixed_flow_shift():
    assert FLOW_SHIFT == 5.0
    sigmas = get_sigmas(4)
    assert sigmas[0] == 1.0 and sigmas[-1] == 0.0
    assert all(a > b for a, b in zip(sigmas, sigmas[1:]))
    assert wan_adapter().resolve_video(VideoSpec(prompt="x", frames=10)).frame_count == 13
    with pytest.raises(ValueError, match="supports only"):
        wan_adapter().resolve_video(VideoSpec(prompt="x", sampler="unknown"))


@pytest.mark.parametrize("shape", [(1, 8, 4, 6), (1, 48, 3, 4, 6)])
@pytest.mark.parametrize("sampler", ["euler", "unipc"])
def test_shared_sampler_constant_velocity_and_step_callback(shape, sampler):
    class Model:
        def begin_sampling(self, schedule):
            pass

        def denoise_step(self, x, t, cond, guidance_scale, i):
            return torch.ones_like(x)

    x = torch.zeros(shape)
    sigmas = get_sigmas(5)
    schedule = [Step(torch.tensor(s), torch.tensor(s - next_s))
                for s, next_s in zip(sigmas, sigmas[1:])]
    events = []
    out = create_sampler(sampler, Model()).sample(
        x, schedule, None, 1.0, 0,
        step_callback=lambda y, i, total: (events.append((i, total)) or y),
    )
    torch.testing.assert_close(out, -torch.ones_like(x), atol=1e-4, rtol=1e-4)
    assert events == [(i, 5) for i in range(1, 6)]


def test_step_callback_cancellation():
    class Model:
        def begin_sampling(self, schedule):
            pass
        def denoise_step(self, x, t, cond, guidance_scale, i):
            return x

    schedule = [Step(torch.tensor(1.0), torch.tensor(0.5)),
                Step(torch.tensor(0.5), torch.tensor(0.5))]
    def stop(x, i, n):
        raise RuntimeError("cancel")

    with pytest.raises(RuntimeError, match="cancel"):
        create_sampler("euler", Model()).sample(
            torch.zeros(1, 3, 4, 4), schedule, None, 1.0, 0,
            step_callback=stop,
        )


# ----------------------------------------------------------------- Adapter lifecycle, progress, cancellation

FRAMES_PER_CHUNK = (1, 4, 4)  # a causal codec's first latent frame decodes alone


def _stand_in_paths(root):
    """The paths `fake_video` stands its three loaders in for."""
    return ModelConfig(
        dit_paths={"": write_wan_dit(root / "dit.safetensors")},
        vae_path=str(write_wan_vae(root / "vae.safetensors")),
        text_encoder_path=str(root), device="cpu", offload_device="cpu")


@pytest.fixture
def fake_video(tmp_path, monkeypatch):
    root = tmp_path / "model"
    root.mkdir()
    calls = []

    class VAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
            self.cache_clears = 0

        @property
        def dtype(self):
            return self.weight.dtype

        def clear_cache(self):
            self.cache_clears += 1

        z_dim = 48
        pixel_channels = 3
        spatial_compression = 16
        temporal_compression = 4

        def encode_pixels_to_latents(self, pixels):
            calls.append("vae_encode")
            return torch.zeros(pixels.shape[0], 48, 1,
                               pixels.shape[-2] // 16, pixels.shape[-1] // 16)

        def decode_to_pixel_frames(self, z):
            calls.append("vae_decode")
            try:
                for i in range(z.shape[2]):
                    yield torch.zeros(z.shape[0], 3, FRAMES_PER_CHUNK[i],
                                      z.shape[3] * 16, z.shape[4] * 16)
            finally:
                self.clear_cache()

    class DiT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.float16))
            self.params = types.SimpleNamespace(patch_size=(1, 2, 2))

        @property
        def dtype(self):
            return self.weight.dtype

        def forward(self, x, timestep, encoder_hidden_states):
            calls.append("dit")
            return torch.zeros_like(x)

    class Text(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))

        def forward(self, ids, mask):
            return torch.zeros(1, 16, 32, dtype=self.weight.dtype)

    class Tokenizer:
        def encode(self, prompt):
            return torch.ones(1, 16, dtype=torch.long), torch.ones(1, 16, dtype=torch.bool)

    monkeypatch.setattr("thenoise.models.wan_video.load_wan_video_vae",
                        lambda _path, _device: calls.append("load_vae") or VAE())
    monkeypatch.setattr("thenoise.models.wan_video.load_wan_native_transformer",
                        lambda _path, _device: calls.append("load_dit") or DiT())
    monkeypatch.setattr("thenoise.models.wan_video.load_wan_text_encoder",
                        lambda _path, _device: calls.append("load_text") or (Text(), Tokenizer()))

    def stub_encode_video(frames, _spec, _token, dest, _progress):
        for _ in frames:  # a real muxer pulls frames lazily; so must this stub
            pass
        dest.write_bytes(b"video")

    monkeypatch.setattr("thenoise.video.controller.encode_video", stub_encode_video)

    # The adapter reads the DiT and codec headers for its variant and its pairing even
    # though every loader here is a stub, so both files have to be real.
    video = Wan22Model(config=_stand_in_paths(root))
    return video, VideoPipelineController(video), calls


def _spec(video, *, image=None, **kwargs):
    return video.resolve_video(VideoSpec(
        prompt="fox", width=480, height=320, frames=9, steps=2, image=image, **kwargs))


def test_construction_loads_every_component_once(fake_video):
    """There is no second entry point: an adapter is loaded when it is built."""
    video, _, calls = fake_video
    assert video.dit is not None and video.text_encoder is not None
    assert [calls.count(name) for name in
            ("load_dit", "load_text", "load_vae")] == [1, 1, 1]


def test_native_t2v_i2v_t2v_reuses_models(fake_video, tmp_path):
    video, controller, calls = fake_video
    image = io.BytesIO()
    Image.new("RGB", (64, 64), "red").save(image, format="PNG")
    out = tmp_path / "out.mp4"
    for spec in (_spec(video), _spec(video, image=image.getvalue()), _spec(video)):
        controller.render(spec, Cancellation(), lambda *_: None, out)
    assert calls.count("load_dit") == 1  # a render never loads for itself
    # Only the I2V request encodes a first frame.
    assert calls.count("vae_encode") == 1
    assert video.vae.dtype is torch.float32
    assert video.dit.dtype is torch.float16
    assert video.text_encoder.weight.dtype is torch.bfloat16
    assert out.read_bytes() == b"video"


def test_unload_releases_weights_and_allows_reloading(fake_video, tmp_path):
    video, controller, calls = fake_video
    controller.render(_spec(video), Cancellation(), lambda *_: None, tmp_path / "clip.mp4")
    old_manager = video.memory

    video.unload()
    assert (video.vae, video.text_encoder, video.tokenizer, video.dit) == (None,) * 4
    assert video.memory is not old_manager
    assert video.memory.weight_bytes("dit") == 0

    # A reload is a rebuild, not a second call into the adapter that was unloaded.
    fresh = Wan22Model(config=_stand_in_paths(tmp_path / "model"))
    controller2 = VideoPipelineController(fresh)
    controller2.render(_spec(fresh), Cancellation(), lambda *_: None,
                       tmp_path / "again.mp4")
    assert (tmp_path / "again.mp4").read_bytes() == b"video"
    assert calls.count("load_dit") == calls.count("load_vae") == 2


def test_cancel_sampling_recovers_for_next_job(fake_video, tmp_path):
    video, controller, calls = fake_video
    token = Cancellation()

    def cancel_at_one(stage, step, _total):
        if stage == "sampling" and step == 1:
            token.cancel()

    with pytest.raises(VideoCancelled):
        controller.render(_spec(video), token, cancel_at_one, tmp_path / "clip.mp4")
    controller.render(_spec(video), Cancellation(), lambda *_: None, tmp_path / "clip.mp4")
    assert calls.count("load_dit") == 1  # a cancelled job leaves the components usable


def test_decode_progress_uses_shared_tqdm_style(fake_video, tmp_path, monkeypatch):
    seen = []

    class Bar:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def update(self, value):
            seen.append(("update", value))

    monkeypatch.setattr("thenoise.video.controller.tqdm",
                        lambda *, total, desc: seen.append((desc, total)) or Bar())
    video, controller, _calls = fake_video
    controller.render(_spec(video), Cancellation(), lambda *_: None, tmp_path / "clip.mp4")
    assert seen == [("decoding", 9)] + [("update", 1)] * 9


def test_decode_progress_and_recovery(fake_video, tmp_path):
    video, controller, _calls = fake_video
    events = []
    controller.render(_spec(video), Cancellation(), lambda *e: events.append(e),
                      tmp_path / "clip.mp4")
    assert ("sampling", 1, 2) in events
    assert [e for e in events if e[0] == "decoding"] == [
        ("decoding", i, 9) for i in range(10)
    ]
    assert ("encoding", 0, 9) in events
    assert video.vae.cache_clears >= 1


def test_an_unresolved_size_and_frame_count_reach_the_kernels(fake_video):
    """The controller resolves nothing itself; the adapter's grid decides."""
    video, controller, _calls = fake_video
    spec = controller.resolve(VideoSpec(prompt="fox", width=500, height=330, duration=1.2))
    # 1.2 s at the variant's 24 fps, then onto the codec's 4n+1 grid.
    assert (spec.width, spec.height, spec.frame_count) == (480, 320, 29)
    assert spec.explicit_frames is False


# ----------------------------------------------------------------- LoRAs

def _lora_config(tmp_path, *, experts: bool = False):
    """A real (tiny) DiT, because the adapter reads the variant out of its header.

    Kept out of ``lora_dir``, which is scanned recursively and would otherwise offer
    the DiT itself as a LoRA.
    """
    (tmp_path / "loras").mkdir(exist_ok=True)
    channels = 16 if experts else 48
    dit = write_wan_dit(tmp_path / "dit.safetensors", channels=channels)
    return ModelConfig(
        dit_paths={"": dit, "low": dit} if experts else {"": dit},
        vae_path=str(write_wan_vae(tmp_path / "vae.safetensors", channels=channels)),
        text_encoder_path="te",
        device="cpu", offload_device="cpu", lora_dir=str(tmp_path / "loras"))


def _expert_pair(tmp_path):
    """Two LoRA-able modules, and two LoRAs whose deltas are 1x and 2x the weight."""
    def target():
        dit = torch.nn.Module()
        dit.blocks = torch.nn.ModuleList([torch.nn.Module()])
        dit.blocks[0].attn1 = torch.nn.Module()
        dit.blocks[0].attn1.to_q = QuantizedLinear(4, 4, bias=False)
        return dit, dit.blocks[0].attn1.to_q.weight.detach().clone()

    for name, magnitude in (("style", 1.0), ("pose", 2.0)):
        save_file({
            "blocks.0.self_attn.q.lora_A.weight": torch.ones(1, 4),
            "blocks.0.self_attn.q.lora_B.weight": torch.full((4, 1), magnitude),
        }, str(tmp_path / "loras" / f"{name}.safetensors"))
    return target(), target()


@pytest.fixture
def stub_wan_component_loaders(monkeypatch):
    """LoRA bookkeeping never reaches the codec or the encoder, but the adapter loads
    both as it is built."""
    vae, text_encoder = stub_wan_components()
    monkeypatch.setattr("thenoise.models.wan_video.load_wan_video_vae", vae)
    monkeypatch.setattr("thenoise.models.wan_video.load_wan_text_encoder", text_encoder)


@pytest.mark.usefixtures("stub_wan_component_loaders")
def test_video_loras_use_the_shared_switch():
    """Listing and switching come from ``LoRASupport``; all the adapter adds is the
    trainer's key names and a second LoRA-able component for its low-noise expert.
    """
    assert Wan22Model.list_loras is DiffusionModel.list_loras
    assert Wan22Model._lora_key_map is not DiffusionModel._lora_key_map
    assert sorted(wan_adapter().lora_components()) == ["dit"]
    assert sorted(wan_adapter("t2v_14b").lora_components()) == ["dit", "dit_low"]


@pytest.mark.usefixtures("stub_wan_component_loaders")
def test_each_expert_takes_its_own_loras(tmp_path):
    """The role is the component: what one expert is asked for never reaches the other,
    and switching one leaves the other's LoRA in place.
    """
    video = Wan22Model(config=_lora_config(tmp_path, experts=True))
    (dit, dit_original), (low, low_original) = _expert_pair(tmp_path)

    video.switch_loras(["style"], dit, "dit")
    video.switch_loras(["pose:0.5"], low, "dit_low")
    torch.testing.assert_close(dit.blocks[0].attn1.to_q.weight, dit_original + 1.0,
                               atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(low.blocks[0].attn1.to_q.weight, low_original + 1.0,
                               atol=1e-6, rtol=1e-6)

    video.switch_loras(None, dit, "dit")
    torch.testing.assert_close(dit.blocks[0].attn1.to_q.weight, dit_original,
                               atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(low.blocks[0].attn1.to_q.weight, low_original + 1.0,
                               atol=1e-6, rtol=1e-6)


@pytest.mark.usefixtures("stub_wan_component_loaders")
def test_a_role_the_model_does_not_offer_is_refused_at_resolve():
    """/videos answers a bad role at submit, and the two-expert model offers both.

    A bare list is the single component's, which is every request written before roles.
    """
    from thenoise.video.spec import VideoSpec

    single = wan_adapter()
    with pytest.raises(ValueError, match="does not offer"):
        single.resolve_video(VideoSpec(prompt="fox", lora_specs={"dit_low": ["style"]}))
    assert single.resolve_video(
        VideoSpec(prompt="fox", lora_specs=["style"])).lora_specs == {"dit": ["style"]}
    both = wan_adapter("t2v_14b")
    assert both.resolve_video(VideoSpec(
        prompt="fox",
        lora_specs={"dit": ["a"], "dit_low": ["b"]})).lora_specs == {
        "dit": ["a"], "dit_low": ["b"]}
    # A bare list is the default role's on a two-expert model too: the high-noise
    # expert, not both of them.
    assert both.resolve_video(VideoSpec(prompt="fox", lora_specs=["a"])).lora_specs == {
        "dit": ["a"]}


@pytest.mark.usefixtures("stub_wan_component_loaders")
def test_an_out_of_memory_switch_refuses_only_that_component(tmp_path, monkeypatch):
    """Half-merged weights cannot be repaired, so that component is finished on this
    instance. The other one was never touched, and keeps taking LoRAs.
    """
    video = Wan22Model(config=_lora_config(tmp_path, experts=True))
    (dit, _), (low, _) = _expert_pair(tmp_path)

    def maybe_oom(module, _lora_sds, _multipliers, **_kwargs):
        if module is low:
            raise torch.cuda.OutOfMemoryError

    monkeypatch.setattr("thenoise.models.lora_support.apply_lora_to_model", maybe_oom)
    with pytest.raises(ModelReloadNeeded, match="'dit_low'"):
        video.switch_loras(["pose"], low, "dit_low")
    with pytest.raises(ModelReloadNeeded, match="'dit_low'"):
        video.switch_loras(None, low, "dit_low")
    video.switch_loras(["style"], dit, "dit")


@pytest.mark.parametrize("raw,canonical", [
    ("blocks.0.self_attn.q", "blocks.0.attn1.to_q"),
    ("diffusion_model.blocks.0.cross_attn.v", "blocks.0.attn2.to_v"),
    ("lora_unet_blocks_0_self_attn_q", "blocks.0.attn1.to_q"),
    ("blocks.0.ffn.0", "blocks.0.ffn_in"),
    ("blocks.0.ffn.2", "blocks.0.ffn_out"),
])
def test_wan_lora_key_map(raw, canonical):
    video = wan_adapter()
    assert video._lora_key_map(raw + ".lora_A.weight") == canonical + ".lora_A.weight"
    assert video._lora_key_map(raw + ".lora_B.weight") == canonical + ".lora_B.weight"


@pytest.mark.usefixtures("stub_wan_component_loaders")
def test_wan_applies_switches_and_undoes_native_lora(tmp_path):
    video = Wan22Model(config=_lora_config(tmp_path))
    dit = torch.nn.Module()
    dit.blocks = torch.nn.ModuleList([torch.nn.Module()])
    dit.blocks[0].attn1 = torch.nn.Module()
    dit.blocks[0].attn1.to_q = QuantizedLinear(4, 4, bias=False)
    original = dit.blocks[0].attn1.to_q.weight.detach().clone()
    save_file({
        "blocks.0.self_attn.q.lora_A.weight": torch.ones(1, 4),
        "blocks.0.self_attn.q.lora_B.weight": torch.ones(4, 1),
    }, str(tmp_path / "loras" / "style.safetensors"))

    assert video.list_loras() == ["style"]
    video.switch_loras(["style:0.5"], dit)  # a bare role: the single component's
    torch.testing.assert_close(dit.blocks[0].attn1.to_q.weight, original + 0.5,
                               atol=1e-6, rtol=1e-6)
    video.switch_loras(["style:0.5"], dit)  # idempotent
    video.switch_loras(None, dit)
    torch.testing.assert_close(dit.blocks[0].attn1.to_q.weight, original,
                               atol=1e-6, rtol=1e-6)


@pytest.mark.usefixtures("stub_wan_component_loaders")
def test_video_lora_names_cannot_escape_model_lora_directory(tmp_path):
    video = Wan22Model(config=_lora_config(tmp_path))
    with pytest.raises(ValueError, match="escapes base directory"):
        video.switch_loras(["../escape"], torch.nn.Module())


# ----------------------------------------------------------------- Two experts (A14B)

A14B = VARIANTS["t2v_14b"]


def test_one_adapter_claims_both_transformers(tmp_path):
    """Detection stops at "a Wan 2.2 DiT": the variant is a shape read afterwards."""
    for channels, name in ((48, "ti2v_5b"), (16, "t2v_14b")):
        path = write_wan_dit(tmp_path / f"dit{channels}.safetensors", channels=channels)
        assert resolve(path) is Wan22Model
        assert variant_for(detect_params(path)) is VARIANTS[name]


def test_the_model_name_is_one_machine_string():
    assert Wan22Model.name == "wan22"
    assert wan_adapter("ti2v_5b").CAPABILITIES["i2v"] is True
    assert wan_adapter("t2v_14b").CAPABILITIES["i2v"] is False


def test_a_two_expert_transformer_demands_its_second_file(tmp_path):
    high = write_wan_dit(tmp_path / "high.safetensors", channels=16)
    with pytest.raises(ValueError, match="--dit-low"):
        Wan22Model.validate_paths({"": high})
    with pytest.raises(FileNotFoundError, match="low-noise"):
        Wan22Model.validate_paths({"": high, "low": str(tmp_path / "gone.safetensors")})
    with pytest.raises(ValueError, match="different checkpoints"):
        Wan22Model.validate_paths({"": high, "low": high})
    Wan22Model.validate_paths(
        {"": high, "low": write_wan_dit(tmp_path / "low.safetensors", channels=16)})


def test_a_codec_from_the_other_release_is_refused_from_the_headers(tmp_path):
    """The clip's latent is the only thing that crosses between the two checkpoints, and
    the normalization table is keyed by its width, so a mismatched `--vae` is answered
    before either module is built rather than at the first denoising step.
    """
    config = ModelConfig(
        dit_paths={"": write_wan_dit(tmp_path / "dit.safetensors", channels=48)},
        vae_path=str(write_wan_vae(tmp_path / "vae.safetensors", channels=16)),
        text_encoder_path="", device="cpu", offload_device="cpu")
    with pytest.raises(ValueError, match="different Wan releases"):
        Wan22Model(config=config)


def test_a_single_expert_transformer_refuses_a_second_file(tmp_path):
    with pytest.raises(ValueError, match="one expert"):
        Wan22Model.validate_paths({
            "": write_wan_dit(tmp_path / "a.safetensors"),
            "low": write_wan_dit(tmp_path / "b.safetensors"),
        })


def test_runtime_maps_the_cli_flags_into_roles(tmp_path, monkeypatch):
    """The runtime hands the adapter the paths by role and keeps no Wan knowledge."""
    import thenoise.models as catalog

    class StubVideo:
        name = "stub"
        VIDEO = True
        validated = []

        def __init__(self, *, config):
            self.config = config

        @classmethod
        def validate_paths(cls, dit_paths):
            cls.validated.append(dict(dit_paths))

    monkeypatch.setattr(catalog, "resolve", lambda path: StubVideo)
    rt = Runtime(Settings(device="cpu", offload_device="cpu"))
    high, low = str(tmp_path / "high.safetensors"), str(tmp_path / "low.safetensors")
    Path(high).write_bytes(b"high")
    Path(low).write_bytes(b"low")
    rt.load(ModelPaths(high, "vae", "te", dit_low_path=low))
    assert StubVideo.validated == [{"": high, "low": low}]
    assert rt.model.config.dit_paths == {"": high, "low": low}
    assert rt.model.config.dit_path == high


def test_14b_topology_meta_and_vae21_key_mapping():
    params = Wan22Params(in_channels=16, out_channels=16, dim=5120, num_layers=40,
                         ffn_dim=13824, freq_dim=256, text_dim=4096, patch_size=(1, 2, 2))
    with torch.device("meta"):
        dit = WanNativeDiT(params)
    assert params.heads == 40
    assert dit.patch_embedding.weight.shape == (5120, 16, 1, 2, 2)
    assert dit.proj_out.weight.shape == (64, 5120)
    assert len(dit.blocks) == 40
    assert dit.blocks[0].ffn_in.weight.shape == (13824, 5120)

    vae_cfg = _comfy_vae_config(dict(base_dim=8, decoder_base_dim=8, z_dim=16,
                                    in_channels=3, out_channels=3, patch_size=1,
                                    is_residual=False,
                                    temperal_downsample=[True, True, False],
                                    latents_mean=[0.] * 16, latents_std=[1.] * 16))
    with torch.device("meta"):
        vae = AutoencoderKLWanVideo(vae_cfg)
    assert vae.encoder.conv_in.weight.shape[1] == 3
    assert vae.decoder.up_blocks[0].upsampler.resample[1].weight.shape[:2] == (16, 32)
    assert wan_vae21_key("encoder.downsamples.2.time_conv.weight") == (
        "encoder.down_blocks.0.downsampler.time_conv.weight")
    assert wan_vae21_key("decoder.upsamples.3.resample.1.weight") == (
        "decoder.up_blocks.0.upsampler.resample.1.weight")
    assert wan_vae21_key("encoder.downsamples.0.residual.0.gamma") == (
        "encoder.down_blocks.0.resnets.0.norm1.gamma")


def test_the_two_experts_split_the_schedule_at_the_sigma_boundary():
    model = wan_adapter("t2v_14b")
    high, low = object(), object()
    model.dit, model.low_dit = high, low
    steps = model.schedule(VideoSamplingParams(height=64, width=64, steps=4, seed=1,
                                               guidance_scale=4.0, sampler="euler"))
    spans = model.segments(None, steps, guidance_scale=4.0)

    assert [s.stage for s in spans] == ["dit", "dit_low"]
    assert [s.module for s in spans] == [high, low]
    assert [len(s.steps) for s in spans] == [3, 1]
    assert sum(len(s.steps) for s in spans) == len(steps)
    # Upstream ships guidance 4.0 on the high half and 3.0 on the low.
    assert spans[1].guidance_scale == pytest.approx(4.0 * A14B.experts.low_cfg_ratio)
    sigmas = get_sigmas(4, shift=A14B.experts.flow_shift)
    assert sigmas[2] > A14B.experts.boundary > sigmas[3]


def test_a_single_expert_variant_answers_one_segment():
    model = wan_adapter()
    model.dit = object()
    spans = model.segments(None, [Step(t=torch.zeros(()), delta=torch.zeros(()))], 5.0)
    assert [s.stage for s in spans] == ["dit"]


def test_the_16ch_video_codec_is_the_qwen_image_codec_in_3d():
    """One architecture, two folds: the geometry is `qwen_image`'s rather than restated
    here, and the latent statistics are the same numbers -- which is why the 14B experts
    and Qwen-Image agree on what a canonical latent is.
    """
    video_cfg = _comfy_vae_config(_wan21_vae_config())
    assert (video_cfg.base_dim, video_cfg.z_dim, list(video_cfg.dim_mult),
            video_cfg.num_res_blocks) == (
        _WAN21_FAMILY_ARCH["base_dim"], _WAN21_FAMILY_ARCH["z_dim"],
        list(_WAN21_FAMILY_ARCH["dim_mult"]), _WAN21_FAMILY_ARCH["num_res_blocks"])
    assert video_cfg.latents_mean == QWEN_IMAGE_LATENTS_MEAN
    assert video_cfg.latents_std == QWEN_IMAGE_LATENTS_STD

    with torch.device("meta"):
        video = AutoencoderKLWanVideo(video_cfg)
    image = AutoencoderKLQwenImage(**_WAN21_FAMILY_ARCH)
    assert video.spatial_compression == image.spatial_compression == 8
    assert video.pixel_channels == image.pixel_channels == 3
    # The family signature: Wan 2.1 halves channels on the way up, 2.2 keeps them.
    upsample = video.decoder.up_blocks[0].upsampler.resample[1]
    assert upsample.weight.shape[0] == upsample.weight.shape[1] // 2


def test_the_16ch_latent_normalisation_is_not_identity():
    assert len(QWEN_IMAGE_LATENTS_MEAN) == len(QWEN_IMAGE_LATENTS_STD) == 16
    with torch.device("meta"):
        vae = AutoencoderKLWanVideo(_comfy_vae_config(_wan21_vae_config() | dict(
            base_dim=4, decoder_base_dim=4, num_res_blocks=1)))
    z = torch.randn(1, 16, 2, 3, 4)
    normalized = vae._normalize(z)
    assert not torch.allclose(z, normalized)
    torch.testing.assert_close(vae._denormalize(normalized), z)


def test_a_video_codec_of_another_latent_width_is_refused_by_name():
    assert sorted(_RELEASES) == [16, 48]
    assert _vae_release(16)[1] is not _vae_release(48)[1]
    with pytest.raises(ValueError, match=r"32-channel video VAE is no Wan release.*\[16, 48\]"):
        _vae_release(32)

def test_each_video_codec_declares_its_own_compute_dtype():
    """The 48ch 2.2 codec runs at the 2.2 family's BF16 (see
    ``vae.wan22.COMPUTE_DTYPE``), the 16ch 2.1 codec at the FP16 it ships in; the
    loader moves the module to whatever its config declares.
    """
    assert _comfy_vae_config().compute_dtype is torch.bfloat16
    assert _comfy_vae_config(_wan21_vae_config()).compute_dtype is torch.float16


def test_wan21_fp16_vae_decodes_float32_latents_and_clears_cache():
    cfg = _comfy_vae_config(_wan21_vae_config() | dict(
        base_dim=4, decoder_base_dim=4, num_res_blocks=1,
    ))
    with torch.inference_mode():
        vae = AutoencoderKLWanVideo(cfg).half().eval()
        latents = torch.randn(1, 16, 2, 2, 2, dtype=torch.float32)
        first = vae.decode(latents)
        second = vae.decode(latents)
        assert first.dtype == torch.float16
        assert first.shape == second.shape == (1, 3, 5, 16, 16)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        assert all(item is None for item in vae._feat_map)


def test_the_two_expert_variant_has_no_i2v():
    assert A14B.i2v is False and VARIANTS["ti2v_5b"].i2v is True
    with pytest.raises(ValueError, match="I2V"):
        wan_adapter("t2v_14b").resolve_video(VideoSpec(prompt="x", image=b"image"))
