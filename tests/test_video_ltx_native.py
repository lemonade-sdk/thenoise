"""Small CPU contract tests for native LTX checkpoint loading and sampling."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from conftest import tiny_ltx_video_vae
from thenoise.dit.ltx_video import checkpoint


def _save(path, tensors, metadata=None):
    save_file({name: tensor.clone() for name, tensor in tensors.items()}, str(path),
              metadata=metadata)
    return path


#: Codecs as a checkpoint that ships them inside the transformer file names them.
MONOLITH_CODEC_KEYS = (
    "vae.decoder.up_blocks.0.res_blocks.0.conv1.conv.weight",
    "vae.decoder.up_blocks.1.conv.conv.weight",
    "vae.decoder.up_blocks.8.res_blocks.1.conv2.conv.bias",
    "vae.encoder.down_blocks.8.res_blocks.0.conv1.conv.weight",
    "vae.encoder.down_blocks.1.conv.conv.weight",
    "vae.per_channel_statistics.mean-of-means",
    "audio_vae.per_channel_statistics.std-of-means",
    "vocoder.vocoder.conv_pre.weight",
    "vocoder.vocoder.resblocks.0.acts1.0.act.alpha",
    "vocoder.vocoder.ups.0.weight",
    "vocoder.bwe_generator.act_post.downsample.lowpass.filter",
)


def test_a_codec_packed_with_the_transformer_fills_what_a_codec_alone_fills():
    """The released all-in-one spellings of the codec tree reach the module.

    A file holding one codec names the blocks the way the module does; a codec packed
    with the transformer uses the original naming, where the res-block groups and the
    sampling convs share one numbered list. Missing the rename is a refusal of weights
    the checkpoint does have.
    """
    from thenoise.dit.ltx_video.checkpoint import (make_audio_vae, make_vocoder,
                                                   make_video_vae)
    from thenoise.dit.ltx_video.keys import component_for

    with torch.device("meta"):
        modules = {"vae": make_video_vae(), "audio_vae": make_audio_vae(),
                   "vocoder": make_vocoder()}
    for raw in MONOLITH_CODEC_KEYS:
        name = raw.split(".", 1)[0]
        packed = component_for(raw, transformer=True)
        assert packed == component_for(raw), raw
        assert packed[1] in modules[name].state_dict(), f"{raw} -> {packed[1]!r}"


def test_a_transformer_is_claimed_by_exactly_one_version(tmp_path):
    """Two declarations do what a hand-written detector did, and rule out the other way
    too: the version a file is not cannot claim it and then build it wrong."""
    from thenoise.models.ltx_video import LTX23Video, LTX25Video
    from thenoise.utils.safetensors import read_header

    for version, winner, loser in (("2.3", LTX23Video, LTX25Video),
                                   ("2.5", LTX25Video, LTX23Video)):
        data = {"model.diffusion_model.patchify_proj.weight": torch.ones(2, 2),
                "model.diffusion_model.audio_patchify_proj.weight": torch.ones(2, 2)}
        if version == "2.5":
            data["model.diffusion_model.keyframes_abs_pos_embedding"] = torch.ones(1, 2)
        header = read_header(str(_save(tmp_path / f"ltx-{version}.safetensors", data)))
        assert winner.detect(header), version
        assert not loser.detect(header), version


def test_a_file_stating_another_version_is_refused_by_the_model_it_was_given_to(tmp_path):
    """Metadata cannot move a tensor, so it is not the detector. It is still what a file
    says about itself, and a request that named one model and handed over the other's
    file is a mistake worth naming -- from the header the runtime already parsed."""
    from thenoise.models.ltx_video import LTX23Video, LTX25Video
    from thenoise.utils.safetensors import read_header

    tensors = {"patchify_proj.weight": torch.ones(2, 2),
               "audio_patchify_proj.weight": torch.ones(2, 2)}
    stated = read_header(str(_save(
        tmp_path / "stated.safetensors", tensors, metadata={"model_version": "2.5.0"})))
    silent = read_header(str(_save(tmp_path / "silent.safetensors", tensors)))

    LTX25Video.validate_paths({"": stated})
    # A file that states nothing about itself is not a lie either.
    LTX23Video.validate_paths({"": silent})
    with pytest.raises(ValueError, match=r"states model_version 2\.5\.0"):
        LTX23Video.validate_paths({"": stated})


def _geometry_keys(**extra):
    """The names a transformer's geometry is read from, at sizes too small to run.

    The widths that have to divide a head count are kept -- 128 video, 64 audio -- and
    every other number differs, so a swapped pair cannot pass.
    """
    return {
        "model.diffusion_model.patchify_proj.weight": torch.zeros(128, 4),
        "model.diffusion_model.proj_out.weight": torch.zeros(6, 128),
        "model.diffusion_model.audio_patchify_proj.weight": torch.zeros(64, 8),
        "model.diffusion_model.audio_proj_out.weight": torch.zeros(10, 64),
        "model.diffusion_model.adaln_single.linear.weight": torch.zeros(9 * 128, 128),
        "model.diffusion_model.audio_adaln_single.linear.weight": torch.zeros(6 * 64, 64),
        "model.diffusion_model.transformer_blocks.0.attn2.to_k.weight": torch.zeros(128, 96),
        "model.diffusion_model.transformer_blocks.0.audio_attn2.to_k.weight": torch.zeros(64, 48),
        **extra,
    }


def _geometry_file(tmp_path, name="av.safetensors", **extra):
    return _save(tmp_path / name, _geometry_keys(**extra))


def test_a_transformer_measures_itself_from_the_header(tmp_path):
    """A released configuration is not a fact to keep in a dict: depth, widths and the
    optional halves are read, which is what lets a re-export of another size, or one
    shipped without its zero biases, load instead of failing a shape check."""
    from thenoise.dit.ltx_video.checkpoint import detect_params

    two_blocks = {
        "model.diffusion_model.transformer_blocks.1.attn2.to_k.weight": torch.zeros(128, 96),
    }
    released = detect_params(_geometry_file(
        tmp_path, "released.safetensors",
        **two_blocks,
        **{"model.diffusion_model.transformer_blocks.0.ff.net.0.proj.bias": torch.zeros(512)}))
    assert (released.dim, released.heads, released.num_layers) == (128, 1, 2)
    assert (released.in_channels, released.out_channels) == (4, 6)
    assert (released.audio_dim, released.audio_heads) == (64, 1)
    assert (released.audio_in_channels, released.audio_out_channels) == (8, 10)
    assert (released.cross_attention_dim, released.audio_cross_attention_dim) == (96, 48)
    assert released.cross_attn_mod and not released.audio_cross_attn_mod
    assert released.ff_bias and not released.audio_ff_bias
    assert not released.keyframes

    stripped = detect_params(_geometry_file(
        tmp_path, "stripped.safetensors",
        **{"model.diffusion_model.keyframes_abs_pos_embedding": torch.zeros(1, 128),
           "model.diffusion_model.transformer_blocks.0.attn1.to_gate_logits.weight":
               torch.zeros(1, 128)}))
    assert stripped.keyframes and stripped.gated_attn
    assert not stripped.ff_bias


def test_a_trunk_that_is_not_a_whole_number_of_heads_is_refused(tmp_path):
    from thenoise.dit.ltx_video.checkpoint import detect_params

    path = _geometry_file(tmp_path, name="odd.safetensors",
                          **{"model.diffusion_model.patchify_proj.weight": torch.zeros(100, 4)})
    with pytest.raises(ValueError, match="video trunk 100 is not a multiple"):
        detect_params(path)


def test_a_checkpoint_key_reaches_the_parameter_it_fills():
    from thenoise.dit.ltx_video import keys

    assert keys.split_ltx_key("model.diffusion_model.prompt_adaln_single.linear.weight") == (
        "dit", "prompt_adaln.linear.weight"
    )
    # A file named for one part spells nothing, so its keys are the module's own.
    assert keys.component_for("upsampler.0.weight", named=("upscaler",)) == (
        "upscaler", "upsampler.0.weight")


def test_a_gemma4_key_lands_where_the_built_model_keeps_it():
    """The released 2.5 files spell their keys one way; the model nests them deeper.

    Checked against the official BF16 text encoder and the BF16 and INT8 ConvRot builds
    `scripts/download.py` installs, whose key sets are identical bar the quantization
    sidecars: these are the prefixes they use, and the module each file's own
    `gemma_config` metadata builds for them.
    """
    from thenoise.text_encoders.ltx_video import _gemma4_key

    for raw, want in (
        ("model.layers.0.mlp.down_proj.weight",
         "model.model.language_model.layers.0.mlp.down_proj.weight"),
        ("model.norm.weight", "model.model.language_model.norm.weight"),
        ("vision_model.patch_dense.weight", "model.model.embed_vision.patch_dense.weight"),
        ("multi_modal_projector.embedding_projection.weight",
         "model.model.embed_vision.multimodal_embedder.embedding_projection.weight"),
        ("audio_projector.embedding_projection.weight",
         "model.model.embed_audio.embedding_projection.weight"),
    ):
        assert _gemma4_key(raw) == ("gemma", want)

    # Assets, and the LTX connectors that load as a component of their own.
    for raw in ("tokenizer_json", "hf_asset__tokenizer_config.json",
                "text_embedding_projection.video_aggregate_embed.weight"):
        assert _gemma4_key(raw) is None
    # An unmapped learned key also comes back None, and the loader refuses it rather
    # than skipping a tensor; only the generation head is dropped without a word.
    assert _gemma4_key("vision_tower.blocks.0.attn.qkv.weight") is None
    assert _gemma4_key("lm_head.weight") is None


def test_a_packed_gemma_asset_is_read_whichever_way_the_packer_stored_it(tmp_path):
    from safetensors.torch import save_file
    from thenoise.text_encoders.ltx_video import _read_asset
    from thenoise.utils.safetensors import MemoryEfficientSafeOpen

    config = b'{"bos_token": "<bos>"}'
    packed = tmp_path / "packed.safetensors"
    save_file({"tokenizer_json": torch.tensor(list(b"{}"), dtype=torch.uint8),
               "hf_asset__tokenizer_config.json": torch.tensor(list(config), dtype=torch.uint8)},
              str(packed))
    with MemoryEfficientSafeOpen(str(packed)) as f:
        assert _read_asset(f, f.metadata(), "tokenizer_json") == b"{}"
        assert _read_asset(f, f.metadata(), "tokenizer_config.json") == config

    described = tmp_path / "described.safetensors"
    save_file({"model.embed_tokens.weight": torch.zeros(2, 2)}, str(described),
              metadata={"tokenizer_config.json": config.decode()})
    with MemoryEfficientSafeOpen(str(described)) as f:
        assert _read_asset(f, f.metadata(), "tokenizer_config.json") == config
        with pytest.raises(ValueError, match="lacks required"):
            _read_asset(f, f.metadata(), "chat_template.jinja")


def test_a_gemma3_checkpoint_key_reaches_the_model():
    from thenoise.text_encoders.ltx_video import _gemma3_key

    for raw in ("model.layers.0.self_attn.q_proj.weight",
                "gemma3_12b.model.layers.0.self_attn.q_proj.weight"):
        assert _gemma3_key(raw) == ("gemma", "model.layers.0.self_attn.q_proj.weight")


def test_a_gemma_built_on_meta_keeps_the_buffers_no_checkpoint_carries():
    """The encoder's deterministic constants have to be real, not meta.

    ``init_empty_weights`` patches parameter registration and leaves buffers alone, so
    the embedding scale and the RoPE frequencies are built once by the library. The
    learned weights stay on meta, which is what keeps the strict load strict: the
    alternative was rebuilding these two behind the loader, from the library's own
    internals, every time it changed them.
    """
    from accelerate import init_empty_weights
    from transformers import Gemma3TextConfig, Gemma3TextModel

    config = Gemma3TextConfig(vocab_size=64, hidden_size=16, num_hidden_layers=2,
                              num_attention_heads=2, num_key_value_heads=2,
                              intermediate_size=32, head_dim=8)
    with init_empty_weights():
        model = Gemma3TextModel(config)

    assert model.embed_tokens.weight.is_meta, "learned weights still come from the file"
    scale = model.embed_tokens._buffers["embed_scale"]
    assert not scale.is_meta and scale.item() > 0
    frequencies = list(model.rotary_emb.buffers())
    assert frequencies and not any(freq.is_meta for freq in frequencies)


def test_denoise_scales_timesteps_and_does_not_mark_regular_frames():
    from thenoise.models.ltx_video import Clip, LTXVideo
    from thenoise.models.video_base import Segment
    from thenoise.samplers import Step

    class RecordingDiT:
        dtype = torch.float32
        config = SimpleNamespace(timestep_scale_multiplier=1000)

        def __init__(self):
            self.arguments = None

        def __call__(self, **kwargs):
            self.arguments = kwargs
            return (torch.zeros_like(kwargs["hidden_states"]),
                    torch.zeros_like(kwargs["audio_hidden_states"]))

    for version, i2v in (("2.3", False), ("2.5", False), ("2.5", True)):
        dit = RecordingDiT()
        ref = torch.rand(1, 128, 1, 2, 3) if i2v else None
        fake = object.__new__(LTXVideo)
        fake.device = "cpu"
        fake.dit = dit
        fake.version = version
        fake._clip = Clip(reference_low=ref)
        # A step of the main half: its first sigma is still pure noise.
        fake._segment = Segment("dit", None, (Step(t=1.0, delta=0.25),), 1.0,
                                SimpleNamespace(fps=24))
        fake._geometry = lambda *_: (2, 2, 3)
        nv = 12
        x = torch.randn(1, nv + 4, 128)
        cond = SimpleNamespace(video=torch.randn(1, 2, 4096),
                               audio=torch.randn(1, 2, 2048), mask=torch.ones(1, 2),
                               null_video=None, null_audio=None, null_mask=None)
        out = fake.denoise_step(x, torch.tensor(.75), cond, 1., 0)
        assert out.shape == x.shape
        args = dit.arguments
        assert args["sigma"].item() == 750
        assert args["audio_sigma"].item() == 750
        assert args["audio_timestep"].item() == 750
        # Plain T2V tokens and I2V reference latents are not standalone
        # generated keyframe slots; only those slots receive a keyframe mask.
        assert args["video_keyframes_mask"] is None
        if i2v:
            assert torch.all(args["timestep"][:, :6] == 0)
            assert torch.all(args["timestep"][:, 6:] == 750)
        else:
            assert args["timestep"].item() == 750


def test_starting_frame_is_encoded_in_the_codecs_own_dtype():
    """A BF16 codec handed the fp32 frame the resize produced.

    The cast is the codec's own, at its own boundary, so neither half of it can be
    forgotten: the adapter passes what it has, and the latents come back in the
    precision the rest of the run continues in.
    """
    from thenoise.models.ltx_video import LTXVideo

    model = object.__new__(LTXVideo)
    model.device = "cpu"
    model.vae = tiny_ltx_video_vae().to(torch.bfloat16)
    low, high = model.encode_references([torch.zeros(3, 64, 64)])
    assert low.dtype is torch.bfloat16 and high.dtype is torch.bfloat16
    assert (low.shape, high.shape) == ((1, 8, 1, 4, 4), (1, 8, 1, 8, 8))


def test_denoise_guidance_runs_both_branches_with_one_scale():
    """One ``guidance_scale`` over the joined stream, combined like every other adapter."""
    from thenoise.models.ltx_video import AVConditioning, Clip, LTXVideo
    from thenoise.models.video_base import Segment
    from thenoise.samplers import Step

    class TwoBranchDiT:
        dtype = torch.float32
        config = SimpleNamespace(timestep_scale_multiplier=1000)

        def __init__(self):
            self.calls = 0

        def __call__(self, **kwargs):
            self.calls += 1
            level = 1.0 if self.calls == 1 else 2.0
            return (torch.full_like(kwargs["hidden_states"], level),
                    torch.full_like(kwargs["audio_hidden_states"], level))

    fake = object.__new__(LTXVideo)
    fake.device = "cpu"
    fake.dit = TwoBranchDiT()
    fake._clip = Clip()
    fake._segment = Segment("dit", None, (Step(t=1.0, delta=0.25),), 1.0,
                            SimpleNamespace(fps=24))
    fake._geometry = lambda *_: (2, 2, 3)
    x = torch.randn(1, 16, 128)
    ones = torch.ones(1, 2, 4096)
    cond = AVConditioning(ones, torch.ones(1, 2, 2048), torch.ones(1, 2),
                          null_video=ones, null_audio=torch.ones(1, 2, 2048),
                          null_mask=torch.ones(1, 2))
    out = fake.denoise_step(x, torch.tensor(0.5), cond, 3.0, 0)
    assert fake.dit.calls == 2
    # uncond + scale * (cond - uncond) with cond=1, uncond=2 -> -1, audio included.
    assert torch.allclose(out, torch.full_like(out, -1.0))


def test_native_ltx_vocoder_checkpoint_layout_and_hann_filter():
    from thenoise.dit.ltx_video.vocoder import LTXVocoderWithBWE

    with torch.device("meta"):
        model = checkpoint.make_vocoder()
    assert isinstance(model, LTXVocoderWithBWE)
    assert model.vocoder.conv_in.weight.shape == (1536, 128, 7)
    assert model.bwe_generator.conv_in.weight.shape == (512, 128, 7)
    assert model.mel_stft.stft_fn.forward_basis.shape == (514, 1, 512)
    assert model.resampler.filter.shape == (1, 1, 43)
    assert not model.resampler.filter.is_meta
    assert model.vocoder.conv_in.weight.is_meta
    assert "resampler.filter" not in model.state_dict()
    assert "vocoder.resnets.0.acts1.0.act.alpha" in model.state_dict()
    assert "vocoder.resnets.0.acts1.0.upsample.filter" in model.state_dict()
    assert "mel_stft.stft_fn.forward_basis" in model.state_dict()
    assert "mel_stft.mel_basis" in model.state_dict()


def test_native_ltx_audio_vae_checkpoint_layout_and_forward():
    from thenoise.vae.ltx_audio import LTXAudioVAE

    with torch.device("meta"):
        model = checkpoint.make_audio_vae()
    assert isinstance(model, LTXAudioVAE)
    assert model.encoder.conv_in.conv.weight.shape == (128, 2, 3, 3)
    assert model.decoder.conv_in.conv.weight.shape == (512, 8, 3, 3)
    assert model.latents_mean.shape == (128,)
    assert "encoder.down.0.block.0.conv1.conv.weight" in model.state_dict()
    assert "decoder.up.0.block.0.conv1.conv.weight" in model.state_dict()
    assert "latents_mean" in model.state_dict()
    small = LTXAudioVAE(base_channels=32, ch_mult=(1, 2), num_res_blocks=1,
                        resolution=32, latent_channels=4, mel_bins=8).eval()
    with torch.no_grad():
        output = small.decode(torch.randn(1, 4, 3, 8))
    assert output.shape == (1, 2, 9, 8)
    assert output.isfinite().all()


def test_native_ltx_vocoder_short_audio_waveform_and_streaming(tmp_path):
    from thenoise.dit.ltx_video.vocoder import LTXVocoderWithBWE

    cfg = dict(
        in_channels=8, hidden_channels=16, out_channels=2,
        upsample_kernel_sizes=[4], upsample_factors=[2],
        resnet_kernel_sizes=[3], resnet_dilations=[[1]],
        act_fn="leaky_relu", antialias=False, final_act_fn="tanh",
        bwe_in_channels=8, bwe_hidden_channels=16, bwe_out_channels=2,
        bwe_upsample_kernel_sizes=[8], bwe_upsample_factors=[6],
        bwe_resnet_kernel_sizes=[3], bwe_resnet_dilations=[[1]],
        bwe_act_fn="leaky_relu", bwe_antialias=False,
        filter_length=16, hop_length=2, window_length=16, num_mel_channels=4,
    )
    reference = LTXVocoderWithBWE(**cfg).eval()
    file = tmp_path / "vocoder.safetensors"
    save_file({"vocoder." + k: v.detach().contiguous()
               for k, v in reference.state_dict().items()}, str(file))
    with torch.device("meta"):
        candidate = LTXVocoderWithBWE(**cfg)
    loaded = _load_part("vocoder", candidate, file, "vocoder.")
    assert loaded is candidate
    assert not any(p.is_meta for p in candidate.parameters())
    candidate.eval()
    mel = torch.randn(1, 2, 8, 4)
    with torch.no_grad():
        got = candidate(mel)
        expected = reference(mel)
    assert got.shape == (1, 2, 48)
    torch.testing.assert_close(got, expected)


def test_native_ltx_audio_vae_streams_full_weights(tmp_path):
    from thenoise.vae.ltx_audio import LTXAudioVAE

    cfg = dict(base_channels=32, ch_mult=(1, 2), num_res_blocks=1,
               resolution=32, latent_channels=4, mel_bins=8)
    reference = LTXAudioVAE(**cfg).eval()
    file = tmp_path / "audio_vae.safetensors"
    save_file({"audio_vae." + k: v.detach().contiguous()
               for k, v in reference.state_dict().items()}, str(file))
    with torch.device("meta"):
        candidate = LTXAudioVAE(**cfg)
    loaded = _load_part("audio_vae", candidate, file, "audio_vae.")
    assert loaded is candidate
    assert not any(t.is_meta for t in candidate.parameters())
    candidate.eval()
    latents = torch.randn(1, 4, 3, 8)
    with torch.no_grad():
        torch.testing.assert_close(
            candidate.decode(latents),
            reference.decode(latents),
        )


def test_the_audio_codec_turns_stream_tokens_into_mel():
    """The stream carries audio as flat tokens, and the codec says how wide one is.

    The released pair is 8 channels of 16 mel bins -- the mel divided by what the
    encoder folds -- which used to be two literals in the middle of a decode.
    """
    from thenoise.vae.ltx_audio import LTXAudioVAE

    with torch.no_grad():
        vae = LTXAudioVAE(base_channels=8, ch_mult=(1, 2), num_res_blocks=1,
                          resolution=32, latent_channels=4, mel_bins=8).eval()
        flat = vae.latent_channels * vae.latent_bins
        vae.latents_mean.copy_(torch.linspace(-0.4, 0.4, flat))
        vae.latents_std.copy_(torch.linspace(0.8, 1.6, flat))
        assert (vae.latent_channels, vae.latent_bins) == (4, 4)

        tokens = torch.randn(1, 3, flat)
        mel = vae.decode_to_mel(tokens)
        assert mel.shape == (1, 2, 9, 8)
        direct = vae.decode((tokens * vae.latents_std + vae.latents_mean)
                            .unflatten(2, (4, 4)).transpose(1, 2))
        torch.testing.assert_close(mel, direct)


def _load_part(part: str, module, file, prefix: str = ""):
    """Fill one component from a file that spells it under ``prefix``."""
    from thenoise.utils.loader import load_selected_state

    load_selected_state(
        module, [(str(file), {prefix + key: key for key in module.state_dict()})],
        dtype=torch.float32, device="cpu", what=f"LTX {part}")
    module.eval()
    return module


def _tiny_components(monkeypatch):
    """Every LTX component factory replaced by a two-parameter stand-in.

    Geometry is read out of a header before any factory runs, and these stand-ins are
    not built to the names a real transformer uses, so the read is replaced with the
    numbers a two-parameter module wants.
    """
    m = checkpoint

    def linear():
        with torch.device("meta"):
            return torch.nn.Linear(2, 2, bias=False)

    def geometry(_header):
        return m.LTXParams(
            in_channels=2, out_channels=2, dim=128, num_layers=1,
            cross_attention_dim=2, audio_in_channels=2, audio_out_channels=2,
            audio_dim=64, audio_cross_attention_dim=2, ff_bias=True,
            audio_ff_bias=True, gated_attn=True, audio_gated_attn=True,
            cross_attn_mod=True, audio_cross_attn_mod=True, keyframes=False,
        )

    def dit(_params):
        model = torch.nn.Module()
        model.proj_in = linear()
        return model

    def connectors(_config, _params):
        model = torch.nn.Module()
        for modality in ("video", "audio"):
            block = torch.nn.Module()
            block.register_parameter("learnable_registers", torch.nn.Parameter(
                torch.empty(1, 2, device="meta")
            ))
            setattr(model, f"{modality}_connector", block)
            setattr(model, f"{modality}_text_proj_in", linear())
        return model

    def upscaler():
        model = torch.nn.Module()
        model.upsampler = torch.nn.Sequential(linear())
        return model

    monkeypatch.setattr(m, "detect_params", geometry)
    monkeypatch.setattr(m, "make_transformer", dit)
    monkeypatch.setattr(m, "make_connectors", connectors)
    for name in ("make_video_vae", "make_audio_vae", "make_vocoder"):
        monkeypatch.setattr(m, name, linear)
    monkeypatch.setattr(m, "make_upsampler", upscaler)
    return m


#: What the released all-in-one checkpoint holds: transformer, both codecs, vocoder and
#: text projections. Its spatial upscaler ships apart.
def _monolith_keys(weight):
    return {
        "model.diffusion_model.patchify_proj.weight": weight,
        "model.diffusion_model.video_embeddings_connector.learnable_registers":
            torch.ones(1, 2),
        "model.diffusion_model.audio_embeddings_connector.learnable_registers":
            torch.ones(1, 2),
        "text_embedding_projection.video_aggregate_embed.weight": weight,
        "text_embedding_projection.audio_aggregate_embed.weight": weight,
        "vae.weight": weight,
        "audio_vae.weight": weight,
        "vocoder.weight": weight,
    }


def _load(m, files, version="2.3"):
    return m.load_original_components(files, version=version,
                                      gemma_text_config=object(),
                                      device_for=lambda paths: "cpu",
                                      dtype=torch.float32)


def _upscaler_file(tmp_path, weight):
    return _save(tmp_path / "upscaler.safetensors", {"upsampler.0.weight": weight})


def test_one_file_answers_for_every_part_it_holds(tmp_path, monkeypatch):
    m = _tiny_components(monkeypatch)
    weight = torch.ones(2, 2)
    monolith = _save(tmp_path / "all-in-one.safetensors", _monolith_keys(weight))
    parts, sources = _load(m, {"": monolith, "upscaler": _upscaler_file(tmp_path, weight)})
    assert set(parts) == {"dit", "vae", "audio_vae", "vocoder", "upscaler", "connectors"}
    assert {sources[name] for name in ("dit", "vae", "audio_vae", "vocoder",
                                       "connectors")} == {monolith.resolve()}


def test_a_named_file_takes_its_part_from_the_all_in_one(tmp_path, monkeypatch):
    """Naming a separately packaged codec replaces the one inside the transformer file
    instead of colliding with it."""
    m = _tiny_components(monkeypatch)
    weight = torch.ones(2, 2)
    keys = _monolith_keys(weight)
    del keys["vae.weight"]
    monolith = _save(tmp_path / "all-in-one.safetensors", keys)
    codec = _save(tmp_path / "video-vae.safetensors", {"vae.weight": weight})
    parts, sources = _load(m, {"": monolith, "vae": codec,
                               "upscaler": _upscaler_file(tmp_path, weight)})
    assert sources["vae"] == codec.resolve()
    assert sources["vocoder"] == monolith.resolve()
    torch.testing.assert_close(parts["vae"].weight, weight)


def test_a_part_is_asked_where_the_files_that_carry_it_should_live(tmp_path, monkeypatch):
    """The memory manager decides placement from the files a part arrives from, and the
    read goes there directly -- a copy afterwards would double a 22 GB transformer."""
    m = _tiny_components(monkeypatch)
    weight = torch.ones(2, 2)
    keys = _monolith_keys(weight)
    del keys["vae.weight"]
    monolith = _save(tmp_path / "all-in-one.safetensors", keys)
    codec = _save(tmp_path / "video-vae.safetensors", {"vae.weight": weight})
    ups = _upscaler_file(tmp_path, weight)

    asked: list[list[Path]] = []

    def device_for(paths):
        asked.append(list(paths))
        return "cpu"

    m.load_original_components({"": monolith, "vae": codec, "upscaler": ups},
                               version="2.3", gemma_text_config=object(),
                               dtype=torch.float32, device_for=device_for)
    assert asked.count([monolith.resolve()]) == 4, "dit, audio_vae, vocoder, connectors"
    assert asked.count([codec.resolve()]) == 1
    assert asked.count([ups.resolve()]) == 1
    assert len(asked) == 6


def test_a_file_naming_a_part_without_fitting_it_is_not_taken_for_it(tmp_path, monkeypatch):
    """A codec the release is not does not become the codec because it shares a prefix.

    Only the file the request named for a part is held to it; where the engine went
    looking, tensors that do not fit are skipped, and what is left over is asked for by
    flag -- with the near miss named, since "pass --vae" is no help to a request that
    already passed a file with ``vae`` all over it.
    """
    m = _tiny_components(monkeypatch)
    weight = torch.ones(2, 2)
    keys = _monolith_keys(weight)
    del keys["vae.weight"]
    keys["vae.encoder.conv_in.weight"] = torch.ones(7, 7)
    monolith = _save(tmp_path / "all-in-one.safetensors", keys)
    upscaler = _upscaler_file(tmp_path, weight)
    parts, sources = _load(m, {"": monolith, "vae": _save(
        tmp_path / "video-vae.safetensors", {"vae.weight": weight}), "upscaler": upscaler})
    assert sources["vae"].name == "video-vae.safetensors"
    with pytest.raises(ValueError,
                       match=r"pass --vae \(all-in-one\.safetensors spells 1 tensors as vae"):
        _load(m, {"": monolith, "upscaler": upscaler})


def test_a_part_no_given_file_holds_names_the_flag(tmp_path, monkeypatch):
    m = _tiny_components(monkeypatch)
    weight = torch.ones(2, 2)
    keys = _monolith_keys(weight)
    del keys["vae.weight"]
    monolith = _save(tmp_path / "all-in-one.safetensors", keys)
    with pytest.raises(ValueError, match=r"LTX vae weights are in none of.*--vae"):
        _load(m, {"": monolith, "upscaler": _upscaler_file(tmp_path, weight)})


def test_split_connectors_are_complete_across_dit_and_text_file(tmp_path, monkeypatch):
    """A split LTX 2.5 connector cannot pass with missing learned weights."""
    m = _tiny_components(monkeypatch)



    weight = torch.ones(2, 2)
    path_dit = _save(tmp_path / "dit.safetensors", {
        "model.diffusion_model.patchify_proj.weight": weight,
        "model.diffusion_model.video_embeddings_connector.learnable_registers": torch.ones(1, 2),
        "model.diffusion_model.audio_embeddings_connector.learnable_registers": torch.ones(1, 2),
    })
    path_vae = _save(tmp_path / "vae.safetensors", {"vae.weight": weight})
    path_audio = _save(tmp_path / "audio.safetensors", {
        "audio_vae.weight": weight, "vocoder.weight": weight,
    })
    path_upscaler = _save(tmp_path / "upscaler.safetensors", {
        "upsampler.0.weight": weight,
    })
    path_text = tmp_path / "text.safetensors"

    def load():
        return m.load_original_components(
            {"": path_dit, "vae": path_vae, "audio_vae": path_audio,
             "upscaler": path_upscaler, "text_encoder": path_text},
            version="2.5", gemma_text_config=object(),
            device_for=lambda paths: "cpu", dtype=torch.float32,
        )

    projections = {
        "text_embedding_projection.video_aggregate_embed.weight": weight,
        "text_embedding_projection.audio_aggregate_embed.weight": weight,
    }
    _save(path_text, projections)
    parts, sources = load()
    assert set(parts) == {"dit", "vae", "audio_vae", "vocoder", "upscaler", "connectors"}
    assert all(not t.is_meta for module in parts.values() for t in module.parameters())

    _save(path_text, {"text_embedding_projection.video_aggregate_embed.weight": weight})
    with pytest.raises(ValueError, match="leave missing="):
        load()


def test_released_grids_survive_their_own_resampling():
    """The default step count must reproduce the published ladder, not approximate it."""
    from thenoise.dit.ltx_video.sampling import MAIN_GRID, REFINE_GRID, stage_grids

    assert stage_grids(8) == (list(MAIN_GRID), list(REFINE_GRID))
    for steps in (1, 2, 3, 5, 7, 9, 16):
        main, refine = stage_grids(steps)
        assert len(main) == steps + 1 and len(refine) >= 2
        assert main[0] == 1.0 and main[-1] == 0.0 and refine[-1] == 0.0
        assert all(a > b for a, b in zip(main, main[1:]))
        assert refine[0] == REFINE_GRID[0]  # the hand-over re-noise level is fixed


def test_a_clip_decoded_whole_asks_for_the_pixels_it_holds():
    """Not every frame leaves the decoder before the last latent is read.

    The released decoder pads both ends of time, so the clip's pixels are live together
    and a decode stage sized by one frame would be saved only by the memory retry.
    """
    from thenoise.models.ltx_video import LTX25Video
    from thenoise.video.spec import VideoSpec

    model = object.__new__(LTX25Video)
    model.checkpoint_prefs = {}
    model.vae = SimpleNamespace(spatial_compression_ratio=32, temporal_compression_ratio=8)
    clip = model.resolve_video(VideoSpec(prompt="moving fox", frames=34))
    assert model.decoded_pixels(clip) == clip.width * clip.height * clip.frame_count


def test_ltx_variants_resolve_with_shared_video_contract_and_fixed_refinement():
    """Both released variants use VideoModel, sigma_steps and shared Segments."""
    from thenoise.models.config import VideoSamplingParams
    from thenoise.models.ltx_video import LTX23Video, LTX25Video
    from thenoise.video.controller import VideoPipelineController
    from thenoise.video.spec import VideoSpec

    for model_cls in (LTX23Video, LTX25Video):
        model = object.__new__(model_cls)
        model.checkpoint_prefs = {}
        model.device = "cpu"
        model.dit = torch.nn.Identity()
        model.vae = SimpleNamespace(spatial_compression_ratio=32,
                                    temporal_compression_ratio=8)

        def params_for(resolved):
            return VideoSamplingParams(
                height=resolved.height, width=resolved.width,
                steps=resolved.steps, seed=0, guidance_scale=resolved.guidance_scale,
                sampler=resolved.sampler, frames=resolved.frame_count,
                flow_shift=resolved.flow_shift, fps=resolved.fps,
            )

        resolved = model.resolve_video(VideoSpec(prompt="moving fox"))
        assert (resolved.width, resolved.height, resolved.frame_count) == (768, 512, 33)
        assert (resolved.steps, resolved.sampler, resolved.output_format) == (8, "euler", "mp4")
        explicit = model.resolve_video(VideoSpec(prompt="moving fox", frames=34))
        assert explicit.frame_count == 41  # explicit frame request rounds up to 8n+1

        schedule = model.schedule(params_for(resolved))
        assert len(schedule) == 8
        assert [len(s.steps) for s in model.segments(params_for(resolved), schedule, 1.0)] == [8, 3]
        # The released recipe is the default, not a floor: another count resamples the
        # published grid and the refine pass keeps its share of it.
        for steps, refine in ((1, 1), (4, 2), (16, 6)):
            other = model.resolve_video(VideoSpec(prompt="moving fox", steps=steps))
            assert other.steps == steps
            spans = model.segments(other, model.schedule(params_for(other)), 1.0)
            assert [len(s.steps) for s in spans] == [steps, refine]
        # Guidance and a negative prompt belong to the request, not to the adapter.
        guided = model.resolve_video(VideoSpec(prompt="moving fox",
                                               negative_prompt="blurry", guidance_scale=3.5))
        assert (guided.guidance_scale, guided.negative_prompt) == (3.5, "blurry")
        # Sound rides in either container, so neither format is refused.
        for container in ("mp4", "webm"):
            clip = VideoPipelineController(model).resolve(
                VideoSpec(prompt="moving fox", output_format=container))
            assert clip.output_format == container


def test_native_ltx_spatial_upscaler_weights_and_forward():
    """x2 spatial, temporal length unchanged, original checkpoint state names."""
    from thenoise.dit.ltx_video.upscaler import LTXSpatialUpscaler

    with torch.device("meta"):
        prod = LTXSpatialUpscaler()
    assert prod.initial_conv.weight.shape == (1024, 128, 3, 3, 3)
    assert prod.upsampler[0].weight.shape == (4096, 1024, 3, 3)
    assert prod.final_conv.weight.shape == (128, 1024, 3, 3, 3)
    expected = {
        "initial_conv.weight", "initial_norm.weight", "res_blocks.0.conv1.weight",
        "upsampler.0.weight", "post_upsample_res_blocks.3.norm2.weight",
        "final_conv.weight",
    }
    assert expected.issubset(prod.state_dict())
    assert prod.dtype == torch.float32
    # CPU-size stub exercises the same parameter names and temporal handling.
    small = LTXSpatialUpscaler(in_channels=8, mid_channels=32, num_blocks_per_stage=1)
    sample = torch.randn(1, 8, 2, 4, 4)
    with torch.no_grad():
        result = small(sample)
    assert result.shape == (1, 8, 2, 8, 8)
    assert torch.isfinite(result).all()


def test_native_ltx_spatial_upscaler_streams_original_checkpoint_keys(tmp_path):
    """Shared streaming loader must materialize exactly the native parameter keys."""
    from thenoise.dit.ltx_video.upscaler import LTXSpatialUpscaler

    source = LTXSpatialUpscaler(in_channels=8, mid_channels=32, num_blocks_per_stage=1).eval()
    weights = {key: tensor.detach().contiguous() for key, tensor in source.state_dict().items()}
    checkpoint = tmp_path / "native_spatial_x2.safetensors"
    save_file(weights, str(checkpoint))

    with torch.device("meta"):
        target = LTXSpatialUpscaler(in_channels=8, mid_channels=32, num_blocks_per_stage=1)
    loaded = _load_part("upscaler", target, checkpoint)
    assert loaded is target
    assert all(not param.is_meta for param in target.parameters())
    target.eval()
    sample = torch.randn(1, 8, 2, 4, 4)
    with torch.no_grad():
        torch.testing.assert_close(target(sample), source(sample))


def test_ltx_gemma3_requires_single_safetensors_file(tmp_path):
    from thenoise.text_encoders.ltx_video import load_ltx_gemma
    with pytest.raises(ValueError, match="one existing"):
        load_ltx_gemma(tmp_path, version="2.3", device="cpu")
    with pytest.raises(ValueError, match="one existing"):
        load_ltx_gemma(tmp_path / "model-00001-of-00005.safetensors",
                       version="2.3", device="cpu")


def test_the_text_encoder_is_built_where_the_manager_says(monkeypatch):
    """Its weights are written at the residency device, not read onto CPU and carried
    back afterwards: that round trip holds the whole encoder in host RAM while the
    transformer is being read as well."""
    from pathlib import Path

    from thenoise.text_encoders import ltx_video

    seen = {}

    def record(module, checkpoint, **kwargs):
        seen.update(kwargs)
        return module

    monkeypatch.setattr(ltx_video, "load_text_encoder_weights", record)
    ltx_video._load_ltx_text_weights(torch.nn.Linear(1, 1), Path("gemma.safetensors"),
                                     lambda key: (None, key), (), "cuda:1")
    assert seen["device"] == "cuda:1"


def test_ltx_component_roles_and_lora_contract():
    from thenoise.models.ltx_video import LTX23Video, LTX25Video, LTXVideo
    assert LTXVideo.COMPONENT_ROLES == ("audio_vae", "upscaler", "text_projection")
    # 2.3 packs its codecs with the transformer, so only the encoder is a second file.
    assert LTX23Video.REQUIRED_COMPONENTS == ("text_encoder",)
    assert LTX25Video.REQUIRED_COMPONENTS == ("vae", "text_encoder")
    assert LTXVideo.lora_components(object()) == {}
