"""Small CPU contract tests for native LTX checkpoint loading and sampling."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from thenoise.dit.ltx_video.checkpoint import load_ltx_component
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]


def _checkpoint():
    path = ROOT / "thenoise/dit/ltx_video/checkpoint.py"
    spec = importlib.util.spec_from_file_location("_ltx_native_checkpoint_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _save(path, tensors):
    save_file({name: tensor.clone() for name, tensor in tensors.items()}, str(path))
    return path


def test_header_detection_and_mapping(tmp_path):
    m = _checkpoint()
    for version in ("2.3", "2.5"):
        data = {
            "model.diffusion_model.patchify_proj.weight": torch.ones(2, 2),
            "model.diffusion_model.audio_patchify_proj.weight": torch.ones(2, 2),
        }
        if version == "2.5":
            data["model.diffusion_model.keyframes_abs_pos_embedding"] = torch.ones(1, 2)
        path = _save(tmp_path / f"ltx-{version}-test.safetensors", data)
        assert m.version_from_checkpoint(path) == version
    assert m.split_ltx_key("model.diffusion_model.prompt_adaln_single.linear.weight") == (
        "dit", "prompt_adaln.linear.weight"
    )
    assert m._upscaler_key("upsampler.0.weight") == ("upscaler", "upsampler.0.weight")


def test_streamed_load_is_strict_and_materializes_parameters_and_buffers(tmp_path):
    m = _checkpoint()
    with torch.device("meta"):
        model = torch.nn.Linear(2, 3)
        model.register_buffer("running", torch.zeros(3))
    data = {"net.weight": torch.ones(3, 2), "net.bias": torch.ones(3),
            "net.running": torch.full((3,), 5.)}
    path = _save(tmp_path / "okay.safetensors", data)
    result = load_ltx_component(
        "dit", model, {path: {"dit"}}, cancel=lambda: None, dtype=torch.float32)
    assert result is model
    assert not any(x.is_meta for x in (*model.parameters(), *model.buffers()))
    torch.testing.assert_close(model(torch.ones(2)), torch.tensor([3., 3., 3.]))
    torch.testing.assert_close(model.running, torch.full((3,), 5.))

    path = _save(tmp_path / "bad.safetensors", {**data, "net.weight": torch.ones(3, 4)})
    with torch.device("meta"):
        model = torch.nn.Linear(2, 3)
        model.register_buffer("running", torch.zeros(3))
    with pytest.raises(ValueError, match="shape"):
        load_ltx_component(
            "dit", model, {path: {"dit"}}, cancel=lambda: None, dtype=torch.float32)
    assert model.weight.is_meta


def test_int8_requires_scale_and_comfy_quant_marker(tmp_path):
    m = _checkpoint()

    class QuantLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(2, 4, device="meta"))

        def load_quantized(self, weight, *, input_scale=None):
            self.weight = weight

    for missing in ("scale", "marker"):
        data = {"net.layer.weight": torch.ones(2, 4, dtype=torch.int8)}
        if missing != "scale":
            data["net.layer.weight_scale"] = torch.tensor([.01, .01])
        if missing != "marker":
            data["net.layer.comfy_quant"] = torch.tensor(
                list(json.dumps({"convrot": True}).encode()), dtype=torch.uint8
            )
        path = _save(tmp_path / f"missing_{missing}.safetensors", data)
        with pytest.raises(ValueError, match="requires a single"):
            load_ltx_component(
                "dit", torch.nn.ModuleDict({"layer": QuantLinear()}), {path: {"dit"}},
                cancel=lambda: None, dtype=torch.bfloat16)


def test_gemma3_wrapper_mapping_and_deterministic_meta_buffers():
    from thenoise.text_encoders.ltx_video import (
        _gemma3_key, _materialize_gemma3_constants,
    )
    for raw in ("model.layers.0.self_attn.q_proj.weight",
                "gemma3_12b.model.layers.0.self_attn.q_proj.weight"):
        assert _gemma3_key(raw) == ("gemma", "model.layers.0.self_attn.q_proj.weight")

    class Embedding(torch.nn.Module):
        scalar_embed_scale = 2.0

        def __init__(self):
            super().__init__()
            self.register_buffer("embed_scale", torch.empty((), device="meta"), persistent=False)
            self.weight = torch.nn.Parameter(torch.empty(2, 2, device="meta"))

    class Rotary(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.register_buffer("full_attention_inv_freq", torch.arange(2.), persistent=False)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = Embedding()
            with torch.device("meta"):
                self.rotary_emb = Rotary(None)

    model = Model()
    _materialize_gemma3_constants(model)
    assert model.embed_tokens.embed_scale.item() == 2
    assert not model.rotary_emb.full_attention_inv_freq.is_meta
    assert model.embed_tokens.weight.is_meta  # The learned weight must still load strictly.


def test_denoise_scales_timesteps_and_does_not_mark_regular_frames():
    from thenoise.models.ltx_video import LTXVideo

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
        fake = SimpleNamespace(
            device="cpu", dit=dit, version=version, _pass=1,
            _params=SimpleNamespace(fps=24), _ref_high=None, _ref_low=ref,
            _geometry=lambda *_: (2, 2, 3),
        )
        nv = 12
        x = torch.randn(1, nv + 4, 128)
        cond = SimpleNamespace(video=torch.randn(1, 2, 4096),
                               audio=torch.randn(1, 2, 2048), mask=torch.ones(1, 2))
        out = LTXVideo.denoise_step(fake, x, torch.tensor(.75), cond, 1., 0)
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


def test_native_ltx_vocoder_checkpoint_layout_and_hann_filter():
    from thenoise.dit.ltx_video.vocoder import LTXVocoderWithBWE

    with torch.device("meta"):
        model = _checkpoint().make_vocoder()
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
        model = _checkpoint().make_audio_vae()
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
        output = small.decode(torch.randn(1, 4, 3, 8), return_dict=False)[0]
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
    loaded = load_ltx_component(
            "vocoder", candidate, {file: {"audio_vae", "vocoder"}},
            cancel=lambda: None, dtype=torch.float32,
        )
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
    loaded = load_ltx_component(
            "audio_vae", candidate, {file: {"audio_vae", "vocoder"}},
            cancel=lambda: None, dtype=torch.float32,
        )
    assert loaded is candidate
    assert not any(t.is_meta for t in candidate.parameters())
    candidate.eval()
    latents = torch.randn(1, 4, 3, 8)
    with torch.no_grad():
        torch.testing.assert_close(
            candidate.decode(latents, return_dict=False)[0],
            reference.decode(latents, return_dict=False)[0],
        )


def test_split_connectors_are_complete_across_dit_and_text_file(tmp_path, monkeypatch):
    """A split LTX 2.5 connector cannot pass with missing learned weights."""
    m = _checkpoint()

    def linear():
        with torch.device("meta"):
            return torch.nn.Linear(2, 2, bias=False)

    def dit(_version):
        model = torch.nn.Module()
        model.proj_in = linear()
        return model

    def connectors(_version, _config):
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

    monkeypatch.setattr(m, "make_transformer", dit)
    monkeypatch.setattr(m, "make_connectors", connectors)
    for name in ("make_video_vae", "make_audio_vae", "make_vocoder"):
        monkeypatch.setattr(m, name, linear)
    monkeypatch.setattr(m, "make_upsampler", upscaler)

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
            path_dit, path_vae, path_audio, path_upscaler,
            version="2.5", gemma_text_config=object(),
            text_encoder_path=path_text, cancel=lambda: None, dtype=torch.float32,
        )

    projections = {
        "text_embedding_projection.video_aggregate_embed.weight": weight,
        "text_embedding_projection.audio_aggregate_embed.weight": weight,
    }
    _save(path_text, projections)
    parts = load()
    assert set(parts) == {"dit", "vae", "audio_vae", "vocoder", "upscaler", "connectors"}
    assert all(not t.is_meta for module in parts.values() for t in module.parameters())

    _save(path_text, {"text_embedding_projection.video_aggregate_embed.weight": weight})
    with pytest.raises(ValueError, match="checkpoint missing="):
        load()


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

        resolved = model.resolve_video(VideoSpec(prompt="moving fox"))
        assert (resolved.width, resolved.height, resolved.frame_count) == (768, 512, 33)
        assert (resolved.steps, resolved.sampler, resolved.output_format) == (8, "euler", "mp4")
        explicit = model.resolve_video(VideoSpec(prompt="moving fox", frames=34, sampler="native"))
        assert explicit.frame_count == 41  # explicit frame request rounds up to 8n+1
        assert explicit.sampler == "euler"  # reuse the shared solver registry

        params = VideoSamplingParams(
            height=resolved.height, width=resolved.width,
            steps=resolved.steps, seed=0, guidance_scale=1.0,
            sampler="euler", frames=resolved.frame_count,
            flow_shift=resolved.flow_shift, fps=resolved.fps,
        )
        schedule = model.schedule(params)
        assert len(schedule) == 8
        assert [len(s.steps) for s in model.segments(params, schedule, 1.0)] == [8, 3]

        with pytest.raises(ValueError, match="steps=8"):
            model.resolve_video(VideoSpec(prompt="moving fox", steps=4))
        with pytest.raises(ValueError, match="mp4"):
            VideoPipelineController(model).resolve(VideoSpec(prompt="moving fox", output_format="webm"))

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
    loaded = load_ltx_component(
            "upscaler", target, {checkpoint: {"upscaler"}},
            cancel=lambda: None, dtype=torch.float32,
        )
    assert loaded is target
    assert all(not param.is_meta for param in target.parameters())
    target.eval()
    sample = torch.randn(1, 8, 2, 4, 4)
    with torch.no_grad():
        torch.testing.assert_close(target(sample), source(sample))


def test_ltx_gemma3_requires_single_safetensors_file(tmp_path):
    from thenoise.text_encoders.ltx_video import load_ltx_gemma
    with pytest.raises(ValueError, match="one existing"):
        load_ltx_gemma(tmp_path, version="2.3", cancel=lambda: None)
    with pytest.raises(ValueError, match="one existing"):
        load_ltx_gemma(tmp_path / "model-00001-of-00005.safetensors",
                       version="2.3", cancel=lambda: None)


def test_ltx_gemma3_vendored_minimal_assets():
    from transformers import GemmaTokenizer
    from thenoise.utils.text_encoder import GEMMA3_LTX_CONFIG_DIR
    from pathlib import Path
    import json
    assets = Path(GEMMA3_LTX_CONFIG_DIR)
    cfg = json.loads((assets / "text_config.json").read_text())
    assert cfg["hidden_size"] == 3840 and cfg["num_hidden_layers"] == 48
    assert cfg["rope_scaling"] == {"factor": 8.0, "rope_type": "linear"}
    tokenizer = GemmaTokenizer(vocab_file=str(assets / "tokenizer.model"),
                               add_bos_token=True, add_eos_token=False)
    assert tokenizer("A red fox")["input_ids"][0] == tokenizer.bos_token_id


def test_ltx_auxiliary_path_roles_and_lora_contract():
    from thenoise.models.ltx_video import LTXVideo
    assert LTXVideo.AUXILIARY_PATHS == (
        "audio_vae_path", "spatial_upscaler_path", "text_projection_path")
    assert LTXVideo.lora_components(object()) == {}
