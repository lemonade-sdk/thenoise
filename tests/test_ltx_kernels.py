"""Small CPU regressions for native LTX 2.3/2.5 numerical components.

These tests exercise TheNoise contracts; no Diffusers installation or
migration-time reference model is required.
"""
from __future__ import annotations

from types import SimpleNamespace
import pytest
import torch
from torch import nn

from thenoise.dit.ltx_video.connectors import LTX2TextConnectors
from thenoise.dit.ltx_video.transformer import LTX2VideoTransformer3DModel
from thenoise.dit.ltx_video.checkpoint import make_video_vae
from thenoise.utils.rope import apply_rope_split_half, apply_ltx_rope_split_half
from thenoise.vae.ltx_video import LTXVideoVAE

from conftest import tiny_ltx_video_vae





def test_a_qk_norm_spans_the_projection_the_checkpoint_measured():
    """One RMS over the whole projection, before the split into heads.

    The width is numerical, not cosmetic: over ``heads * head_dim`` a norm mixes heads,
    over ``head_dim`` it does not, and only the first matches the released weight. It
    also fixes the layout the norm runs in, which this checkpoint puts before the split.
    """
    from thenoise.dit.ltx_video.transformer import _LTXAttention

    heads, head_dim = 4, 8
    attn = _LTXAttention(query_dim=16, cross_dim=16, heads=heads, head_dim=head_dim,
                         gated=False).eval()
    assert attn.qk_norm.query_norm.weight.shape == (heads * head_dim,)

    x = torch.randn(2, 5, 16)
    with torch.inference_mode():
        raw = attn.to_q(x)
        normed = attn.qk_norm.query_norm(raw)
    weight = attn.qk_norm.query_norm.weight
    whole = raw * torch.rsqrt(raw.square().mean(-1, keepdim=True) + 1e-6) * weight
    assert torch.allclose(normed, whole, atol=1e-6)

    per_head = raw.reshape(*raw.shape[:2], heads, head_dim)
    per_head = per_head * torch.rsqrt(per_head.square().mean(-1, keepdim=True) + 1e-6)
    per_head = per_head * weight.reshape(heads, head_dim)
    assert not torch.allclose(normed, per_head.reshape(raw.shape), atol=1e-3)


def test_a_prompt_runs_at_its_own_length_and_is_padded_afterwards():
    """Gemma sees the prompt; the connectors' register-block context is filled after it.

    The released pipeline left-pads every prompt to the context width and passes a mask
    through 48 layers. Nothing in the connector needs that -- it re-aligns the valid rows
    itself -- so the padding moved behind the encoder, where it is one copy.
    """
    from thenoise.models.ltx_video import _PromptTower
    from thenoise.text_encoders.ltx_video import PROMPT_CONTEXT

    class FakeGemma(nn.Module):
        def __init__(self):
            super().__init__()
            self.probe = nn.Parameter(torch.zeros(1))
            self.seen = []

        def forward(self, ids):
            self.seen.append(tuple(ids.shape))
            return torch.zeros(ids.shape[0], ids.shape[1], 4)

    class FakeConnectors(nn.Module):
        dtype = torch.float32

        def forward(self, hidden, mask):
            assert mask.shape == (hidden.shape[0], hidden.shape[1])
            return hidden, hidden, mask

    tower = _PromptTower(FakeGemma(), FakeConnectors())
    video, audio, mask = tower(torch.arange(6).reshape(1, 6))
    assert tower.gemma.seen == [(1, 6)], "the encoder saw the prompt, not the context"
    assert video.shape == audio.shape == (1, PROMPT_CONTEXT, 4)
    assert int(mask.sum()) == 6 and int(mask[0, :PROMPT_CONTEXT - 6].sum()) == 0


def test_a_prompt_arrives_as_bos_first_ids_truncated_to_the_context():
    from thenoise.text_encoders.ltx_video import PROMPT_CONTEXT, GemmaPrompter

    class Tokens:
        def __init__(self, ids, bos=2):
            self.ids, self.bos_token_id = ids, bos

        def __call__(self, text, **options):
            assert options["add_special_tokens"] is False, "the prompter adds the BOS"
            return {"input_ids": list(self.ids)}

    def prompter(ids, bos=2):
        return GemmaPrompter.from_transformers(Tokens(ids, bos))

    assert prompter([3, 4, 5]).encode("  a red fox  ").tolist() == [[2, 3, 4, 5]]
    assert prompter([2, 3]).encode("fox").tolist() == [[2, 3]]
    assert len(prompter([3] * 4000).encode("fox")[0]) == PROMPT_CONTEXT
    for bos in (None, -1):  # SentencePiece reports a missing BOS as -1
        with pytest.raises(ValueError, match="no BOS token"):
            prompter([3], bos)


def test_a_gemma3_file_carries_its_own_tokenizer(tmp_path):
    """Nothing is vendored for Gemma 3: the encoder file supplies the pieces.

    The released single-file checkpoints hold no config metadata and keep their
    SentencePiece proto among the weights, so a file without one is refused -- a
    vendored copy would be free to drift from the model it conditions.
    """
    import sentencepiece as spm
    from safetensors.torch import save_file
    from thenoise.text_encoders.ltx_video import GemmaPrompter
    from thenoise.utils.loader import spiece_model

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("A red fox walks through the snow. A cat sits on a chair.\n" * 30,
                      encoding="utf-8")
    spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(tmp_path / "sp"),
                                   vocab_size=64, hard_vocab_limit=False,
                                   pad_id=0, eos_id=1, unk_id=3, bos_id=2)
    proto = (tmp_path / "sp.model").read_bytes()
    weights = {"model.embed_tokens.weight": torch.zeros(4, 4)}
    with_proto = tmp_path / "gemma3.safetensors"
    save_file({**weights,
               "spiece_model": torch.tensor(list(proto), dtype=torch.uint8)}, str(with_proto))
    assert spiece_model(with_proto) == proto

    processor = spm.SentencePieceProcessor(model_proto=proto)
    ids = GemmaPrompter.from_sentencepiece(processor).encode("a red fox").tolist()[0]
    assert ids[0] == processor.bos_id()
    assert ids[1:] == processor.encode("a red fox", out_type=int)

    bare = tmp_path / "bare.safetensors"
    save_file(weights, str(bare))
    with pytest.raises(ValueError, match="no SentencePiece tokenizer"):
        spiece_model(bare)


@pytest.mark.parametrize("ltx25", [False, True])
def test_av_transformer_forward_and_checkpoint_variants(ltx25):
    torch.manual_seed(14)
    config = dict(
        in_channels=8, out_channels=8,
        num_attention_heads=4, attention_head_dim=16, cross_attention_dim=64,
        audio_in_channels=8, audio_out_channels=8,
        audio_num_attention_heads=4, audio_attention_head_dim=16,
        audio_cross_attention_dim=64, num_layers=1,
        ff_bias=not ltx25, audio_ff_bias=True,
        gated_attn=True, audio_gated_attn=True,
        cross_attn_mod=True, audio_cross_attn_mod=True,
        use_keyframes_abs_pos_embedding=ltx25,
    )
    model = LTX2VideoTransformer3DModel(**config).eval()
    args = dict(
        hidden_states=torch.randn(1, 4, 8),
        audio_hidden_states=torch.randn(1, 3, 8),
        encoder_hidden_states=torch.randn(1, 4, 64),
        audio_encoder_hidden_states=torch.randn(1, 4, 64),
        encoder_attention_mask=torch.tensor([[1, 0, 1, 1]], dtype=torch.float32),
        audio_encoder_attention_mask=torch.tensor([[1, 0, 1, 1]], dtype=torch.float32),
        timestep=torch.tensor([[0., 750., 750., 750.]]),
        audio_timestep=torch.tensor([750.]), sigma=torch.tensor([750.]),
        audio_sigma=torch.tensor([750.]), num_frames=1, height=2, width=2,
        audio_num_frames=3, use_cross_timestep=True,
    )
    with torch.inference_mode():
        video, audio = model(**args)
        video_isolated, audio_isolated = model(**args, isolate_modalities=True)
    assert video.shape == video_isolated.shape == (1, 4, 8)
    assert audio.shape == audio_isolated.shape == (1, 3, 8)
    assert all(t.isfinite().all() for t in (video, audio, video_isolated, audio_isolated))
    assert not torch.equal(video, video_isolated) or not torch.equal(audio, audio_isolated)

    # The LTX 2.5 keyframe and video-FFN bias difference is checkpoint ABI.
    state = model.state_dict()
    assert ("keyframes_abs_pos_embedding" in state) == ltx25
    assert ("transformer_blocks.0.ff.net.0.proj.bias" in state) != ltx25
    assert "transformer_blocks.0.audio_to_video_attn.to_k.weight" in state


def test_ltx_text_connectors_ignore_padded_features():
    torch.manual_seed(7)
    model = LTX2TextConnectors(
        caption_channels=8, text_proj_in_factor=3,
        video_connector_num_attention_heads=2,
        video_connector_attention_head_dim=16, video_connector_num_layers=2,
        video_connector_num_learnable_registers=4, video_gated_attn=True,
        audio_connector_num_attention_heads=2,
        audio_connector_attention_head_dim=8, audio_connector_num_layers=2,
        audio_connector_num_learnable_registers=4, audio_gated_attn=True,
        video_hidden_dim=32, audio_hidden_dim=16,
    ).eval()
    hidden = torch.randn(1, 8, 8, 3)
    mask = torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0]])
    changed = hidden.clone()
    changed[:, 2:] = torch.randn_like(changed[:, 2:]) * 200
    with torch.inference_mode():
        video, audio, output_mask = model(hidden, mask)
        video_changed, audio_changed, _ = model(changed, mask)
    assert video.shape == (1, 8, 32)
    assert audio.shape == (1, 8, 16)
    assert output_mask.shape == mask.shape
    assert bool(output_mask.all())
    torch.testing.assert_close(video, video_changed)
    torch.testing.assert_close(audio, audio_changed)


def test_ltx_batched_rope_preserves_shared_convention_and_dtype():
    torch.manual_seed(17)
    x = torch.randn(2, 3, 4, 8)
    phases = torch.randn(2, 3, 4, 4)
    cos, sin = phases.cos(), phases.sin()
    expected = torch.cat((x[..., :4] * cos - x[..., 4:] * sin,
                          x[..., 4:] * cos + x[..., :4] * sin), dim=-1)
    torch.testing.assert_close(apply_ltx_rope_split_half(x, cos, sin), expected)
    assert apply_ltx_rope_split_half(x.bfloat16(), cos, sin).dtype == torch.bfloat16
    legacy_cos = torch.cat((cos[0, 0], cos[0, 0]), -1)[:, None, None, :]
    legacy_sin = torch.cat((sin[0, 0], sin[0, 0]), -1)[:, None, None, :]
    assert apply_rope_split_half(x[:1, :2], legacy_cos, legacy_sin).shape == (1, 2, 4, 8)


def test_native_video_vae_meta_layout_and_frame_grid():
    prod = make_video_vae()
    state = prod.state_dict()
    assert all(t.is_meta for t in state.values())
    assert state["encoder.conv_in.conv.weight"].shape == (128, 48, 3, 3, 3)
    assert state["decoder.up_blocks.0.upsamplers.0.conv.conv.weight"].shape == (4096, 1024, 3, 3, 3)

    small = tiny_ltx_video_vae()
    with torch.inference_mode():
        for frames in (1, 9):
            pixels = torch.randn(1, 3, frames, 16, 16)
            latent = small.encode(pixels)
            assert latent.shape == (1, 8, (frames - 1) // 4 + 1, 2, 2)
            decoded = small.decode(latent)
            assert decoded.shape == pixels.shape
            assert torch.isfinite(decoded).all()
