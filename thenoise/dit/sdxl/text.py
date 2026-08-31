# SDXL text encoders: the dual-CLIP pair.
#
# The checkpoint stores the two CLIP encoders under ``conditioner.embedders``:
#   embedders.0 -> CLIP-L (HF ``CLIPTextModel`` layout, 768 dim, 12 layers)
#   embedders.1 -> CLIP-G (OpenCLIP ViT-bigG layout, 1280 dim, 32 layers)
#
# Both use the same CLIP BPE tokenizer (49408 vocab, 77 tokens). For the UNet's
# 2048-dim cross-attention the two penultimate hidden states are concatenated;
# the 1280-dim pooled output (used in the ADM vector) comes from CLIP-G only.
#
# CLIP-L is built with transformers' ``CLIPTextModel`` (a vendored config, no Hub
# fetch). CLIP-G is the OpenCLIP bigG text tower, implemented here directly (the
# fused-QKV ``in_proj_weight`` layout is not loadable by transformers), matching
# ``open_clip`` / ComfyUI's ``SDXLClipG`` exactly: penultimate hidden state for
# the cross-attention stream and ``text_projection(eos @ ln_final)`` for pooled.

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from thenoise.utils.setup_logging import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

#: CLIP-L (``openai/clip-vit-large-patch14``) config, vendored so the model is
#: built offline without fetching ``config.json`` from the Hub.
CLIP_L_CONFIG = {
    "architectures": ["CLIPTextModel"],
    "model_type": "clip_text_model",
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "max_position_embeddings": 77,
    "vocab_size": 49408,
    "layer_norm_eps": 1e-5,
    "initializer_range": 0.02,
    "attention_dropout": 0.0,
    "pad_token_id": 1,
    "bos_token_id": 49406,
    "eos_token_id": 49407,
    "projection_dim": 768,
}

#: OpenCLIP ViT-bigG text config (``laion/CLIP-ViT-bigG-14``).
CLIP_G_CONFIG = {
    "embed_dim": 1280,
    "text_layers": 32,
    "text_heads": 16,
    "text_width": 1280,
    "mlp_ratio": 4.0,
    "vocab_size": 49408,
    "text_max_len": 77,
}


class _OpenClipMultiheadAttention(nn.Module):
    """OpenCLIP fused-QKV attention (``nn.MultiheadAttention``-style).

    Keys: ``in_proj_weight`` (3*dim, dim), ``in_proj_bias``, ``out_proj``.
    """

    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.in_proj_weight = nn.Parameter(torch.empty(dim * 3, dim))
        self.in_proj_bias = nn.Parameter(torch.empty(dim * 3))
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, attn_mask=None):
        B, L, _ = x.shape
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=True)
        out = out.transpose(1, 2).reshape(B, L, self.dim)
        return self.out_proj(out)


class _OpenClipResidualBlock(nn.Module):
    """``ln_1 -> attn -> ln_2 -> mlp`` residual block. Keys match OpenCLIP."""

    def __init__(self, dim, num_heads):
        super().__init__()
        mlp_width = int(dim * 4.0)
        self.ln_1 = nn.LayerNorm(dim)
        self.attn = _OpenClipMultiheadAttention(dim, num_heads)
        self.ln_2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(dim, mlp_width)),
                    ("gelu", nn.GELU()),
                    ("c_proj", nn.Linear(mlp_width, dim)),
                ]
            )
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class OpenClipTextTransformer(nn.Module):
    """OpenCLIP bigG text tower (tokenizer-free).

    Keys are the bare OpenCLIP names: ``token_embedding``,
    ``positional_embedding``, ``transformer.resblocks.N.*``, ``ln_final``,
    ``text_projection``, ``logit_scale``.
    """

    def __init__(self, embed_dim=1280, layers=32, heads=16, vocab_size=49408, max_len=77):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Parameter(torch.empty(max_len, embed_dim))
        # ``transformer`` is a bare container whose ``resblocks`` ModuleList
        # produces the OpenCLIP keys ``transformer.resblocks.N.*``.
        self.transformer = nn.Module()
        self.transformer.resblocks = nn.ModuleList(
            [_OpenClipResidualBlock(embed_dim, heads) for _ in range(layers)]
        )
        self.ln_final = nn.LayerNorm(embed_dim)
        # Bare parameter (checkpoint key is ``text_projection``, not Linear).
        self.text_projection = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones(()))

    def _penultimate_and_pooled(self, x, ids):
        """Run the tower; return ``(context_hidden, projected_pooled)``.

        ``x`` is ``[B, 77, 1280]`` (token + positional embeddings). The
        penultimate hidden state (output of layer ``layers-2``) feeds the UNet's
        cross-attention. (The open_clip tower's final layer output is numerically
        unstable — std ~4.7 vs ~0.3 at the penultimate — so the penultimate is
        the usable context.) The pooled output is ``text_projection(eos @ ln_final)``
        at the end-of-text token.
        """
        hidden = x
        context = None
        blocks = self.transformer.resblocks
        n = len(blocks)
        for i, block in enumerate(blocks):
            hidden = block(hidden)
            if i == n - 2:
                context = hidden
        x = self.ln_final(hidden)
        idx = ids.argmax(dim=-1)
        pooled = x[torch.arange(x.shape[0], device=x.device), idx]
        pooled = pooled @ self.text_projection
        return context, pooled

    def forward(self, input_ids):
        """Encode token ids ``[B, 77]`` -> ``(context [B, 77, 1280], pooled [B, 1280])``."""
        ids = input_ids
        x = self.token_embedding(ids) + self.positional_embedding.unsqueeze(0)
        x = x.to(self.token_embedding.weight.dtype)
        return self._penultimate_and_pooled(x, ids)


def build_clip_l(sd, device="cpu", dtype=torch.bfloat16):
    """Build and load the CLIP-L text encoder from a bare ``text_model.*`` state dict."""
    from transformers import CLIPTextConfig, CLIPTextModel
    from accelerate import init_empty_weights

    config = CLIPTextConfig(**CLIP_L_CONFIG)
    with init_empty_weights():
        model = CLIPTextModel(config)
    # transformers keeps ``position_ids`` as a non-persistent buffer, but merged
    # checkpoints carry it; keep only keys the model actually has (transformers
    # rebuilds arange(77) anyway).
    expected = set(model.state_dict().keys())
    sd = {k: v for k, v in sd.items() if k in expected}
    info = model.load_state_dict(sd, strict=True, assign=True)
    if info.unexpected_keys or info.missing_keys:
        raise RuntimeError(
            f"CLIP-L checkpoint did not match CLIPTextModel: "
            f"missing={info.missing_keys[:10]}, unexpected={info.unexpected_keys[:10]}"
        )
    model.to(device)
    if dtype is not None:
        model.to(dtype)
    model.config.use_cache = False
    logger.info("Loaded CLIP-L text encoder")
    return model.eval().requires_grad_(False)


def build_clip_g(sd, device="cpu", dtype=torch.bfloat16):
    """Build and load the CLIP-G (OpenCLIP bigG) text tower from a bare state dict.

    The checkpoint stores CLIP-G in OpenCLIP layout (fused ``in_proj_weight``,
    ``transformer.resblocks.*``) and the SDXL UNet was trained against the
    open_clip-style text tower, so we load the weights into our OpenCLIP-shaped
    ``OpenClipTextTransformer``. The forward returns the penultimate hidden
    state (layer ``layers-2``) for cross-attention and
    ``text_projection(eos @ ln_final)`` for the pooled ADM vector.
    """
    cfg = CLIP_G_CONFIG
    model = OpenClipTextTransformer(
        embed_dim=cfg["embed_dim"],
        layers=cfg["text_layers"],
        heads=cfg["text_heads"],
        vocab_size=cfg["vocab_size"],
        max_len=cfg["text_max_len"],
    )
    # Merged checkpoints can carry stray buffers (e.g. ``position_ids``) not
    # present in the OpenCLIP tower; keep only keys the model actually has.
    expected = set(model.state_dict().keys())
    sd = {k: v for k, v in sd.items() if k in expected}
    info = model.load_state_dict(sd, strict=True, assign=True)
    if info.unexpected_keys or info.missing_keys:
        raise RuntimeError(
            f"CLIP-G checkpoint did not match OpenClipTextTransformer: "
            f"missing={info.missing_keys[:10]}, unexpected={info.unexpected_keys[:10]}"
        )
    model.to(device)
    if dtype is not None:
        model.to(dtype)
    logger.info("Loaded CLIP-G text encoder")
    return model.eval().requires_grad_(False)
    return model.eval().requires_grad_(False)
