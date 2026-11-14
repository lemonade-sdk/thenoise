"""UMT5-XXL encoder for Comfy-Org Wan checkpoints."""
from __future__ import annotations

import logging
import math

import sentencepiece as spm
import torch
from torch import Tensor, nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

from thenoise.dit.quantized import QuantizedLinear
from thenoise.utils.attention import AttentionParams, attention
from thenoise.utils.loader import is_quantized_checkpoint, load_text_encoder_weights
from thenoise.utils.safetensors import MemoryEfficientSafeOpen

#: UMT5's gated FFN can overflow FP16 before its final normalization. BF16 keeps the
#: FP16 checkpoint's memory footprint but widens the exponent.
COMPUTE_DTYPE = torch.bfloat16


class _T5Norm(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(width))

    def forward(self, x: Tensor) -> Tensor:
        x_float = x.float()
        return (x_float * torch.rsqrt(x_float.square().mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.weight


class _T5Attention(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.q = QuantizedLinear(width, width, bias=False)
        self.k = QuantizedLinear(width, width, bias=False)
        self.v = QuantizedLinear(width, width, bias=False)
        self.o = QuantizedLinear(width, width, bias=False)
        # ComfyUI UMT5 uses the relative-position weights from every block.
        self.relative_attention_bias = nn.Embedding(32, heads)

    @staticmethod
    def _buckets(length: int, device: torch.device) -> Tensor:
        positions = torch.arange(length, device=device, dtype=torch.long)
        relative = positions[None] - positions[:, None]
        half = 16
        result = (relative > 0).long() * half
        distance = relative.abs()
        # Clamp before log; torch.where evaluates both sides.
        big = 8 + (torch.log((distance.clamp_min(8) / 8).float()) /
                   math.log(128 / 8) * 8).long()
        return result + torch.where(distance < 8, distance, big.clamp_max(15))

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        batch, length, width = x.shape
        dim = width // self.heads
        q = self.q(x).reshape(batch, length, self.heads, dim).transpose(1, 2)
        k = self.k(x).reshape(batch, length, self.heads, dim).transpose(1, 2)
        v = self.v(x).reshape(batch, length, self.heads, dim).transpose(1, 2)
        bias = self.relative_attention_bias(self._buckets(length, x.device))
        bias = bias.permute(2, 0, 1)[None].to(x.dtype)
        # T5's attention scores are unscaled, unlike ordinary SDPA.
        bias = bias.masked_fill(~mask[:, None, None, :], torch.finfo(x.dtype).min)
        attended = attention(
            [q, k * math.sqrt(dim), v], attn_params=AttentionParams(attention_mask=bias)
        )
        return self.o(attended)


class _T5AttentionLayer(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.layer_norm = _T5Norm(width)
        self.SelfAttention = _T5Attention(width, heads)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        return x + self.SelfAttention(self.layer_norm(x), mask)


class _T5GatedFFN(nn.Module):
    def __init__(self, width: int, ff_dim: int):
        super().__init__()
        self.wi_0 = QuantizedLinear(width, ff_dim, bias=False)
        self.wi_1 = QuantizedLinear(width, ff_dim, bias=False)
        self.wo = QuantizedLinear(ff_dim, width, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.wo(F.gelu(self.wi_0(x), approximate="tanh") * self.wi_1(x))


class _T5FFNLayer(nn.Module):
    def __init__(self, width: int, ff_dim: int):
        super().__init__()
        self.layer_norm = _T5Norm(width)
        self.DenseReluDense = _T5GatedFFN(width, ff_dim)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.DenseReluDense(self.layer_norm(x))


class _T5Block(nn.Module):
    def __init__(self, width: int, ff_dim: int, heads: int):
        super().__init__()
        self.layer = nn.ModuleList([_T5AttentionLayer(width, heads), _T5FFNLayer(width, ff_dim)])

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        return self.layer[1](self.layer[0](x, mask))


class _T5Encoder(nn.Module):
    def __init__(self, width: int, ff_dim: int, heads: int, layers: int):
        super().__init__()
        self.block = nn.ModuleList([_T5Block(width, ff_dim, heads) for _ in range(layers)])
        self.final_layer_norm = _T5Norm(width)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        for block in self.block:
            x = block(x, mask)
        return self.final_layer_norm(x)


class WanUMT5(nn.Module):
    def __init__(self, *, width: int = 4096, ff_dim: int = 10240, heads: int = 64,
                 layers: int = 24, vocab_size: int = 256384):
        super().__init__()
        if width % heads:
            raise ValueError("UMT5 width must be divisible by number of heads")
        self.shared = nn.Embedding(vocab_size, width)
        self.encoder = _T5Encoder(width, ff_dim, heads, layers)

    @property
    def dtype(self) -> torch.dtype:
        return self.shared.weight.dtype

    def forward(self, token_ids: Tensor, attention_mask: Tensor) -> Tensor:
        if token_ids.ndim != 2 or attention_mask.shape != token_ids.shape:
            raise ValueError("UMT5 needs matching [batch, tokens] ids and mask")
        hidden = self.encoder(self.shared(token_ids), attention_mask.bool())
        return hidden.masked_fill(~attention_mask[..., None].bool(), 0)


class WanSentencePiece:
    def __init__(self, model: bytes):
        self._tokenizer = spm.SentencePieceProcessor(model_proto=model, add_bos=False, add_eos=True)

    def encode(self, prompt: str, *, max_length: int = 512) -> tuple[Tensor, Tensor]:
        if max_length < 2:
            raise ValueError("max_length must allow sentencepiece EOS")
        ids = self._tokenizer.encode(prompt, out_type=int)
        ids = ids[:max_length]
        if not ids or ids[-1] != 1:
            ids = ids[:max_length - 1] + [1]
        count = len(ids)
        ids += [0] * (max_length - count)
        return torch.tensor([ids], dtype=torch.long), torch.tensor([[1] * count + [0] * (max_length - count)], dtype=torch.bool)



def _text_key(name: str) -> str:
    return name.removeprefix("text_encoders.").removeprefix("umt5xxl.transformer.").removeprefix("transformer.")


def _tokenizer_proto(path: str) -> bytes:
    """The SentencePiece proto ComfyUI embeds in its UMT5 file.

    It is tokenizer payload rather than module state, so the weight load drops the
    key and this reads it on its own.
    """
    with MemoryEfficientSafeOpen(path) as f:
        if "spiece_model" not in f.keys():
            raise ValueError(f"{path} is missing its embedded spiece_model tokenizer")
        if f.header["spiece_model"]["dtype"] != "U8":
            raise ValueError("invalid Comfy UMT5 sentencepiece tokenizer data")
        return f.get_tensor("spiece_model").numpy().tobytes()


def load_wan_text_encoder(
    path: str,
    device: str | torch.device,
    dtype: torch.dtype | None = None,
    *,
    params: dict | None = None,
) -> tuple[WanUMT5, WanSentencePiece]:
    """Load Comfy-Org's UMT5 weights and its embedded tokenizer, without Transformers.

    Quantized (FP8-scaled) and plain files both load here: the shared loader picks
    from the header, so nothing guesses from the filename. A plain file is read with
    its own dtypes and the module moved afterwards, which keeps the read from holding
    a second copy of an 11 GB encoder; a quantized one needs its full-precision
    tensors at the compute dtype while they are assigned, because quantized kernels
    emit BF16.
    """
    proto = _tokenizer_proto(path)
    compute = dtype or COMPUTE_DTYPE
    quantized = is_quantized_checkpoint(path)
    logger.info("Loading %sUMT5 from %s", "quantized " if quantized else "", path)
    with torch.device("meta"):
        model = WanUMT5(**(params or {}))
    load_text_encoder_weights(
        model, path, device=device, dtype=compute if quantized else None,
        key_map=_text_key, drop_keys=("spiece_model",),
    )
    if not quantized:
        model.to(dtype=compute)
    return model.eval(), WanSentencePiece(proto)
