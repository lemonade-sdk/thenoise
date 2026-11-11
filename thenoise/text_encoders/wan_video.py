"""UMT5-XXL encoder for Comfy-Org Wan checkpoints."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import sentencepiece as spm
import torch
from safetensors import safe_open
from torch import Tensor, nn
from torch.nn import functional as F

from thenoise.dit.quantized import QuantizedLinear
from thenoise.utils.attention import AttentionParams, attention
from thenoise.utils.wan_weights import validated_key_map


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


def _text_path(root: Path) -> Path:
    for base in (root, root / "split_files"):
        for name in ("umt5_xxl_fp16.safetensors", "umt5_xxl_fp8_e4m3fn_scaled.safetensors"):
            candidate = base / "text_encoders" / name
            if candidate.is_file():
                return candidate
    raise FileNotFoundError("Wan text encoder missing: text_encoders/umt5_xxl_fp16.safetensors")


def _text_key(name: str) -> str:
    return name.removeprefix("text_encoders.").removeprefix("umt5xxl.transformer.").removeprefix("transformer.")


def load_wan_text_encoder(model_dir: str | Path, *, dtype: torch.dtype = torch.float16,
                          check_cancel: Callable[[], None] | None = None,
                          _test_config: dict | None = None) -> tuple[WanUMT5, WanSentencePiece]:
    """Load Comfy-Org's UMT5 weights and tokenizer without Transformers."""
    checkpoint = _text_path(Path(model_dir).expanduser().resolve())
    if "fp8" in checkpoint.name:
        raise ValueError("FP8 UMT5 is not yet supported; use umt5_xxl_fp16.safetensors")
    with torch.device("meta"):
        model = WanUMT5(**(_test_config or {}))
    expected = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    if check_cancel:
        check_cancel()
    with safe_open(str(checkpoint), framework="pt", device="cpu") as sf:
        keys = set(sf.keys())
        if "spiece_model" not in keys:
            raise ValueError("Comfy UMT5 checkpoint is missing embedded spiece_model")
        mapping = validated_key_map(keys - {"spiece_model"}, _text_key, set(expected))
        for raw, target in mapping.items():
            if tuple(sf.get_slice(raw).get_shape()) != expected[target]:
                raise ValueError(f"UMT5 checkpoint shape mismatch: {raw} -> {target}")
            if sf.get_slice(raw).get_dtype() not in ("F16", "BF16", "F32"):
                raise ValueError(f"unsupported UMT5 tensor dtype: {raw}")
        if sf.get_slice("spiece_model").get_dtype() != "U8":
            raise ValueError("invalid Comfy UMT5 sentencepiece tokenizer data")
        tokenizer = WanSentencePiece(sf.get_tensor("spiece_model").numpy().tobytes())
        for i, (raw, target) in enumerate(sorted(mapping.items())):
            if check_cancel and i % 8 == 0:
                check_cancel()
            model.load_state_dict({target: sf.get_tensor(raw).to(dtype=dtype)}, strict=False, assign=True)
    if any(p.is_meta for p in model.parameters()):
        raise RuntimeError("UMT5 contains unloaded meta parameters")
    return model.eval(), tokenizer
