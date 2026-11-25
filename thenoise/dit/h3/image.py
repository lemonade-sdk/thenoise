"""H3-Image: the trunk as a still-image model.

Two things make it the image model rather than the trunk: an output head with one row per
interval of the sigma grid the checkpoint was distilled on, which is what makes it a
four-step model, and a packing of ``[text | references | target]`` whose reference rows
are conditioned rather than denoised.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from thenoise.dit.h3.layout import Layout, pad_to_patch
from thenoise.dit.h3.trunk import (
    COND_TIMESTEP,
    AdalnProj,
    H3DiT,
    H3Params,
    _spans,
)
from thenoise.dit.kvcache import KVCache, cache_mode
from thenoise.utils.dynamo import mark_token_axis
from thenoise.utils.fp32_linear import Fp32Linear
from thenoise.utils.rms_norm import RMSNorm


class FinalLayer(nn.Module):
    """Modulated output head over the target's rows: ``video_out`` stores one head per
    interval of ``pdd_sigmas``, row 0 complete and every later row an offset from it.
    """

    def __init__(self, params: H3Params, compute_dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.video_dim = params.video_dim
        self.pdd_steps = params.pdd_steps
        self.pdd_sigmas = tuple(params.pdd_sigmas)
        self.norm = RMSNorm(params.hidden_size, eps=params.final_norm_eps)
        self.adaln_proj = AdalnProj(params.adaln_dim, params.hidden_size, 2, 1,
                                    out_dtype=compute_dtype)
        self.video_out = Fp32Linear(params.hidden_size, params.video_dim * params.pdd_steps,
                                    bias=True)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, target: Tuple[int, int],
                sigma: float, sample_sigmas: Optional[torch.Tensor]) -> torch.Tensor:
        """The target rows' velocity; ``target`` is their ``(start, stop)``."""
        shift, scale = self.adaln_proj(t_emb)
        start, stop = target
        h = (self.norm(x[:, start:stop]) * (1.0 + scale[0]) + shift[0]).to(torch.float32)
        if self.pdd_steps == 1:
            return self.video_out(h)
        weight, bias = self._bank(sigma, sample_sigmas)
        return F.linear(h, weight, bias)

    def _bank(self, sigma: float, sample_sigmas: Optional[torch.Tensor]):
        """One head for a step: the intervals between its endpoints on the distilled
        grid, averaged with weights proportional to the sigma distance each covers.
        """
        if sample_sigmas is None:
            raise ValueError(
                "the PDD head bank needs the sigma schedule this run is stepping "
                "through; pass it as ``sample_sigmas``"
            )
        n = self.pdd_steps
        grid = self.pdd_sigmas
        i = int((sample_sigmas - sigma).abs().argmin())
        sigma_next = float(sample_sigmas[min(i + 1, sample_sigmas.shape[0] - 1)])
        start = min(range(n + 1), key=lambda k: abs(grid[k] - sigma))
        stop = min(range(n + 1), key=lambda k: abs(grid[k] - sigma_next))
        start = min(start, n - 1)
        stop = max(stop, start + 1)

        dt = [grid[k] - grid[k + 1] for k in range(start, stop)]
        weight_of = self.video_out.weight
        w = torch.tensor(dt, dtype=weight_of.dtype, device=weight_of.device) / sum(dt)
        rows = weight_of.reshape(len(grid) - 1, -1, weight_of.shape[1])
        bias_rows = self.video_out.bias.reshape(len(grid) - 1, -1)
        first = max(start, 1)
        return (
            rows[0] + torch.einsum("n,noi->oi", w[first - start:], rows[first:stop]),
            bias_rows[0] + torch.einsum("n,no->o", w[first - start:], bias_rows[first:stop]),
        )


class H3ImageDiT(H3DiT):
    """The image DiT: ``[text | references | target]`` in, the target's velocity out.

    ``context`` is a refined prompt — :meth:`refine_text` runs once per sampling.
    """

    def __init__(self, params: H3Params,
                 compute_dtype: Optional[torch.dtype] = None):
        super().__init__(params, compute_dtype)
        self.final_layer = FinalLayer(params, compute_dtype)

    def _codec_rows(self, target: torch.Tensor, refs: Sequence[torch.Tensor],
                    seed: int) -> torch.Tensor:
        """Patchified rows in packing order: the target, then the references.

        Each reference restarts the RNG stream on purpose: a second one must not
        inherit the draws the first consumed.
        """
        rows = [self.patchify(target.to(torch.float32))]
        for ref in refs:
            padded, _ = pad_to_patch(ref.to(torch.float32), self.params.patch_size)
            r = self.patchify(padded)
            gen = torch.Generator(device=r.device).manual_seed(seed)
            noise = torch.randn(r.shape, generator=gen, device=r.device,
                                dtype=torch.float32)
            rows.append(COND_TIMESTEP * r + (1.0 - COND_TIMESTEP) * noise)
        return torch.cat(rows, dim=0)

    def forward(
        self,
        latent: torch.Tensor,
        sigma: float,
        context: torch.Tensor,
        layout: Layout,
        refs: Sequence[torch.Tensor] = (),
        tags: Optional[torch.Tensor] = None,
        seed: int = 0,
        sample_sigmas: Optional[torch.Tensor] = None,
        rope: str = "cond",
        kv: Optional[KVCache] = None,
    ) -> torch.Tensor:
        """One forward pass; returns the target's velocity, ``[1, C, H, W]``.

        ``latent`` is repeat-padded to a whole patch, the padding cut back off the
        result; ``refs`` are the reference latents in :attr:`Layout` order and ``tags``
        the text encoder's per-token modality tags; ``sample_sigmas`` is the schedule
        being stepped through, which a head bank of more than one row requires.

        A cached run walks the whole sequence once, then runs its leading rows reading
        the references back from the buffers — the only rows whose K/V is step-invariant
        (:data:`COND_TIMESTEP`).
        """
        device = latent.device
        dtype = context.dtype
        padded, (orig_h, orig_w) = pad_to_patch(latent, self.params.patch_size)
        lat_h, lat_w = padded.shape[-2], padded.shape[-1]
        if (lat_h, lat_w) != (layout.latent_h, layout.latent_w):
            raise ValueError(
                f"layout was built for a {layout.latent_h}x{layout.latent_w} latent, "
                f"got {lat_h}x{lat_w}"
        )

        mode = cache_mode(kv, bool(refs))
        rows = layout.modulation_rows(tags)
        cos, sin = self.pe_embedder[rope]
        if mode == "read":
            n_rows = layout.target_span[1]
            rows = [row for row in rows if row[0] < n_rows]
            cos, sin = cos[..., :n_rows, :], sin[..., :n_rows, :]
            patched = self.video_patch_proj(self.patchify(padded)).to(dtype)
            h = torch.cat([context, patched.unsqueeze(0)], dim=1)
        else:
            patched = self.video_patch_proj(self._codec_rows(padded, refs, seed)).to(dtype)
            h = torch.empty(1, layout.seq_len, self.params.hidden_size,
                            dtype=dtype, device=device)
            codec = 0
            for start, stop, kind in layout.segments:
                if kind == "text":
                    h[:, start:stop] = context[0]
                else:
                    n = stop - start
                    h[:, start:stop] = patched[codec:codec + n]
                    codec += n

        # Row 0 of the modulation is this step's timestep, row 1 the reference's pinned
        # one; the pair keeps its shape at the last step, where they coincide.
        t = 1.0 - float(sigma)
        t_emb = self.time_embedding((t, max(t, COND_TIMESTEP)), device)
        kv_shape = (1, self.params.num_heads, layout.seq_len, self.params.head_dim)
        mark_token_axis(h, dim=1)
        mark_token_axis(cos, sin, dim=-2)
        for i, block in enumerate(self.blocks):
            # Buffers take the stream's dtype: ``attend`` copies k/v into them, and a
            # narrower buffer would hand attention a key narrower than the query.
            bufs = None if kv is None else kv.buffers(("attn", i), mode, kv_shape, h.dtype,
                                                      device)
            h = block(h, _spans(rows, block.adaln_proj(t_emb)), (cos, sin), bufs)
        if kv is not None and mode == "fill":
            kv.set_filled()

        out = self.final_layer(h, t_emb, layout.target_span, sigma, sample_sigmas)
        out = self.unpatchify(out[0], lat_h, lat_w)
        return -out[:, :, :orig_h, :orig_w].to(latent.dtype)


__all__ = ["H3ImageDiT"]
