"""H3-Image: the trunk as a still-image model.

Two things make it the image model rather than the trunk: an output head with one row per
interval of the sigma grid the checkpoint was distilled on, which is what makes it a
four-step model, and a packing of ``[text | target | references]`` whose reference rows
are conditioned rather than denoised — last, so a run's K/V cache can freeze them as a
suffix.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch

from thenoise.dit.h3.layout import COND_TIMESTEP, Layout, pad_to_patch
from thenoise.dit.h3.trunk import FinalLayer, H3DiT, H3Params, _spans
from thenoise.dit.kvcache import KVCache, cache_mode
from thenoise.utils.dynamo import mark_token_axis


class H3ImageDiT(H3DiT):
    """The image DiT: ``[text | target | references]`` in, the target's velocity out.

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
        result; ``refs`` are the reference latents in :class:`Layout` order and ``tags``
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
        if layout.latents != (1, lat_h, lat_w):
            raise ValueError(
                f"layout was built for a {layout.latents} latent, got a still of "
                f"{lat_h}x{lat_w}")

        mode = cache_mode(kv, bool(refs))
        t = 1.0 - float(sigma)
        timesteps = layout.timesteps(t)
        rows = layout.modulation_rows(timesteps, tags)
        cos, sin = self.pe_embedder[rope]
        if mode == "read":
            n_rows = layout.video_span[1]
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

        # The distinct timesteps are known analytically, so the curve is read at exactly
        # those points: this step's for the prompt and target, the pinned one for the
        # references — one row rather than two at the last step, where they coincide.
        t_emb = self.time_embedding(timesteps.values, device)
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

        target = (*layout.video_span, timesteps.rows["video"])
        out = self.final_layer(h, t_emb, target, sigma, sample_sigmas)
        out = self.unpatchify(out[0], (lat_h, lat_w))
        return -out[:, :, :orig_h, :orig_w].to(latent.dtype)


__all__ = ["H3ImageDiT"]
