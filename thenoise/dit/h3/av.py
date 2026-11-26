"""H3 audiovisual: the trunk as a clip with sound.

Both streams are denoised in one sequence, at different strengths: an audio sigma is the
video's re-shifted from twelve to three, so the sound is always further along than the
picture at the same step.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch

from thenoise.dit.h3.layout import COND_TIMESTEP, SEG_EMBED, Layout, pad_to_patch
from thenoise.dit.h3.sparse import sol_plan
from thenoise.dit.h3.trunk import (
    FinalLayer,
    H3DiT,
    H3Params,
    _spans,
    pack_audio,
    unpack_audio,
)
from thenoise.utils.dynamo import mark_token_axis
from thenoise.utils.attention import DEFAULT_ATTENTION


class H3AVDiT(H3DiT):
    """The audiovisual DiT: ``[text | conditions | references | audio | video]`` in, both
    velocities out. ``context`` is a refined prompt, once per run.
    """

    def __init__(self, params: H3Params,
                 compute_dtype: Optional[torch.dtype] = None):
        super().__init__(params, compute_dtype)
        if not params.audio_dim:
            raise ValueError(
                "this checkpoint has no audio stream; an image file is read by "
                "thenoise.dit.h3.image")
        self.final_layer = FinalLayer(params, compute_dtype)

    def _keyframe_rows(self, latent: torch.Tensor, seed: int) -> torch.Tensor:
        """A keyframe's patchified rows, noise-augmented to the conditioned timestep.

        The RNG stream restarts per keyframe: one must not inherit the draws the last
        consumed.
        """
        padded, _ = pad_to_patch(latent.to(torch.float32), self.params.patch_size)
        rows = self.patchify(padded)
        gen = torch.Generator(device=rows.device).manual_seed(seed)
        noise = torch.randn(rows.shape, generator=gen, device=rows.device,
                            dtype=torch.float32)
        return COND_TIMESTEP * rows + (1.0 - COND_TIMESTEP) * noise

    def forward(
        self,
        latent: torch.Tensor,
        audio: torch.Tensor,
        sigma: float,
        audio_sigma: float,
        context: torch.Tensor,
        layout: Layout,
        keyframes: Sequence[torch.Tensor] = (),
        tags: Optional[torch.Tensor] = None,
        seed: int = 0,
        sample_sigmas: Optional[torch.Tensor] = None,
        rope: str = "cond", attention: str = DEFAULT_ATTENTION,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One forward pass, both velocities out.

        The latents are the streams' own at their own sigmas — whatever the solver carries
        between steps is its business, not the network's — and the encoded conditions come
        back in :class:`Layout` order.
        """
        device = latent.device
        dtype = context.dtype
        padded, (orig_t, orig_h, orig_w) = pad_to_patch(latent, self.params.patch_size)
        if tuple(padded.shape[2:]) != layout.latents:
            raise ValueError(
                f"layout was built for a {layout.latents} latent, got "
                f"{tuple(padded.shape[2:])}")

        timesteps = layout.timesteps(1.0 - float(sigma), 1.0 - float(audio_sigma))
        rows = layout.modulation_rows(timesteps, tags)

        # One projection per stream, in segment order: keyframes ahead of the target,
        # since the packing puts the conditions first.
        video_rows = self.patchify(padded.to(torch.float32))
        if keyframes:
            video_rows = torch.cat([*[self._keyframe_rows(k, seed) for k in keyframes],
                                    video_rows])
        audio_rows = pack_audio(audio.to(torch.float32))
        if audio_rows.shape[0] != layout.audio_rows:
            raise ValueError(
                f"layout was built for {layout.audio_rows} audio rows, got "
                f"{audio_rows.shape[0]}")
        embeds = {"video": self.video_patch_proj(video_rows).to(dtype),
                  "audio": self.audio_patch_proj(audio_rows).to(dtype)}

        h = torch.empty(1, layout.seq_len, self.params.hidden_size, dtype=dtype, device=device)
        taken = {"video": 0, "audio": 0}
        for start, stop, kind in layout.segments:
            if kind == "text":
                h[:, start:stop] = context[0]
            else:
                stream = SEG_EMBED[kind]
                source = embeds[stream]
                off = taken[stream]
                h[:, start:stop] = source[off:off + (stop - start)]
                taken[stream] = off + (stop - start)

        # Read at the run's distinct timesteps: the step's two, plus the pinned one when
        # there is a condition to hold.
        t_emb = self.time_embedding(timesteps.values, device)
        cos, sin = self.pe_embedder[rope]
        mark_token_axis(h, dim=1)
        mark_token_axis(cos, sin, dim=-2)
        # One answer for the whole step, read outside the compiled blocks: a sigma read
        # inside them would be a guard, and every block would recompile every step.
        sol = sol_plan(layout=layout, tokens=h.shape[1], dtype=h.dtype, device=device,
                       head_dim=self.params.head_dim, sigma=sigma,
                       sample_sigmas=sample_sigmas, method=attention)
        for block in self.blocks:
            h = block(h, _spans(rows, block.adaln_proj(t_emb)), (cos, sin), None, sol)

        video_seg = (*layout.video_span, timesteps.rows["video"])
        audio_seg = (*layout.audio_span, timesteps.rows["audio"])
        out = self.final_layer(h, t_emb, video_seg, sigma, sample_sigmas, audio=audio_seg)
        velocity = self.unpatchify(out[0][0], tuple(padded.shape[2:]))
        velocity = velocity[:, :, :orig_t, :orig_h, :orig_w]
        return (-velocity.to(latent.dtype),
                -unpack_audio(out[1][0]).to(audio.dtype))


__all__ = ["H3AVDiT"]
