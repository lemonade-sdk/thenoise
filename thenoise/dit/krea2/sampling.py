"""Sampler helpers for the K2 MMDiT (no Scheduler class).

These build the pieces of the K2 flow-matching sampler that are reused by the
model adapter: resolution-aware timestep scheduling, latent patchification, and
text-embedding gathering. The denoising loop itself lives in the shared
``DiffusionModel`` base class.
"""

import torch
from einops import rearrange

from thenoise.utils.math import generalized_time_shift
from thenoise.utils.positions import grid_positions
from thenoise.utils.sequence import make_key_padding_mask, pad_to_batch


def gather_valid_text(txt, mask):
    """Drop masked (invalid) text tokens so the valid ones form a contiguous prefix, then
    right-pad to the batch maximum.

    The Qwen3-VL conditioner pads the prompt to max_length and appends the template suffix,
    so its mask is [valid prompt, pad, valid suffix] — valid tokens are NOT a prefix. The
    shared attention handles padding via a key-padding mask, so interior padding is covered
    there; the trim below is still applied to keep each sample's valid tokens contiguous.
    Dropping it is lossless: text tokens get zero RoPE position
    and padding is masked out, so only the set/order of valid tokens matters.

    txt: (B, seq, L, D), mask: (B, seq) bool -> (B, max_valid, L, D), (B, max_valid) bool.
    """
    valid = [txt[i][mask[i]] for i in range(txt.shape[0])]  # list of (n_i, L, D)
    out, _, seqlens = pad_to_batch(valid)
    newmask = make_key_padding_mask(seqlens, txt.device, always=True)
    return out, newmask


def _patch_image(img, patch):
    """Patchify the canonical latent into target image tokens + RoPE grid + mask.
     Returns ``(tokens, pos, mask, h_, w_)`` where ``h_``/``w_`` are the token-grid dims.
    """
    b, _, h, w = img.shape
    h_, w_ = h // patch, w // patch
    # (t, h, w) grid with t=0 and a row-major h/w ordering.
    pos = grid_positions([1, h_, w_], dtype=torch.float32, device=img.device).unsqueeze(0).expand(b, -1, -1)
    mask = torch.ones(b, h_ * w_, device=img.device, dtype=torch.bool)
    tokens = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)
    return tokens, pos, mask, h_, w_


def prepare(img, txtlen, patch, txtmask):
    """Patchify the latent and build the combined image+text position / mask tensors.

    Image tokens lead the sequence so each sample's valid tokens form a contiguous prefix
    ([img (all valid), text (valid prefix + padding)]), which the shared attention's
    key-padding-mask path uses. Returns (img_tokens, pos, mask).
    """
    tokens, pos, mask, _, _ = _patch_image(img, patch)

    txtpos = torch.zeros(tokens.shape[0], txtlen, 3, device=img.device)
    mask = torch.cat((mask, txtmask), dim=1)
    pos = torch.cat((pos, txtpos), dim=1)
    return tokens, pos, mask


def prepare_edit(img, ref_latents, txtlen, patch, txtmask, ref_method="fit"):
    """Patchify the target latent and prepend the reference tokens (edit path).
    Returns ``(img_tokens, pos, mask, ref_len)``.
    """
    from thenoise.dit.krea2.reference import concat_reference, pack_reference

    if ref_method != "fit":
        raise ValueError(f"unsupported ref_latents_method {ref_method!r}; supported: 'fit'")

    b, _, h, w = img.shape
    h_, w_ = h // patch, w // patch
    target_tokens, target_pos, target_mask, _, _ = _patch_image(img, patch)

    ref_tokens, ref_pos, ref_mask = [], [], []
    for i, ref in enumerate(ref_latents):
        rt, rp = pack_reference(
            ref.to(img.device, dtype=img.dtype), h_, w_, patch, i + 1, device=img.device
        )
        ref_tokens.append(rt)
        ref_pos.append(rp)
        ref_mask.append(torch.ones(b, rt.shape[1], device=img.device, dtype=torch.bool))
    img_tokens = concat_reference(ref_tokens, target_tokens)
    img_pos = torch.cat([*ref_pos, target_pos], dim=1)
    img_mask = torch.cat([*ref_mask, target_mask], dim=1)
    ref_len = img_tokens.shape[1] - target_tokens.shape[1]

    txtpos = torch.zeros(b, txtlen, 3, device=img.device)
    pos = torch.cat((img_pos, txtpos), dim=1)
    mask = torch.cat((img_mask, txtmask), dim=1)
    return img_tokens, pos, mask, ref_len


def timesteps(seq_len, steps, x1, x2, y1=0.5, y2=1.15, sigma=1.0, mu=None):
    """Resolution-aware flow-matching timestep schedule (t: 1 -> 0).

    `mu` is interpolated linearly in image-sequence length between (x1,y1) and
    (x2,y2), then used to time-shift a uniform 1->0 grid. Pass an explicit `mu`
    to pin a constant shift regardless of resolution (used by the distilled
    checkpoint, which was trained at a fixed mu=1.15).
    """
    ts = torch.linspace(1, 0, steps + 1)
    if mu is None:
        slope = (y2 - y1) / (x2 - x1)
        mu = slope * seq_len + (y1 - slope * x1)
    ts = generalized_time_shift(ts, mu, sigma)
    return ts.tolist()


@torch.no_grad()
def encode_prompts(encoder, prompts, negative_prompts=None, *, cfg=True, images=None, grounding_px=768):
    """Encode prompts (and optional negatives) into gathered varlen text embeddings.

    Returns ``(txt, txtmask, untxt, untxtmask)``; the unconditional pair is ``None`` when
    ``cfg`` is False. ``gather_valid_text`` drops the interior padding the encoder
    inserts between prompt and suffix so the valid tokens form a contiguous prefix.
    The encoder stays resident (plenty of unified RAM); it is never freed/reloaded.

    When ``images`` is given the encoder runs the image-grounded (vision-token) path;
    the negatives are grounded on the same images (matching training's unconditional).
    """
    def _run(prompts):
        if images is not None:
            return encoder(prompts, images=images, grounding_px=grounding_px)
        return encoder(prompts)

    txt, txtmask = _run(prompts)
    txt, txtmask = gather_valid_text(txt, txtmask)

    untxt = untxtmask = None
    if cfg:
        if negative_prompts is None:
            negative_prompts = [""] * len(prompts)
        untxt, untxtmask = _run(negative_prompts)
        untxt, untxtmask = gather_valid_text(untxt, untxtmask)

    return txt, txtmask, untxt, untxtmask
