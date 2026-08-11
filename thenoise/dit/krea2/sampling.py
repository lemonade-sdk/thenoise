"""Sampler helpers for the K2 MMDiT (no Scheduler class).

These build the pieces of the K2 flow-matching sampler that are reused by the
model adapter: resolution-aware timestep scheduling, latent patchification, and
text-embedding gathering. The denoising loop itself lives in the shared
``DiffusionModel`` base class.
"""

import math

import torch
from einops import rearrange, repeat


def roundup(value, multiple, name):
    """Round `value` up to the nearest multiple, logging when padding is applied."""
    aligned = ((value + multiple - 1) // multiple) * multiple
    if aligned != value:
        print(f"[sample] {name}={value} is not a multiple of {multiple}; padding to {aligned}")
    return aligned


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
    max_len = max(v.shape[0] for v in valid)
    out = txt.new_zeros(txt.shape[0], max_len, txt.shape[2], txt.shape[3])
    newmask = torch.zeros(txt.shape[0], max_len, device=txt.device, dtype=torch.bool)
    for i, v in enumerate(valid):
        out[i, : v.shape[0]] = v
        newmask[i, : v.shape[0]] = True
    return out, newmask


def prepare(img, txtlen, patch, txtmask):
    """Patchify the latent and build the combined image+text position / mask tensors.

    Image tokens lead the sequence so each sample's valid tokens form a contiguous prefix
    ([img (all valid), text (valid prefix + padding)]), which the shared attention's
    key-padding-mask path uses. Returns (img_tokens, pos, mask).
    """
    b, _, h, w = img.shape
    h_, w_ = h // patch, w // patch
    imgids = torch.zeros((h_, w_, 3), device=img.device)
    imgids[..., 1] = torch.arange(h_, device=img.device)[:, None]
    imgids[..., 2] = torch.arange(w_, device=img.device)[None, :]
    imgpos = repeat(imgids, "h w three -> b (h w) three", b=b, three=3)
    imgmask = torch.ones(b, h_ * w_, device=img.device, dtype=torch.bool)
    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)

    txtpos = torch.zeros(b, txtlen, 3, device=img.device)
    mask = torch.cat((imgmask, txtmask), dim=1)
    pos = torch.cat((imgpos, txtpos), dim=1)
    return img, pos, mask


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
    ts = math.exp(mu) / (math.exp(mu) + (1.0 / ts - 1.0) ** sigma)
    return ts.tolist()


@torch.no_grad()
def encode_prompts(encoder, prompts, negative_prompts=None, *, cfg=True):
    """Encode prompts (and optional negatives) into gathered varlen text embeddings.

    Returns ``(txt, txtmask, untxt, untxtmask)``; the unconditional pair is ``None`` when
    ``cfg`` is False. ``gather_valid_text`` drops the interior padding the encoder
    inserts between prompt and suffix so the valid tokens form a contiguous prefix.
    The encoder stays resident (plenty of unified RAM); it is never freed/reloaded.
    """
    txt, txtmask = encoder(prompts)
    txt, txtmask = gather_valid_text(txt, txtmask)

    untxt = untxtmask = None
    if cfg:
        if negative_prompts is None:
            negative_prompts = [""] * len(prompts)
        untxt, untxtmask = encoder(negative_prompts)
        untxt, untxtmask = gather_valid_text(untxt, untxtmask)

    return txt, txtmask, untxt, untxtmask
