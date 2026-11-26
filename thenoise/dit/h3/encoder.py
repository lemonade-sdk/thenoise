"""The H3-Image conditioner — Qwen3-VL-32B truncated, no final norm.

No chat wrapper: the model was trained on raw text with labelled vision blocks spliced in,
so a system or user turn moves the conditioning off its distribution — and with no prefix
to drop, every token is conditioning. The cut took the final RMSNorm with it: the DiT reads
raw layer output, where a norm would fix a magnitude the DiT uses.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn
from transformers import Qwen3VLForConditionalGeneration

from thenoise.dit.quantized import replace_linears
from thenoise.utils.qwen_configs import QWEN3_VL_32B_H3_CONFIG
from thenoise.utils.safetensors import read_header
from thenoise.utils.text_encoder import (
    H3_SPECIAL_TOKENS,
    VIDEO_PAD,
    VISION_BLOCK,
    QWEN25_TOKENIZER_CONFIG_DIR,
    QWEN3_VL_TOKENIZER_OVERRIDES,
    find_tokenizer_dir,
    load_qwen3_vl_model,
    load_qwen3_vl_processor,
    load_qwen3_vl_tokenizer,
)

if TYPE_CHECKING:  # pragma: no cover - only for annotations
    from PIL import Image

logger = logging.getLogger(__name__)

#: Modality tag of a DiT-modulation group: 0 is visual, 1 is text.
TAG_VISUAL, TAG_TEXT = 0, 1

#: The token ids the language model substitutes every expanded vision patch with -- one for
#: a picture, one for a clip -- and the pad token a wholly empty prompt falls back to.
IMAGE_PAD_ID = QWEN3_VL_32B_H3_CONFIG["image_token_id"]
VIDEO_PAD_ID = QWEN3_VL_32B_H3_CONFIG["video_token_id"]
PAD_TOKEN_ID = QWEN3_VL_32B_H3_CONFIG["text_config"]["bos_token_id"]


def vision_token_tags(input_ids: torch.Tensor, image_pad_id: int = IMAGE_PAD_ID,
                      video_pad_id: int = VIDEO_PAD_ID) -> Tensor:
    """Per-token modality tags: 0 across each vision block, 1 elsewhere.

    Each run of picture or clip pads is widened by one token either side: the markers
    framing a condition describe the condition.
    """
    ids = input_ids.view(-1).tolist()
    tags = [TAG_TEXT] * len(ids)
    pads = (image_pad_id, video_pad_id)
    i = 0
    while i < len(ids):
        if ids[i] not in pads:
            i += 1
            continue
        start = i
        while i < len(ids) and ids[i] in pads:
            i += 1
        for j in range(max(0, start - 1), min(len(ids), i + 1)):
            tags[j] = TAG_VISUAL
    return torch.tensor(tags, dtype=torch.long)


def _truncate_qwen3_vl(config, layers: int) -> nn.Module:
    """A Qwen3-VL with ``layers`` decoder layers and no final norm.

    The depth is a property of the released file, not of the vendored config. The norm is
    replaced rather than deleted because the language model's forward calls it.
    """
    config.text_config.num_hidden_layers = layers
    model = Qwen3VLForConditionalGeneration._from_config(config)
    del model.lm_head
    model.model.language_model.norm = nn.Identity()
    replace_linears(model)
    return model


def layer_count(path: str) -> int:
    """How many decoder layers a text-encoder file holds, from its header."""
    layers = read_header(path).block_count("model.layers.")
    if not layers:
        raise ValueError(f"{path} has no Qwen3-VL decoder layers")
    return layers


#: How the audiovisual release labels a picture condition. The image release spells the
#: same thing without the brackets.
VIDEO_PICTURE_LABEL = "<Picture {}>: "

#: How the audiovisual release labels a clip and a sound condition. A sound earns its label
#: and no vision at all: the trunk reads it, the language model never sees it.
VIDEO_CLIP_LABEL = "<Video {}>: "
VIDEO_SOUND_LABEL = "<Audio {}>: "

#: How many frames a second the language model is shown of a reference clip: its vision
#: blocks take a pair each.
VISION_FRAMES_PER_SECOND = 2.0


class GuideMedia(NamedTuple):
    """One condition as the conditioner is asked to read it.

    ``kind`` is ``picture``, ``clip`` or ``sound``. A picture carries one frame and a sound
    none; a clip carries the frames it is shown with the source frame each came from and the
    source's rate, which is what each block's seconds marker is computed from.
    """

    kind: str
    frames: Tuple["Image.Image", ...] = ()
    indices: Tuple[int, ...] = ()
    fps: Optional[float] = None
    total_num_frames: Optional[int] = None


def presentation(items: Sequence[GuideMedia], prompt: str,
                 labels: dict) -> Tuple[str, list, list, list]:
    """The conditioning text for a run's conditions, and the frames to go with it.

    Ordinals count per kind across the whole run, in request order. A clip is one
    placeholder per pair of frames and nothing else: the processor writes the seconds marker
    and the vision framing around the pads, an odd last frame repeated to fill the patch. A
    sound is its label alone.
    """
    seen = dict.fromkeys(labels, 0)
    text: list[str] = []
    images: list = []
    videos: list = []
    metadata: list = []
    for item in items:
        label = labels.get(item.kind)
        if label is None:
            raise ValueError(f"this conditioner presents no {item.kind} conditions")
        seen[item.kind] += 1
        text.append(label.format(seen[item.kind]))
        if item.kind == "picture":
            images.append(item.frames[0])
            text.append(VISION_BLOCK)
        elif item.kind == "clip":
            for block, at in _pairs(item.frames, item.indices):
                videos.append(np.stack([np.asarray(frame) for frame in block]))
                metadata.append({"total_num_frames": item.total_num_frames
                                 or len(item.frames),
                                 "fps": item.fps, "frames_indices": list(at)})
                text.append(VIDEO_PAD)
    return "".join(text) + prompt, images, videos, metadata


def _pairs(frames: Sequence["Image.Image"], indices: Sequence[int]):
    """The frames two by two, the last one of an odd run repeated with its own index.

    A vision block's temporal patch takes two frames, and repeating is what the release
    does: the alternative is a block that shows one frame twice as if it were a still.
    """
    for start in range(0, len(frames), 2):
        block = list(frames[start:start + 2])
        at = list(indices[start:start + 2])
        if len(block) < 2:
            block.append(block[0])
            at.append(at[0])
        yield tuple(block), tuple(at)

#: Short edge of the picture the language model is shown, whichever release is asking. The
#: DiT is shown a larger one: the vision tower compresses 32x32 pixels into one token, so a
#: bigger copy buys detail the text stream cannot use.
VISION_SHORT_EDGE = 1024


class H3TextEncoder(nn.Module):
    """Qwen3-VL-32B conditioner: ``(prompt, media) -> (embeds, mask, tags)``.

    Both releases share the weights and the spliced-vision presentation; the labels are
    where they spell it differently, and a release with nothing to say about a kind of
    condition leaves that label out and refuses one.
    """

    def __init__(self, qwen: nn.Module, tokenizer, processor, picture_label: str,
                 clip_label: Optional[str] = None,
                 sound_label: Optional[str] = None) -> None:
        super().__init__()
        self.qwen = qwen
        self.tokenizer = tokenizer
        self.processor = processor
        self.labels = {"picture": picture_label, "clip": clip_label,
                       "sound": sound_label}

    @property
    def dtype(self) -> torch.dtype:
        return next(self.qwen.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.qwen.parameters()).device

    def encode(
        self, prompt: str,
        media: Optional[Sequence[Union["Image.Image", GuideMedia]]] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """``(embeds [1, L, 5120], mask [1, L], tags [1, L])`` for one prompt.

        ``media`` is the run's conditions in the order they are presented: a bare PIL image
        is a picture, which is all the image release ever has. One sequence is encoded at a
        time, so ``mask`` is all ones.
        """
        items = [one if isinstance(one, GuideMedia) else GuideMedia("picture", (one,))
                 for one in ([] if media is None else list(media))]
        text, images, videos, metadata = presentation(items, prompt, self.labels)

        call = {"text": [text], "images": images or None}
        if videos:
            # The frames are already sampled; asked to sample them the processor would
            # resample them at a rate of its own guessing.
            call.update(videos=videos, video_metadata=metadata,
                        videos_kwargs={"do_sample_frames": False})
        inputs = self.processor(**call)

        input_ids = torch.as_tensor(inputs["input_ids"]).to(self.device, torch.long)
        if input_ids.shape[-1] == 0:
            # An empty prompt with no references tokenizes to nothing at all.
            input_ids = torch.full((1, 1), PAD_TOKEN_ID, dtype=torch.long, device=self.device)

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "use_cache": False,
        }
        if images:
            model_inputs["pixel_values"] = torch.as_tensor(inputs["pixel_values"]).to(
                self.device, self.dtype
            )
            model_inputs["image_grid_thw"] = torch.as_tensor(inputs["image_grid_thw"]).to(
                self.device, torch.long
            )
        if videos:
            model_inputs["pixel_values_videos"] = torch.as_tensor(
                inputs["pixel_values_videos"]).to(self.device, self.dtype)
            model_inputs["video_grid_thw"] = torch.as_tensor(
                inputs["video_grid_thw"]).to(self.device, torch.long)
        if images or videos:
            # M-RoPE needs the per-token modality map to place the vision spans.
            model_inputs["mm_token_type_ids"] = torch.as_tensor(
                inputs["mm_token_type_ids"]
            ).to(self.device, torch.long)

        # The truncated model ends in an identity: nothing normalizes this output.
        embeds = self.qwen.model(**model_inputs).last_hidden_state
        tags = vision_token_tags(input_ids[0]).view(input_ids.shape).to(self.device)
        mask = torch.ones_like(input_ids)
        return embeds.to(self.dtype), mask, tags

    def forward(self, prompt: str, media=None):
        return self.encode(prompt, media)


def h3_tokenizer(tokenizer_dir: Optional[str] = None):
    """The conditioner's tokenizer: the shared Qwen3-VL table plus the release's own tokens.

    A checkpoint's own ``tokenizer/`` wins when it ships one; the vendored table is the
    fallback, and is where the extra ids are registered by hand.
    """
    tokenizer, _ = load_qwen3_vl_tokenizer(
        tokenizer_dir or QWEN25_TOKENIZER_CONFIG_DIR,
        max_length=QWEN3_VL_32B_H3_CONFIG["text_config"]["max_position_embeddings"],
        overrides=QWEN3_VL_TOKENIZER_OVERRIDES,
        additional_special_tokens=H3_SPECIAL_TOKENS,
    )
    return tokenizer


def load_h3_text_encoder(
    path: str,
    *,
    dtype: torch.dtype,
    device,
    picture_label: str,
    clip_label: Optional[str] = None,
    sound_label: Optional[str] = None,
    tokenizer_dir: Optional[str] = None,
) -> H3TextEncoder:
    """Load the truncated Qwen3-VL-32B conditioner plus its tokenizer and processor."""
    layers = layer_count(path)
    qwen = load_qwen3_vl_model(
        path,
        dtype=dtype,
        device=device,
        config=QWEN3_VL_32B_H3_CONFIG,
        model_factory=lambda config: _truncate_qwen3_vl(config, layers),
    )
    tokenizer_dir = tokenizer_dir or find_tokenizer_dir(path)
    tokenizer = h3_tokenizer(tokenizer_dir)
    encoder = H3TextEncoder(qwen, tokenizer, load_qwen3_vl_processor(tokenizer),
                    picture_label=picture_label, clip_label=clip_label,
                    sound_label=sound_label)
    logger.info("Loaded H3 text encoder (Qwen3-VL-32B, %d layers) from %s",
                layers, path)
    return encoder


__all__ = [
    "GuideMedia",
    "H3TextEncoder",
    "VISION_FRAMES_PER_SECOND",
    "VIDEO_CLIP_LABEL",
    "VIDEO_PICTURE_LABEL",
    "VIDEO_SOUND_LABEL",
    "VISION_SHORT_EDGE",
    "h3_tokenizer",
    "layer_count",
    "presentation",
    "load_h3_text_encoder",
    "vision_token_tags",
]
