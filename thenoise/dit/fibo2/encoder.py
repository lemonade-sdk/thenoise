"""Fibo 2.0 text encoder — Qwen3-VL-4B tapped at fifteen layers, handed over in bundles.

The DiT never sees a prompt. What it cross-attends to is six **bundles** of three
tapped hidden states, each normalized to unit RMS and concatenated: the Perceiver
reads the first bundle, each injection block its own. A tap is the *input* to that
decoder layer — the model's own final norm is not applied, the unit RMS norm replaces
it. Taps that repeat between bundles (layers 9/20/31 are both the Perceiver's and an
injection block's) are read out of the same run.

The prompt is a VGL caption: a JSON object the model was trained on, which a plain
sentence becomes by being left exactly as it is. With images to edit, the JSON carries
the vision markers and the instruction; and every ``text_render`` entry of a caption
has its text handed to the vision tower as a strip of rendered type, which is where
Fibo 2.0's text rendering comes from.

Loading follows the other Qwen3-VL conditioners: the model config is vendored, weights
come from one local safetensors file, and the tokenizer is either near the checkpoint,
or the vendored one.

The conditioning is never padded: the DiT carries no attention mask and the gist count
follows the prompt length, so padding would be attended to — and would move nothing,
which is the only thing that would make it harmless.
"""
from __future__ import annotations

import json
import logging
import math
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor, nn

from thenoise.utils.image_tensor import flatten_alpha, resize_to_long_edge
from thenoise.utils.qwen_configs import QWEN3_VL_4B_INSTRUCT_CONFIG
from thenoise.utils.text_encoder import (
    IM_END,
    IM_START,
    QWEN25_TOKENIZER_CONFIG_DIR,
    QWEN3_VL_TOKENIZER_OVERRIDES,
    VISION_BLOCK,
    find_tokenizer_dir,
    load_qwen3_vl_model,
    load_qwen3_vl_processor,
    load_qwen3_vl_tokenizer,
)

logger = logging.getLogger(__name__)

#: One user turn, then the assistant header the caption is cut off at.
TEMPLATE = IM_START + "user\n{}" + IM_END + "\n" + IM_START + "assistant\n"

#: The whole empty caption, as classifier-free guidance was trained with it: the
#: negative of a generation and of an edit alike, and never the images.
NULL_CAPTION = TEMPLATE.format(
    '{"short_description":"","objects":[],"background_setting":"","lighting":{},'
    '"aesthetics":{},"photographic_characteristics":{},"style_medium":"","context":"",'
    '"artistic_style":""}'
)

#: ``(9, 20, 31)`` is both the Perceiver's bundle and an injection block's.
BUNDLE_LAYERS = ((9, 20, 31), (5, 16, 27), (7, 18, 29), (9, 20, 31), (11, 22, 33), (13, 24, 35))
#: The eighteen taps that make the output rows, in bundle order: each is a decoder
#: layer, and the tap is the state ENTERING it, which is what ``output_hidden_states``
#: reports at that index (index 0 being the word embeddings).
BUNDLE_TAPS = tuple(layer for bundle in BUNDLE_LAYERS for layer in bundle)

#: Caption scores steer training, not the image; they never reach the prompt.
SCORES = ("aesthetic_score", "preference_score")

#: Longest side of an image the vision tower is asked about, as in training.
VISION_LONG_EDGE = 768

#: The image grid the model was trained at: multiples of the 32-pixel vision cell,
#: between 56x56 and 1252x1282 pixels of area.
MIN_PIXELS = 3136
MAX_PIXELS = 1_605_632

#: Unit-RMS normalization of the tapped hidden states.
NORM_EPS = 1e-6

#: Pillow's built-in face, at the size training rendered its text strips with. The
#: training-only fallback faces for non-Latin scripts are not shipped, so one face
#: stands in for every script, and a glyph it lacks is drawn as its notdef box.
TEXT_FONT_SIZE = 20

#: A rendered text strip breaks into a new line every this many characters.
TEXT_LINE_CHARS = 256


def dumps(value) -> str:
    """Compact JSON, in the caption's own key order and alphabet."""
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _empty(value) -> bool:
    if isinstance(value, float):
        return math.isnan(value)
    return value is None or (isinstance(value, (str, dict, list)) and not value)


def _clean(value):
    """A caption with every empty entry gone, recursively."""
    if isinstance(value, dict):
        value = {k: _clean(v) for k, v in value.items()}
        return {k: v for k, v in value.items() if not _empty(v)}
    if isinstance(value, list):
        return [v for v in map(_clean, value) if not _empty(v)]
    return value


def parse_prompt(prompt: str) -> Tuple[Optional[dict], Optional[str]]:
    """``(caption, edit instruction)`` out of a prompt.

    The caption is the JSON the model was trained with, minus its scores and empty
    entries; the instruction is either the ``edit_instruction`` key or, for a plain
    sentence, nothing at all — a prompt that is not a JSON object is a caption as it
    stands.
    """
    try:
        caption = json.loads(prompt)
    except json.JSONDecodeError:
        return prompt, None
    if not isinstance(caption, dict):
        return prompt, None

    instruction = caption.pop("edit_instruction", None)
    if "structured_caption" in caption:
        caption = caption.pop("structured_caption")
        if not isinstance(caption, dict):
            return (caption or None), instruction

    keys = set(caption)
    caption = _clean(
        {
            key: {k: v for k, v in value.items() if k not in SCORES}
            if isinstance(value, dict) else value
            for key, value in caption.items()
            if key not in SCORES
        }
    )
    # A caption that is only a description and a text list keeps both, the list even
    # when empty: the image has no text in it, and that is worth saying.
    if keys == {"short_description", "text_render"}:
        caption = {
            "short_description": caption.get("short_description") or "",
            "text_render": caption.get("text_render") or [],
        }
    return (caption or None), instruction


def _has_glyph(font: "ImageFont.FreeTypeFont", notdef, character: str) -> bool:
    """Whether a font can draw ``character``, judged by what it draws.

    A font without the glyph draws its notdef box — the same one it draws for U+FFFE,
    which no font has. A font that cannot be asked is taken to have it, as in training.
    """
    try:
        mask = font.getmask(character)
    except (OSError, ValueError):
        return True
    return not (mask.size == notdef.size and bytes(mask) == bytes(notdef))


def render_text_image(text: str, font: "ImageFont.FreeTypeFont") -> Image.Image:
    """A strip of ``text`` set in ``font``, black on white, as training drew it.

    Whitespace collapses and a line breaks every :data:`TEXT_LINE_CHARS` characters.
    Lines are placed by their bounding boxes — unless a glyph had to fall back to a
    notdef box, which forces every line onto the baseline at the full font height.
    """
    line = " ".join(text.split())
    lines = [line[i:i + TEXT_LINE_CHARS] for i in range(0, len(line), TEXT_LINE_CHARS)] or [""]
    notdef = font.getmask(chr(0xFFFE))
    if all(ord(c) < 128 or _has_glyph(font, notdef, c) for c in set(line)):
        min_box = font.getbbox("Ag")
        boxes = [font.getbbox(entry or " ") for entry in lines]
        heights = [max(1, min_box[3] - min_box[1], box[3] - box[1]) for box in boxes]
        width = max([1] + [box[2] - box[0] for box in boxes])
        origins, anchor = [(2 - box[0], -box[1]) for box in boxes], None
    else:
        ascent, descent = font.getmetrics()
        lines = [entry or " " for entry in lines]
        heights = [max(1, ascent + descent)] * len(lines)
        width = max(max(1, round(font.getlength(entry))) for entry in lines)
        origins, anchor = [(2.0, ascent)] * len(lines), "ls"

    image = Image.new("RGB", (width + 4, sum(heights) + 4 * len(lines)), "white")
    draw = ImageDraw.Draw(image)
    y = 2
    for entry, (x, dy), height in zip(lines, origins, heights):
        draw.text((x, y + dy), entry, font=font, fill=(0, 0, 0), anchor=anchor)
        y += height + 4
    return image


def _text_entries(caption) -> list:
    """The ``text_render`` entries a caption actually asks to render, in order."""
    if not isinstance(caption, dict):
        return []
    return [
        entry
        for entry in caption.get("text_render") or []
        if isinstance(entry, dict) and isinstance(entry.get("text"), str) and entry["text"]
    ]


def build_prompt(
    prompt: str, images: Union[None, Image.Image, Sequence[Image.Image]] = None
) -> Tuple[str, list]:
    """``(text, images)`` for the encoder, from a prompt and the images to edit.

    The result is one user turn: the tokenizer takes a string as it stands, so the chat
    framing happens here — and the null caption needs none, being a whole turn already,
    images and all: the negative of a generation or of an edit. Otherwise the images the
    vision tower sees are the edit images, capped to :data:`VISION_LONG_EDGE`, followed
    by one rendered strip per caption text — the order their vision markers are written
    into the JSON in.
    """
    images = [images] if isinstance(images, Image.Image) else list(images or [])
    if not prompt.strip():
        return NULL_CAPTION, []

    caption, instruction = parse_prompt(prompt)
    if images and instruction is None:
        # A plain instruction over images: the caption is the instruction.
        caption, instruction = None, prompt
    entries = _text_entries(caption)
    for entry in entries:
        entry["image_tokens"] = VISION_BLOCK

    if images:
        edit = (
            {"image": VISION_BLOCK}
            if len(images) == 1
            else {f"image_{k}": VISION_BLOCK for k in range(1, len(images) + 1)}
        )
        if caption:
            edit["structured_caption"] = caption
        edit["edit_instruction"] = instruction
        text = dumps(edit)
    elif isinstance(caption, str):
        text = caption
    else:
        text = dumps(caption or {})

    vision = [resize_to_long_edge(flatten_alpha(img), VISION_LONG_EDGE) for img in images]
    font = ImageFont.load_default(TEXT_FONT_SIZE)
    vision += [render_text_image(entry["text"], font) for entry in entries]
    return TEMPLATE.format(text), vision


class Fibo2TextEncoder(nn.Module):
    """Qwen3-VL-4B conditioner: ``(prompt, images) -> [1, L, 6 * hidden * 3]``.

    The six bundles along the channel axis, in :data:`BUNDLE_LAYERS` order.
    """

    def __init__(self, qwen: nn.Module, tokenizer, processor) -> None:
        super().__init__()
        self.qwen = qwen
        self.tokenizer = tokenizer
        self.processor = processor

    @property
    def dtype(self) -> torch.dtype:
        return next(self.qwen.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.qwen.parameters()).device

    def _to_model(self, value, dtype: torch.dtype) -> Tensor:
        return torch.as_tensor(value).to(self.device, dtype)

    def encode(
        self, prompt: str, images: Union[None, Image.Image, Sequence[Image.Image]] = None
    ) -> Tensor:
        """One prompt -> the conditioning the DiT reads, at the prompt's own length."""
        text, vision = build_prompt(prompt, images)
        inputs = self.processor(text=[text], images=vision or None)

        model_inputs = {
            "input_ids": self._to_model(inputs["input_ids"], torch.long),
            "attention_mask": self._to_model(inputs["attention_mask"], torch.long),
            "use_cache": False,
            "output_hidden_states": True,
        }
        if vision:
            model_inputs["pixel_values"] = self._to_model(inputs["pixel_values"], self.dtype)
            model_inputs["image_grid_thw"] = self._to_model(inputs["image_grid_thw"], torch.long)
            # M-RoPE needs the per-token modality map.
            model_inputs["mm_token_type_ids"] = self._to_model(
                inputs["mm_token_type_ids"], torch.long
            )

        states = self.qwen.model(**model_inputs).hidden_states
        taps = [
            F.rms_norm(states[tap].float(), (states[tap].shape[-1],), eps=NORM_EPS).to(self.dtype)
            for tap in BUNDLE_TAPS
        ]
        # [1, L, 18, hidden] -> [1, L, 6 bundles * 3 * hidden].
        return torch.stack(taps, dim=2).flatten(2)

    def forward(
        self, prompt: str, images: Union[None, Image.Image, Sequence[Image.Image]] = None
    ) -> Tensor:
        return self.encode(prompt, images)


def load_fibo2_text_encoder(
    path: str,
    *,
    dtype: torch.dtype,
    device: Union[str, torch.device],
    tokenizer_dir: Optional[str] = None,
) -> Fibo2TextEncoder:
    """Load the Qwen3-VL-4B conditioner + tokenizer/processor for Fibo 2.0.

    ``path`` is a single safetensors file (official HF or ComfyUI layout, bf16 or
    int8-convrot). The tokenizer defaults to a ``tokenizer/`` directory next to the
    checkpoint, else the vendored Qwen config.
    """
    qwen = load_qwen3_vl_model(
        path, dtype=dtype, device=device, config=QWEN3_VL_4B_INSTRUCT_CONFIG
    )
    tokenizer_dir = tokenizer_dir or find_tokenizer_dir(path) or QWEN25_TOKENIZER_CONFIG_DIR
    tokenizer, _ = load_qwen3_vl_tokenizer(
        tokenizer_dir,
        max_length=QWEN3_VL_4B_INSTRUCT_CONFIG["text_config"]["max_position_embeddings"],
        overrides=QWEN3_VL_TOKENIZER_OVERRIDES,
    )
    # The image grid is the trained one, narrower than the processor's default: the
    # vision tower is never asked about more pixels than training showed it.
    processor = load_qwen3_vl_processor(
        tokenizer, size={"shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS}
    )
    logger.info("Loaded Fibo 2.0 text encoder (Qwen3-VL-4B) from %s", path)
    return Fibo2TextEncoder(qwen, tokenizer, processor)


__all__ = [
    "BUNDLE_LAYERS",
    "BUNDLE_TAPS",
    "MAX_PIXELS",
    "MIN_PIXELS",
    "NULL_CAPTION",
    "TEMPLATE",
    "VISION_LONG_EDGE",
    "Fibo2TextEncoder",
    "build_prompt",
    "dumps",
    "load_fibo2_text_encoder",
    "parse_prompt",
    "render_text_image",
]
