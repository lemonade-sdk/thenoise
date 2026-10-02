"""PNG metadata helpers for generation images."""
from __future__ import annotations

import json
from typing import List, Optional

from PIL.PngImagePlugin import PngInfo, iTXt


def build_pnginfo(
    *,
    model: str,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    seed: int,
    upscale: bool,
    upscale_factor: float,
    upscale_type: str,
    sampler: str,
    qwen_vae_enhance: bool,
    film_grain: float,
    sharpening: float,
    lora_specs: Optional[List[str]],
    pixel_upscaler: Optional[str],
    sigmas: Optional[List[float]] = None,
) -> PngInfo:
    """Build a PngInfo object with generation metadata (JSON + human-readable).

    Writes a ``generation_data`` chunk with the full JSON of all resolved
    parameters and a ``parameters`` chunk of human-readable text.
    """
    pnginfo = PngInfo()

    gen_data = json.dumps({
        "model": model,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "upscale": upscale,
        "upscale_factor": upscale_factor,
        "upscale_type": upscale_type,
        "sampler": sampler,
        "qwen_vae_enhance": qwen_vae_enhance,
        "film_grain": film_grain,
        "sharpening": sharpening,
        "lora_specs": lora_specs,
        "pixel_upscaler": pixel_upscaler,
        "sigmas": sigmas,
    })
    pnginfo.add_text("generation_data", gen_data)

    # Human-readable "parameters" text (A1111-compatible): prompt lines, optional
    # "Negative prompt: " line, then one line of comma-separated "Key: value" pairs.
    parts: list[str] = [prompt]
    if negative_prompt:
        parts.append(f"Negative prompt: {negative_prompt}")

    meta_parts = [
        f"Model: {model}",
        f"Steps: {steps}",
        f"Sampler: {sampler}",
        f"Cfg scale: {guidance_scale}",
        f"Seed: {seed}",
    ]
    if upscale:
        meta_parts.append("Upscale: true")
    if upscale_factor != 1.0:
        meta_parts.append(f"Upscale factor: {upscale_factor:g}")
        meta_parts.append(f"Upscale type: {upscale_type}")
    if lora_specs:
        meta_parts.append(f"LoRA: {'; '.join(lora_specs)}")
    if pixel_upscaler:
        meta_parts.append(f"Pixel upscaler: {pixel_upscaler}")
    if sigmas:
        meta_parts.append(f"Sigmas: {'/'.join(f'{s:g}' for s in sigmas)}")
    parts.append(", ".join(meta_parts))
    pnginfo.add_text("parameters", "\n".join(parts))

    return pnginfo


def build_upscale_pnginfo(
    image: "PIL.Image.Image",
    upscaler_model: str,
    upscale_factor: float,
) -> PngInfo:
    """Build a PngInfo carrying over pre-existing text chunks + upscale metadata.

    Copies the text chunks already present on ``image`` (tEXt, zTXt, iTXt) into a
    fresh :class:`PngInfo`, then adds an ``upscale_data`` JSON chunk. An existing
    ``upscale_data`` chunk is replaced rather than copied, so repeated upscales do
    not accumulate records.
    """
    pnginfo = PngInfo()

    for key, value in getattr(image, "info", {}).items():
        if not isinstance(key, str):
            continue
        if key == "upscale_data":
            continue
        if isinstance(value, str) or isinstance(value, iTXt):
            pnginfo.add_text(key, value)

    pnginfo.add_text(
        "upscale_data",
        json.dumps({
            "upscaler_model": upscaler_model,
            "upscale_factor": upscale_factor,
        }),
    )

    return pnginfo
