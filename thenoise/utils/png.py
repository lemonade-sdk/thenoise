"""PNG metadata helpers for generation images."""
from __future__ import annotations

import json
from typing import List, Optional

from PIL.PngImagePlugin import PngInfo


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
) -> PngInfo:
    """Build a PngInfo object with generation metadata (JSON + human-readable).

    Writes two text chunks:
      * ``generation_data`` — full JSON of all resolved parameters
      * ``parameters`` — human-readable text for older-software compatibility
    """
    pnginfo = PngInfo()

    # JSON metadata (all resolved parameters + model name)
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
    })
    pnginfo.add_text("generation_data", gen_data)

    # Human-readable "parameters" text (A1111 compatibility)
    # Format: prompt lines, optional "Negative prompt: " line,
    # then a single last line of comma-separated "Key: value" pairs.
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
    parts.append(", ".join(meta_parts))
    pnginfo.add_text("parameters", "\n".join(parts))

    return pnginfo
