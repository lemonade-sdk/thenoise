"""PNG metadata helpers for the images the engine writes.

Controllers attach a :class:`PngInfo` to their result; the writers here keep it on disk.
"""
from __future__ import annotations

import json
from typing import List, Optional

from PIL.PngImagePlugin import PngInfo, iTXt

from thenoise.utils.paths import ensure_png_extension


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
    ref_method: Optional[str] = None,
    kv_cache: Optional[bool] = None,
) -> PngInfo:
    """A ``generation_data`` JSON chunk of every resolved parameter, plus a
    human-readable ``parameters`` chunk (A1111-compatible: prompt lines then pairs).
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
        "ref_method": ref_method,
        "kv_cache": kv_cache,
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


def _provenance(image, keyword: str, data: dict) -> PngInfo:
    """A :class:`PngInfo` of ``image``'s own text chunks plus ``data`` under ``keyword``.
    The chunk about to be written is dropped, so repeated operations replace it.
    """
    pnginfo = PngInfo()
    for key, value in getattr(image, "info", {}).items():
        if not isinstance(key, str) or key == keyword:
            continue
        if isinstance(value, str) or isinstance(value, iTXt):
            pnginfo.add_text(key, value)
    pnginfo.add_text(keyword, json.dumps(data))
    return pnginfo


def pnginfo_of(image) -> Optional[PngInfo]:
    """The provenance a controller attached to ``image``, or ``None`` if it attached none."""
    return getattr(image, "_pnginfo", None)


def save_output(image, path: str) -> str:
    """Write a controller's image, keeping the chunks it attached; a name with no
    extension becomes ``.png``, and the path written comes back for the caller to log.
    """
    path = ensure_png_extension(path)
    image.save(path, pnginfo=pnginfo_of(image))
    return path


def build_upscale_pnginfo(
    image: "PIL.Image.Image",
    upscaler_model: str,
    upscale_factor: float,
) -> PngInfo:
    """The source image's chunks plus this upscale's ``upscale_data`` record."""
    return _provenance(image, "upscale_data", {
        "upscaler_model": upscaler_model,
        "upscale_factor": upscale_factor,
    })


def build_controlmap_pnginfo(
    source: "PIL.Image.Image",
    control_type: str,
    model: str,
    options: dict,
) -> PngInfo:
    """The source's chunks plus this extraction's ``controlmap_data`` record. The map
    is its input's size, so the PNG header already states the geometry.
    """
    return _provenance(source, "controlmap_data", {
        "type": control_type,
        "model": model,
        "options": options,
    })
