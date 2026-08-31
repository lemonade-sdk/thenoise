# SDXL model-loading utilities.
#
# The splitter produces four files from the combined single-file checkpoint:
#   dit/         bare LDM UNet keys (input_blocks.*, middle_block.*, ...)
#   vae/         bare SDXL VAE keys (post_quant_conv.*, decoder.*)
#   clip_l/      bare CLIP-L keys (text_model.*)
#   clip_g/      bare CLIP-G keys (token_embedding.*, transformer.*, ...)
#
# The CLIP tokenizer config files are vendored under ``configs/tokenizer/`` so
# the tokenizer loads offline without fetching from the Hub.

import logging
import os
from typing import Optional, Union

import torch

from thenoise.dit.sdxl.models import SdxlUNet
from thenoise.utils.safetensors import load_split_weights, strip_wrap_prefixes

logger = logging.getLogger(__name__)

#: Vendored CLIP tokenizer config directory (vocab.json, merges.txt, ...).
SDXL_TOKENIZER_CONFIG_DIR = os.path.join(
    os.path.dirname(__file__), "configs", "tokenizer"
)

#: Prediction-type marker tensors (``v_pred``, ``edm_mean``/``edm_std``, ...) that
#: some SDXL checkpoints carry at the top level. They are NOT UNet weights, so
#: they are dropped before ``load_state_dict``. Mirrors ComfyUI's ``SDXL.model_type``.
PREDICTION_MARKERS = frozenset(
    {
        "v_pred",
        "ztsnr",
        "edm_mean",
        "edm_std",
        "edm_vpred.sigma_max",
        "edm_vpred.sigma_min",
    }
)


def build_sdxl_dit(
    unet_sd: dict,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> SdxlUNet:
    """Build the SDXL UNet on meta and load a bare ``input_blocks.*`` state dict.

    ``unet_sd`` is the UNet weights with wrapper prefixes already stripped (e.g.
    ``model.diffusion_model.``); it may also carry the prediction-type markers
    (``v_pred``, ...), which are dropped before loading.
    """
    with torch.device("meta"):
        dit = SdxlUNet()
    dit.load_state_dict(
        {k: v for k, v in unet_sd.items() if k not in PREDICTION_MARKERS},
        strict=True,
        assign=True,
    )
    dit.to(dtype).to(device)
    return dit


def load_sdxl_dit(
    dit_path: str,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    loading_device: Optional[Union[str, torch.device]] = None,
) -> SdxlUNet:
    """Build the SDXL UNet from a split ``--dit`` safetensors file."""
    device = torch.device(device)
    loading_device = device if loading_device is None else torch.device(loading_device)

    logger.info("Loading SDXL UNet weights from %s", dit_path)
    sd = load_split_weights(dit_path, device=str(loading_device), disable_mmap=True, dtype=dtype)
    sd = strip_wrap_prefixes(sd)
    return build_sdxl_dit(sd, device, dtype)


def find_sdxl_tokenizer_dir(text_encoder_path: str, max_depth: int = 4) -> Optional[str]:
    """Locate a local CLIP ``tokenizer/`` directory near a text encoder file.

    The downloader drops the tokenizer under ``<out>/tokenizer/`` while the
    text encoders land under ``<out>/split_files/text_encoders/``. Search
    ``max_depth`` parents so the tokenizer is loaded offline when present.
    Returns ``None`` to fall back to the vendored ``configs/tokenizer/`` dir.
    """
    base = os.path.dirname(os.path.abspath(text_encoder_path))
    for _ in range(max_depth):
        cand = os.path.join(base, "tokenizer")
        if os.path.isdir(cand):
            return cand
        parent = os.path.dirname(base)
        if parent == base:
            break
        base = parent
    return None


def load_sdxl_tokenizer(tokenizer_dir: Optional[str] = None):
    """Load the shared CLIP tokenizer (local ``tokenizer_dir`` or vendored)."""
    from transformers import CLIPTokenizer

    tokenizer_dir = tokenizer_dir or SDXL_TOKENIZER_CONFIG_DIR
    if not os.path.isdir(tokenizer_dir):
        raise FileNotFoundError(
            f"CLIP tokenizer config directory not found at {tokenizer_dir}. "
            "Expected vocab.json, merges.txt and tokenizer_config.json."
        )
    return CLIPTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)


def build_sdxl_text_encoders(
    clip_l_sd: dict,
    clip_g_sd: dict,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.bfloat16,
):
    """Build both CLIP text encoders from bare CLIP-L / CLIP-G state dicts.

    ``clip_l_sd`` is the transformers CLIP-L weights (``text_model.*``); ``clip_g_sd``
    is the OpenCLIP CLIP-G weights (``token_embedding.*``, ``transformer.*``, ...).
    Returns ``(clip_l, clip_g)``.
    """
    from .text import build_clip_g, build_clip_l

    device = torch.device(device)
    if not clip_l_sd:
        raise ValueError("text encoder state dict has no CLIP-L weights")
    if not clip_g_sd:
        raise ValueError("text encoder state dict has no CLIP-G weights")
    return build_clip_l(clip_l_sd, device=device, dtype=dtype), build_clip_g(
        clip_g_sd, device=device, dtype=dtype
    )


def load_sdxl_text_encoders(
    text_encoder_path: str,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.bfloat16,
):
    """Load both CLIP text encoders from the splitter's combined file.

    The combined file keys are prefixed ``clip_l.`` and ``clip_g.``. Loads
    CLIP-L (transformers) and CLIP-G (OpenCLIP) and returns ``(clip_l, clip_g)``.
    """
    from thenoise.utils.safetensors import load_safetensors

    device = torch.device(device)
    sd = load_safetensors(text_encoder_path, device=str(device), disable_mmap=True, dtype=dtype)

    clip_l_sd = {k[len("clip_l."):]: v for k, v in sd.items() if k.startswith("clip_l.")}
    clip_g_sd = {k[len("clip_g."):]: v for k, v in sd.items() if k.startswith("clip_g.")}
    return build_sdxl_text_encoders(clip_l_sd, clip_g_sd, device=device, dtype=dtype)
