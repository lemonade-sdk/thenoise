"""Model-specific VAE components.

Each type is its own module, exported here for the models to pick. Loading one is
always ``load_<name>_vae(path, device, dtype)``; where a model accepts more than
one codec, ``load_<name>_family_vae`` reads the checkpoint header and dispatches
to the right one. Variants of ONE codec — the Flux.2 AE's full or small decoder —
are read off the weights inside that codec's own loader, so they need no choice.

A codec runs in its own ``COMPUTE_DTYPE`` whatever the DiT beside it runs in -- FP16,
except the Wan 2.2 family, whose activations leave that range. Its loader casts to that
dtype while reading the file, so the file's own format decides nothing; passing
``dtype`` overrides the codec and is not a request for the file's.
"""
from .qwen_image import (
    AutoencoderKLQwenImage,
    load_ming_vae,
    load_qwen_family_vae,
    load_qwen_vae,
)
from .flux import AutoencoderKLFlux, load_flux_vae
from .flux2 import AutoencoderKLFlux2, load_flux2_vae
from .mage_flow import AutoencoderKLMageFlow, load_mage_family_vae, load_mage_vae
from .wan22 import AutoencoderKLWan22, load_wan22_vae

__all__ = [
    "AutoencoderKLQwenImage",
    "load_ming_vae",
    "load_qwen_family_vae",
    "load_qwen_vae",
    "AutoencoderKLFlux",
    "load_flux_vae",
    "AutoencoderKLFlux2",
    "load_flux2_vae",
    "AutoencoderKLMageFlow",
    "load_mage_family_vae",
    "load_mage_vae",
    "AutoencoderKLWan22",
    "load_wan22_vae",
]
