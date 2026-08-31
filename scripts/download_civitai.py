"""Download checkpoints from Civitai via ``civitapy``

thenoise can load models from separate ``--dit`` / ``--vae`` / ``--text-encoder``
files, or from a unified ``.safetensors`` file in Stability-AI SDXL layout. This
script downloads a model by its Civitai model ID using the ``civitapy`` package,
then splits the combined checkpoint into the three parts, writing them alongside
the downloaded file:

  <out>/Checkpoint/<modelid>_<name>_<creator>/<basemodel>/
    <combined>.safetensors                          (--checkpoint)
    diffusion_models/{name}_unet.safetensors        (--dit)
    vae/{name}_vae.safetensors                      (--vae)
    text_encoders/{name}_clip_l_g.safetensors       (--text-encoder)

Downloads are scoped by base-model filter so only versions whose base model
thenoise can actually run are fetched — unrelated checkpoints and LoRA/VAE-only  # FIXME LoRA downloads are allowed now
versions are skipped. By default every base model thenoise supports is allowed;
use ``--base`` (repeatable) to restrict to specific ones.

Usage:
    python scripts/download_civitai.py 1331249 --out ./models/bubbli
    python scripts/download_civitai.py 1331249 5678 --base anima --base krea --out ./models/foo
    python scripts/download_civitai.py 1331249 --no-base-model-filter --keep-combined
    python scripts/download_civitai.py "https://civitai.com/models/1302719/bla-bla-bla?modelVersionId=1591915"
    python scripts/download_civitai.py "1302719/bla-bla-bla?modelVersionId=1591915"
    python scripts/download_civitai.py --preset anima --preset hyper --out ./models/civitai
    python scripts/download_civitai.py --preset anima --preset zimage --no-download

Model targets may be a bare model ID, a full Civitai URL, or a bare Civitai
model path with the ``https://civitai.com/models/`` prefix omitted
(``<id>/<slug>``). A URL/path with ``?modelVersionId=<id>`` downloads only that
specific version. ``--preset`` downloads a curated set of models (see PRESETS)
by name; it may be repeated and combined with positional targets. Pass
``--no-download`` to resolve presets/targets and print the download plan
without actually fetching anything.
"""

from __future__ import annotations

import argparse
import logging
import urllib.parse
from pathlib import Path
from typing import Sequence

import anyio
from safetensors.torch import load_file, save_file

try:
    from civitapy import CivitAIClient, CivitAIError, Model, ModelVersion
except ImportError:
    raise SystemExit(
        "civitapy is required to download from Civitai.\n"
        "Install it with:  uv pip install civitapy"
    )

logger = logging.getLogger(__name__)

#: Civitai URL prefix. Presets store bare model paths (``<id>/<slug>?modelVersionId=<id>``)
#: without this prefix; :func:`parse_target` and the ``--preset`` resolver add it back.
CIVITAI_MODELS = "https://civitai.com/models/"

#: Some download preset. Each maps a short name to a dict whose keys
#: are the artifact kind:
#:
#:   - single-file kinds (``checkpoint``, ``lora``, ``upscaler``) hold one
#:     Civitai model path each;
#:   - a split model is described by ``dit``, ``vae`` and ``text-encoder``
#:     paths, each a separate Civitai model.
#:
#: These were selected solely because of popularity on Civitai; if a lot of
#: people report good results with these models they'll probably be good for
#: getting started. You can, of course, use this script to download whatever
#: other models you want.
PRESETS: dict[str, dict[str, str]] = {
# SDXL (and derivative) AIO Checkpoints
    "illustrious-base": {"checkpoint": "1369089/illustrious-xl-20?modelVersionId=1546777"},
    "illustrious-wai": {"checkpoint": "827184/wai-illustrious-sdxl?modelVersionId=2883731"},
    "illustrious-anime": {"checkpoint": "376130/nova-anime-xl?modelVersionId=2940478"},
    "illustrious-3dcg": {"checkpoint": "715287/nova-3dcg-xl?modelVersionId=2744564"},
    "illustrious-goddess": { "checkpoint": "1515023/illustrious-xl-20-goddessmix?modelVersionId=1733909" },
    "illustrious-photo": { "checkpoint": "974693/realism-illustrious-by-stable-yogi?modelVersionId=2831979" },
    "illustrious-bubbli": {"checkpoint": "1331249/bubbli-cartoon-il?modelVersionId=1503014"},
    "sdxl-animagine": {"checkpoint": "260267/animagine-xl-v31?modelVersionId=403131 " },
    "sdxl-base": {"checkpoint": "101055/sd-xl?modelVersionId=128078"},
    "sdxl-cyberreal": {"checkpoint": "312530/cyberrealistic-xl?modelVersionId=2840768"},
    "sdxl-epicreal": {"checkpoint": "277058/epicrealism-xl?modelVersionId=2514955" },
    "sdxl-juggernaut": {"checkpoint": "133005/juggernaut-xl?modelVersionId=1759168"},
    "sdxl-realviz": {"checkpoint": "139562/realvisxl-v50?modelVersionId=789646" },
    "pony-animerge": {"checkpoint": "613147/animerge-pony-xl?modelVersionId=1762147"},
    "pony-base": {"checkpoint": "257749/pony-diffusion-v6-xl?modelVersionId=290640"},
    "pony-cyber": {"checkpoint": "443821/cyberrealistic-pony?modelVersionId=2884631"},
    "pony-real": {"checkpoint": "372465/pony-realism?modelVersionId=914390"},
    "pony-xl": {"checkpoint": "439889/prefect-pony-xl?modelVersionId=2114187"},
    "noobai-nova": {"checkpoint": "376130/nova-anime-xl?modelVersionId=1500882"},
    "noobai-obsession": {"checkpoint": "1318945/one-obsession?modelVersionId=2044887"},
    "noobai-wai": {"checkpoint": "989367/wai-shuffle-noob?modelVersionId=2444683"},
    "noobai-xl": {"checkpoint": "833294/noobai-xl-nai-xl?modelVersionId=1190596"},

    # "zit-aio": {"checkpoint": "2259646/z-image-turbo-anime?modelVersionId=2543657"}, #not supported yet ;)
# Regular 3-file models
    "anima": {"dit": "2458426/anima?modelVersionId=3263843"},
    "anima-semireal": {
        "dit": "2668799/cyberrealistic-anima?modelVersionId=3136380"
    },
    "anima-photo": {"dit": "2645333/photanima?modelVersionId=3112450"},
    "zimage": {
        "dit": "2342797/z-image-base?modelVersionId=2635223",
        "vae": "2740928/flux1-ae?modelVersionId=3082494",
        "text_encoder": "2742977/qwen3?modelVersionId=3085020",
    },
    "zit": {"dit": "2168935/z-image-turbo?modelVersionId=2442439"},
    "zit-smol": { "dit": "2169712/z-image-turbo-quantized-for-low-vram?modelVersionId=2549032" },
    "krea2": { "dit": "2726029/krea-2-turbo-official-comfy-org-checkpoints-krea2?modelVersionId=3064584" },
    "krea2-int8": { "dit": "2726029/krea-2-turbo-official-comfy-org-checkpoints-krea2?modelVersionId=3091481" },

    "klein4b": {"dit": "2322332/flux2-klein?modelVersionId=2612557"},
    "klein9b": {"dit": "2322332/flux2-klein?modelVersionId=2612554"},
    "klein4b-fp8": {"dit": "2311742/flux-klein-fp8?modelVersionId=2600878"},
    "klein9b-fp8": {"dit": "2311742/flux-klein-fp8?modelVersionId=2606187"},
# Extra stuff
    "anima-turbo": {"lora": "2560840/anima-turbo-lora?modelVersionId=2979642"},
    "hyper-sd": {"lora": "800496/lightninghyper-8step"},
    "esrgan": {"upscaler": "147817/realesrganx4plus"},
    "remacri": {"upscaler": "147759/remacri"},
    "sdxl-vae": {"vae": "https://civitai.red/models/296576/sdxl-vae"},
}


def parse_target(s: str) -> tuple[int, int | None]:
    """Parse a model ID or Civitai URL into ``(model_id, version_id_or_None)``.

    Accepts a bare model ID (``1331249``), a full model URL
    (``https://civitai.com/models/<id>/<slug>``), a bare Civitai model path with
    the ``https://civitai.com/models/`` prefix omitted
    (``<id>/<slug>``), and any of these pinning one version
    (``...?modelVersionId=<id>``), on any Civitai domain (``civitai.com``,
    ``civitai.red``, ...).
    """
    try:
        return int(s), None
    except ValueError:
        pass
    split = urllib.parse.urlsplit(s)
    parts = [p for p in split.path.split("/") if p]
    # Drop a leading "models/" component from full URLs so bare model paths
    # (starting at the numeric ID) resolve the same way.
    if parts and parts[0] == "models":
        parts.pop(0)
    if not parts or not parts[0].isdigit():
        raise ValueError(f"not a Civitai model ID or path: {s!r}")
    model_id = int(parts[0])
    version_id = None
    qs = urllib.parse.parse_qs(split.query)
    if "modelVersionId" in qs:
        version_id = int(qs["modelVersionId"][0])
    return model_id, version_id


#: Stability-AI SDXL layout key prefixes inside a combined checkpoint.
UNET_PREFIX = "model.diffusion_model."
VAE_PREFIX = "first_stage_model."
CLIP_L_PREFIX = "conditioner.embedders.0.transformer."
CLIP_G_PREFIX = "conditioner.embedders.1.model."

#: Prediction-type marker tensors (``v_pred``, ``edm_mean``/``edm_std``, ...) that
#: some SDXL checkpoints carry at the top level. Preserved into the dit split so
#: ``SdxlModel`` can autodetect the prediction type.
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

#: Human-friendly ``--base`` choices mapped to the Civitai base-model strings
#: each one covers. Kept in this script (not derived from the model catalog);
#: extend it when thenoise gains support for another base model.
BASE_MODEL_CHOICES: dict[str, list[str]] = {
    "sdxl": ["Illustrious", "SDXL 1.0", "NoobAI", "Pony", "NoobAI"],
    "anima": ["Anima"],
    "zimage": ["ZImageTurbo", "Z-Image"],
    "flux-klein": ["Flux.2 Klein 4B", "Flux.2 Klein 9B", "Flux.2 Klein"],
    "krea": ["Krea 2", "Krea"],
}

#: Every base model thenoise supports (the default filter when ``--base`` is
#: unspecified).
DEFAULT_BASE_MODELS: list[str] = [
    base for bases in BASE_MODEL_CHOICES.values() for base in bases
]


def split_checkpoint(checkpoint: str, out: Path, name: str) -> tuple[Path, Path, Path]:
    """Split a combined SDXL/Illustrious checkpoint into thenoise parts.

    Loads ``checkpoint`` and partitions its state dict into a UNet (DiT), VAE,
    and a combined CLIP-L + CLIP-G text encoder file, saving them under ``out``
    with the given ``name`` stem:

      out/diffusion_models/{name}_unet.safetensors        (--dit)
      out/vae/{name}_vae.safetensors                      (--vae)
      out/text_encoders/{name}_clip_l_g.safetensors       (--text-encoder)

    Returns the (dit, vae, text_encoder) paths.

    Raises:
        ValueError: If any partition is empty (the file is not an SDXL/Illustrious
            combined checkpoint).
    """
    print(f"Loading {checkpoint} ...")
    sd = load_file(checkpoint)

    unet = {
        k[len(UNET_PREFIX) :]: v for k, v in sd.items() if k.startswith(UNET_PREFIX)
    }
    # Keep prediction-type marker tensors (``v_pred``, ``edm_mean``/``edm_std``,
    # ...) so ``SdxlModel`` can autodetect the prediction type from the dit file.
    unet.update({k: v for k, v in sd.items() if k in PREDICTION_MARKERS})
    vae = {
        k[len(VAE_PREFIX) :]: v
        for k, v in sd.items()
        if k.startswith(VAE_PREFIX)
        and (k.startswith(VAE_PREFIX + "decoder.") or "post_quant_conv" in k)
    }
    clip_l = {
        k[len(CLIP_L_PREFIX) :]: v
        for k, v in sd.items()
        if k.startswith(CLIP_L_PREFIX) and not k.endswith("position_ids")
    }
    clip_g = {
        k[len(CLIP_G_PREFIX) :]: v for k, v in sd.items() if k.startswith(CLIP_G_PREFIX)
    }
    for part_name, part in [
        ("unet", unet),
        ("vae", vae),
        ("clip_l", clip_l),
        ("clip_g", clip_g),
    ]:
        if not part:
            raise ValueError(
                f"partition {part_name!r} is empty: {checkpoint} may not be an "
                "SDXL/Illustrious combined checkpoint"
            )

    (out / "diffusion_models").mkdir(parents=True, exist_ok=True)
    (out / "vae").mkdir(parents=True, exist_ok=True)
    (out / "text_encoders").mkdir(parents=True, exist_ok=True)

    dit = out / "diffusion_models" / f"{name}_unet.safetensors"
    vae_path = out / "vae" / f"{name}_vae.safetensors"
    te = out / "text_encoders" / f"{name}_clip_l_g.safetensors"

    save_file(unet, str(dit))
    save_file(vae, str(vae_path))
    combined_te = {
        **{f"clip_l.{k}": v for k, v in clip_l.items()},
        **{f"clip_g.{k}": v for k, v in clip_g.items()},
    }
    save_file(combined_te, str(te))

    return dit, vae_path, te


def _split_paths(base: Path, name: str) -> tuple[Path, Path, Path]:
    """The three thenoise output paths for a split rooted at ``base``."""
    return (
        base / "diffusion_models" / f"{name}_unet.safetensors",
        base / "vae" / f"{name}_vae.safetensors",
        base / "text_encoders" / f"{name}_clip_l_g.safetensors",
    )


def _already_split(client: CivitAIClient, model_id: int, name: str) -> bool:
    """True if every version of ``model_id`` already has all three split files.

    Skips the download when the model was previously downloaded and split.
    """
    model = Model(**client.models_get(model_id))
    if model.type != "Checkpoint":
        return False
    for version in model.model_versions:
        base = Path(client._version_download_dir(model, version.base_model))
        if not all(p.exists() for p in _split_paths(base, name)):
            return False
    return True


def _already_split_version(client: CivitAIClient, version_id: int, name: str) -> bool:
    """True if a single version already has all three split files."""
    version = ModelVersion(**client.model_versions_get(version_id))
    model = Model(**client.models_get(version.model_id))
    base = Path(client._version_download_dir(model, version.base_model))
    return all(p.exists() for p in _split_paths(base, name))


def _split_downloaded(
    client: CivitAIClient, model_id: int, paths: Sequence[str], name: str
) -> None:
    """Split every downloaded combined checkpoint under ``paths``.

    Prints a thenoise block per split checkpoint. Non-checkpoint or non-SDXL
    files are left in place (with a message) rather than failing the download.
    """
    import os

    for path in sorted(map(Path, paths)):
        print(f"Downloaded: {path}")

    # Only split checkpoints; a model may also carry LoRA/VAE/config files that
    # don't fit the combined SDXL layout.
    model_type = client.models_get(model_id).get("type")
    if model_type != "Checkpoint":
        print(f"Model type is {model_type!r} — not a checkpoint, skipping split.")
        return

    results: list[tuple[Path, Path, Path]] = []
    for path in sorted(map(Path, paths)):
        # Split files live next to the downloaded checkpoint (its parent dir), so
        # multiple models never overwrite one another.
        split_paths = _split_paths(path.parent, name)
        if all(p.exists() for p in split_paths):
            print(f"  (already split, skipping: {path.parent})")
            results.append(split_paths)
            continue
        try:
            results.append(split_checkpoint(str(path), path.parent, name))
        except ValueError as e:
            # Not a combined SDXL/Illustrious checkpoint (e.g. a LoRA or VAE
            # file); leave it in place rather than failing the whole download.
            print(f"  (skipping split: {e})")
        except CivitAIError as e:
            print(f"  (download error: {e})")

    if not results:
        print("\nNo combined checkpoints were split.")
        return

    # A model may ship one combined checkpoint (single part) or several versions
    # / parts (multi part); print a separate thenoise block per split checkpoint.
    print(f"\nDone. Split {len(results)} checkpoint(s). Point thenoise at:")
    for i, (dit, vae, te) in enumerate(results, 1):
        if len(results) > 1:
            print(f"  [{i}]")
        print(f"  --dit            {os.path.relpath(dit)}")
        print(f"  --vae            {os.path.relpath(vae)}")
        print(f"  --text-encoder   {os.path.relpath(te)}")


def download_model(
    model_id: int,
    out: Path,
    *,
    base_models: Sequence[str] | None,
    keep_combined: bool,
    name: str,
    progress: bool,
    version_id: int | None = None,
    split: bool = True,
) -> None:
    import os

    if not os.environ.get("CIVITAI_TOKEN"):
        raise SystemExit(
            "CIVITAI_TOKEN is not set. Civitai requires a bearer token to download model files.\n"
            "Create an API key at https://civitai.com/account (Account -> API Keys), then set it:\n"
            "  export CIVITAI_TOKEN=<your-key>"
        )

    out.mkdir(parents=True, exist_ok=True)

    client = CivitAIClient(
        download_dir=str(out),
        base_models=list(base_models) if base_models else None,
    )
    if base_models:
        print(f"Base-model filter: {', '.join(base_models)}")
    else:
        print("Base-model filter: none (downloading every version)")

    if version_id is not None:
        if _already_split_version(client, version_id, name):
            print(
                f"Model {model_id} version {version_id} already downloaded and split — skipping."
            )
            return
        print(f"Downloading civitai model {model_id} version {version_id} ...")
        paths = client.download_model_version(version_id, progress=progress, )
        if not paths:
            print("Nothing downloaded — version does not match the base-model filter.")
            return
    else:
        if _already_split(client, model_id, name):
            print(f"Model {model_id} already downloaded and split — skipping.")
            return
        print(f"Downloading civitai model {model_id} ...")
        paths = client.download_model(model_id, progress=progress)
        if not paths:
            print("Nothing downloaded — no version matched the filter.")
            return

    if not split:
        for path in sorted(paths):
            print(f"Downloaded: {path}")
        print(f"\nDone. Point thenoise at: {paths[0]}")
        return

    _split_downloaded(client, model_id, paths, name)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    # httpx (used by civitapy) logs every connection/retry at INFO; keep that
    # noise out of the downloader's output.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    ap = argparse.ArgumentParser(
        description="Download + split Civitai checkpoints via civitapy"
    )
    ap.add_argument(
        "targets",
        nargs="*",
        help="Civitai model ID(s), full model URL(s), or bare model path(s) with "
        "the https://civitai.com/models/ prefix omitted; a URL/path with "
        "?modelVersionId=<id> downloads only that version",
    )
    ap.add_argument(
        "--preset",
        action="append",
        choices=sorted(PRESETS),
        help="download a curated model set by name (see PRESETS); repeatable "
        "and combinable with positional targets",
    )
    ap.add_argument("--out", default="./models/civitai", help="output directory")
    ap.add_argument(
        "--name",
        default="model",
        help="output filename stem (e.g. 'bubbli' -> bubbli_unet.safetensors)",
    )
    ap.add_argument(
        "--no-download",
        action="store_true",
        help="resolve presets/targets and print the download plan without "
        "actually downloading anything",
    )
    ap.add_argument(
        "--base",
        action="append",
        choices=sorted(BASE_MODEL_CHOICES),
        help="restrict downloads to one base model (repeatable; default: all supported)",
    )
    ap.add_argument(
        "--no-base-model-filter",
        action="store_true",
        help="disable the base-model filter and download every version",
    )
    ap.add_argument(
        "--keep-combined",
        action="store_true",
        help="keep the downloaded combined checkpoint after splitting",
    )
    ap.add_argument(
        "--no-progress",
        action="store_true",
        help="disable the per-file download progress bar",
    )
    args = ap.parse_args()

    if args.no_base_model_filter:
        base_models = None
    elif args.base:
        base_models = [base for name in args.base for base in BASE_MODEL_CHOICES[name]]
    else:
        base_models = DEFAULT_BASE_MODELS

    # Resolve positional targets and presets into a list of (kind, path) pairs.
    items: list[tuple[str, str]] = [(CHECKPOINT, t) for t in args.targets]
    for preset in args.preset or []:
        items.extend(PRESETS[preset].items())
    if not items:
        ap.error("no targets or --preset given")

    if args.no_download:
        for kind, path in items:
            model_id, version_id = parse_target(path)
            version = f" version {version_id}" if version_id else ""
            print(f"[{kind:14s}] model {model_id}{version}  {CIVITAI_MODELS}{path}")
        return

    try:
        for kind, path in items:
            model_id, version_id = parse_target(path)
            try:
                download_model(
                    model_id,
                    Path(args.out),
                    base_models=base_models,
                    keep_combined=args.keep_combined,
                    name=args.name,
                    progress=not args.no_progress,
                    version_id=version_id,
                    split=False,
                )
            except CivitAIError as e:
                print(f"{e}")
                pass
        print()
    except KeyboardInterrupt:
        raise SystemExit("\nInterrupted — aborting download.")
    except anyio.NoEventLoopError:
        # Ctrl-C tears down the async event loop used by civitapy/httpx; anyio
        # raises NoEventLoopError during that teardown. Treat it as an interrupt.
        raise SystemExit("\nInterrupted — aborting download.")


if __name__ == "__main__":
    main()
