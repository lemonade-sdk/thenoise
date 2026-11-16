"""LTX downloader manifest and cache checks without network access."""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace


def _module(monkeypatch, download):
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    path = Path(__file__).resolve().parents[1] / "scripts/download.py"
    spec = importlib.util.spec_from_file_location("_ltx_downloader_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_ltx_manifests_select_correct_weights_and_sources(monkeypatch):
    m = _module(monkeypatch, lambda *args, **kwargs: None)
    ltx25 = m.MODELS["ltx25"]
    assert ltx25.default_variant == "int8-convrot"
    jobs = ltx25.jobs(SimpleNamespace())
    assert any("distilled-transformer-comfy-int8-convrot" in p for _, _, p in jobs)
    assert any("gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot" in p for _, _, p in jobs)
    assert len(jobs) == 5

    ltx23 = m.MODELS["ltx23"]
    assert ltx23.default_variant == "int8-convrot"
    jobs = ltx23.jobs(SimpleNamespace())
    assert any(repo == "milo01/ComfyUI-Native-INT8_ConvRot" for _, repo, _ in jobs)
    assert ("vae", "Kijai/LTX2.3_comfy", "vae/LTX23_video_vae_bf16.safetensors") in jobs
    assert ("audio_vae", "Kijai/LTX2.3_comfy", "vae/LTX23_audio_vae_bf16.safetensors") in jobs
    assert ("text_projection", "Kijai/LTX2.3_comfy",
            "text_encoders/ltx-2.3_text_projection_bf16.safetensors") in jobs
    assert any(repo == "Stick9190/gemma-3-12b-it-int8-convrot"
               and path == "gemma_3_12B_it_int8_convrot.safetensors"
               for _, repo, path in jobs)
    assets = [path for _, repo, path in jobs
              if repo == "Lightricks/gemma-3-12b-it-qat-q4_0-unquantized"]
    assert {"config.json", "tokenizer.json", "tokenizer_config.json"} <= set(assets)
    assert not any(path.startswith("model-") and path.endswith(".safetensors")
                   for path in assets)

    for key in ("ltx23", "ltx25"):
        bf16 = m.MODELS[key].jobs(SimpleNamespace(variant="bf16"))
        assert any(".safetensors" in path for _, _, path in bf16)
        assert all("int8" not in path.lower() for _, _, path in bf16)


def test_ltx_dry_run_and_cached_files_do_not_download(monkeypatch, capsys, tmp_path):
    calls = []

    def fake_download(_repo, filename, *, local_dir):
        calls.append(filename)
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake")
        return str(path)

    module = _module(monkeypatch, fake_download)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["download.py", "--model", "ltx25", "--dry-run"])
    module.main()
    assert not calls
    assert "missing" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", ["download.py", "--model", "ltx25"])
    module.main()
    assert len(calls) == 5
    module.main()
    assert len(calls) == 5
