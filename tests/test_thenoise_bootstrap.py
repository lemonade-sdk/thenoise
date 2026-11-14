"""GPU-free regression tests for the source checkout's thenoise.sh bootstrap."""

from __future__ import annotations

import os
import pty
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "thenoise.sh"

FAKE_UV = r"""#!/usr/bin/env python3
import os
from pathlib import Path
import subprocess
import sys


def wheel(python, backend, broken_tv=False):
    lib = Path(subprocess.check_output(
        [python, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        text=True,
    ).strip())
    hip = repr("10.1" if backend == "rocm" else None)
    cuda = repr("13.2" if backend == "cuda" else None)
    torch_version = {
        "rocm": "2.14.0+rocm10.1.0rc2",
        "cuda": "2.14.0+cu132",
        "cpu": "2.15.0",
    }[backend]
    tv_version = {
        "rocm": "0.29.0a0+rocm10.1.0rc2",
        "cuda": "0.29.0+cu132",
        "cpu": "0.30.0",
    }[backend]
    for name, version, body in (
        ("torch", torch_version, f'''
import os
from types import SimpleNamespace
__version__ = "{torch_version}"
version = SimpleNamespace(hip={hip}, cuda={cuda})
cuda = SimpleNamespace(is_available=lambda: os.environ.get("FAKE_GPU_AVAILABLE", "yes") == "yes")
'''),
        ("torchvision", tv_version, (
            "raise RuntimeError('operator torchvision::nms does not exist')\n"
            if broken_tv else f'__version__ = "{tv_version}"\n'
        )),
    ):
        d = lib / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "__init__.py").write_text(body)
        for old in lib.glob(f"{name}-*.dist-info"):
            import shutil
            shutil.rmtree(old)
        info = lib / (name + "-" + version + ".dist-info")
        info.mkdir(exist_ok=True)
        (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")


args = sys.argv[1:]
with open(os.environ["FAKE_UV_LOG"], "a") as out:
    out.write(" ".join(args) + "\n")

if args[0] == "venv":
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", args[1]], check=True)
elif args[:2] == ["pip", "install"]:
    python = args[args.index("--python") + 1]
    if any(arg.startswith("torch[device-") for arg in args):
        wheel(python, "rocm")
    elif "https://download.pytorch.org/whl/cu132" in args:
        wheel(python, "cuda")
    elif "-e" in args and os.environ.get("FAKE_UV_CORRUPT_AFTER") == "yes":
        wheel(python, "cpu")
    if "-e" in args and os.environ.get("FAKE_UV_CONFLICT") == "yes":
        print("forced dependency conflict", file=sys.stderr)
        sys.exit(42)
else:
    raise ValueError(args)
"""


def wheel_in(venv: Path, backend: str, broken_tv: bool = False):
    """Preinstall a fake torch variant in a venv; no PyTorch wheel is fetched."""
    code = FAKE_UV.split("args = sys.argv[1:]")[0]
    scope: dict = {"__name__": "mock_uv_wheel"}
    exec(code, scope)
    scope["wheel"](str(venv / "bin/python"), backend, broken_tv=broken_tv)


@pytest.fixture
def sandbox(tmp_path: Path):
    project = tmp_path / "thenoise"
    project.mkdir()
    (project / "thenoise.sh").write_bytes(SCRIPT.read_bytes())
    package = project / "thenoise"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "__main__.py").write_text("print('launched')\n")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    uv = fake_bin / "uv"
    uv.write_text(FAKE_UV)
    uv.chmod(0o755)
    log = tmp_path / "uv.log"
    env = os.environ.copy()
    env.pop("GFX_ARCH", None)
    env.pop("THENOISE_BACKEND", None)
    env.update({
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FAKE_UV_LOG": str(log),
        "FAKE_GPU_AVAILABLE": "yes",
    })

    def run(**overrides):
        actual_env = {**env, **overrides}
        return subprocess.run(
            ["bash", str(project / "thenoise.sh"), "--help"],
            cwd=project,
            env=actual_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=50,
        )

    def make_venv():
        venv = project / ".venv"
        subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True)
        return venv

    return project, fake_bin, log, env, run, make_venv


def check_run(result):
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "launched" in result.stdout


def lines(log: Path):
    return log.read_text().splitlines()


def assert_targeted(log: Path, project: Path):
    for line in lines(log):
        if line.startswith("pip install"):
            assert f"--python {project}/.venv/bin/python" in line


def test_fresh_rocm_keeps_amd_wheels(sandbox):
    project, _, log, _, run, _ = sandbox
    result = run(GFX_ARCH="gfx1151")
    check_run(result)
    assert any("torch[device-gfx1151]==2.14.0+rocm10.1.0rc2" in x for x in lines(log))
    assert any("torch==2.14.0+rocm10.1.0rc2" in x and "-e" in x for x in lines(log))
    assert_targeted(log, project)


def test_second_run_does_not_reinstall_torch(sandbox):
    _, _, log, _, run, _ = sandbox
    check_run(run(GFX_ARCH="gfx1151"))
    log.write_text("")
    check_run(run(GFX_ARCH="gfx1151"))
    assert len(lines(log)) == 1
    assert "-e" in lines(log)[0]


def test_fresh_nvidia_uses_cuda_wheels(sandbox):
    project, fake_bin, log, _, run, _ = sandbox
    nvidia_smi = fake_bin / "nvidia-smi"
    nvidia_smi.write_text("#!/bin/sh\necho 'GPU 0: Mock NVIDIA GPU'\n")
    nvidia_smi.chmod(0o755)
    check_run(run(GFX_ARCH=""))
    assert any("https://download.pytorch.org/whl/cu132" in x for x in lines(log))
    assert_targeted(log, project)


def test_preinstalled_cuda_remains_untouched(sandbox):
    project, _, log, _, run, make_venv = sandbox
    wheel_in(make_venv(), "cuda")
    check_run(run(GFX_ARCH=""))
    assert len(lines(log)) == 1
    assert "torch==2.14.0+cu132" in lines(log)[0]
    assert "torchvision==0.29.0+cu132" in lines(log)[0]
    assert_targeted(log, project)


def test_preinstalled_rocm_remains_untouched(sandbox):
    _, _, log, _, run, make_venv = sandbox
    wheel_in(make_venv(), "rocm")
    check_run(run(GFX_ARCH="gfx1151"))
    assert len(lines(log)) == 1


def test_preinstalled_cpu_is_repaired(sandbox):
    _, _, log, _, run, make_venv = sandbox
    wheel_in(make_venv(), "cpu")
    check_run(run(GFX_ARCH="gfx1151"))
    assert any("torch[device-gfx1151]" in x for x in lines(log))


def test_wrong_cuda_on_amd_is_repaired(sandbox):
    _, _, log, _, run, make_venv = sandbox
    wheel_in(make_venv(), "cuda")
    check_run(run(GFX_ARCH="gfx1151"))
    assert any("torch[device-gfx1151]" in x for x in lines(log))


def test_cuda_with_no_gpu_on_amd_is_repaired(sandbox):
    _, _, log, _, run, make_venv = sandbox
    wheel_in(make_venv(), "cuda")
    result = run(GFX_ARCH="gfx1151", FAKE_GPU_AVAILABLE="no")
    assert result.returncode != 0
    assert any("torch[device-gfx1151]" in x for x in lines(log))
    assert "cannot access a GPU" in result.stderr


def test_rocm_with_no_gpu_fails_without_reinstall(sandbox):
    _, _, log, _, run, make_venv = sandbox
    wheel_in(make_venv(), "rocm")
    result = run(GFX_ARCH="gfx1151", FAKE_GPU_AVAILABLE="no")
    assert result.returncode != 0
    assert "Check GPU drivers" in result.stderr
    assert not log.exists() or not lines(log)


def test_broken_torchvision_is_repaired(sandbox):
    _, _, log, _, run, make_venv = sandbox
    wheel_in(make_venv(), "rocm", broken_tv=True)
    check_run(run(GFX_ARCH="gfx1151"))
    assert any("torchvision[device-gfx1151]" in x and "--reinstall-package torchvision" in x for x in lines(log))


def test_foreign_venv_and_pythonpath_do_not_leak(sandbox):
    project, _, log, _, run, _ = sandbox
    foreign = project.parent / "foreign"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(foreign)], check=True)
    outside = project.parent / "outside"
    outside.mkdir()
    (outside / "torch.py").write_text("raise RuntimeError('foreign torch imported')\n")
    result = run(GFX_ARCH="gfx1151", VIRTUAL_ENV=str(foreign), PYTHONPATH=str(outside))
    check_run(result)
    assert_targeted(log, project)
    assert not list(foreign.glob("lib/python*/site-packages/torch"))


def test_fails_if_project_install_changes_gpu_wheel(sandbox):
    _, _, log, _, run, _ = sandbox
    result = run(GFX_ARCH="gfx1151", FAKE_UV_CORRUPT_AFTER="yes")
    assert result.returncode != 0
    assert "broken by the project dependency install" in result.stderr


def test_does_not_run_on_dependency_conflict(sandbox):
    _, _, _, _, run, _ = sandbox
    result = run(GFX_ARCH="gfx1151", FAKE_UV_CONFLICT="yes")
    assert result.returncode == 42
    assert "launched" not in result.stdout


def add_mock_nvidia(fake_bin: Path):
    nvidia_smi = fake_bin / "nvidia-smi"
    nvidia_smi.write_text("#!/bin/sh\necho 'GPU 0: Mock NVIDIA GPU'\n")
    nvidia_smi.chmod(0o755)


def run_with_tty(project: Path, env: dict[str, str], reply: str):
    """Send a real terminal answer to the bootstrap prompt (no GPU needed)."""
    master, slave = pty.openpty()
    try:
        proc = subprocess.Popen(
            ["bash", str(project / "thenoise.sh"), "--help"],
            cwd=project, env=env, stdin=slave, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True,
        )
        os.close(slave)
        slave = -1
        os.write(master, (reply + "\n").encode())
        stdout, stderr = proc.communicate(timeout=50)
        return subprocess.CompletedProcess(proc.args, proc.returncode, stdout, stderr)
    finally:
        os.close(master)
        if slave >= 0:
            os.close(slave)


def test_dual_gpu_noninteractive_defaults_rocm(sandbox):
    _, fake_bin, log, _, run, _ = sandbox
    add_mock_nvidia(fake_bin)
    result = run(GFX_ARCH="gfx1151")
    check_run(result)
    assert any("torch[device-gfx1151]" in x for x in lines(log))
    assert "defaulting to ROCm" in result.stderr


def test_dual_gpu_interactive_can_choose_cuda(sandbox):
    project, fake_bin, log, env, run, _ = sandbox
    add_mock_nvidia(fake_bin)
    result = run_with_tty(project, {**env, "GFX_ARCH": "gfx1151"}, "y")
    check_run(result)
    assert "Install NVIDIA CUDA instead?" in result.stderr
    assert any("https://download.pytorch.org/whl/cu132" in x for x in lines(log))
    log.write_text("")
    check_run(run(GFX_ARCH="gfx1151"))
    assert len(lines(log)) == 1  # does not prompt/switch or reinstall torch on a later launch


def test_dual_gpu_interactive_default_no(sandbox):
    project, fake_bin, log, env, _, _ = sandbox
    add_mock_nvidia(fake_bin)
    check_run(run_with_tty(project, {**env, "GFX_ARCH": "gfx1151"}, ""))
    assert any("torch[device-gfx1151]" in x for x in lines(log))


def test_explicit_cuda_backend_noninteractive(sandbox):
    _, fake_bin, log, _, run, _ = sandbox
    add_mock_nvidia(fake_bin)
    check_run(run(GFX_ARCH="gfx1151", THENOISE_BACKEND="cuda"))
    assert any("https://download.pytorch.org/whl/cu132" in x for x in lines(log))


def test_explicit_backend_switches_installed_rocm_to_cuda(sandbox):
    _, fake_bin, log, _, run, make_venv = sandbox
    add_mock_nvidia(fake_bin)
    wheel_in(make_venv(), "rocm")
    check_run(run(GFX_ARCH="gfx1151", THENOISE_BACKEND="cuda"))
    assert any("https://download.pytorch.org/whl/cu132" in x for x in lines(log))


def test_invalid_backend_fails_early(sandbox):
    _, _, log, _, run, _ = sandbox
    result = run(THENOISE_BACKEND="invalid")
    assert result.returncode == 2
    assert "THENOISE_BACKEND must be auto, rocm, or cuda" in result.stderr
    assert not log.exists()
