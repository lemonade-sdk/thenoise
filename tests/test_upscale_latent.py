"""Latent-domain upscaler strategies (weight-free).

The pipeline drives ``LatentUpscaler`` through a mock (see ``conftest``), so what
the strategies themselves promise — the target they hand their network, the
canonical space they return, and the dtype they keep — is asserted here, with the
network faked and no weights loaded. The real committed weights per latent format
are exercised in ``test_catalog.py`` / ``test_zimage.py`` / ``test_flux_klein.py``.
"""
from __future__ import annotations

import pytest
import torch

from thenoise.upscale import LatentUpscaler, SesquiLSRUpscaler
from thenoise.upscale import sesqui as sesqui_mod
from thenoise.upscale.inference_adaptors import LatentFormatAdaptor, make_flux2, make_wan21


class _RecordingNet:
    """Stands in for ``SesquiLSRNet``: records the raw input and target."""

    def __init__(self):
        self.raw_shapes = []
        self.raw_dtypes = []
        self.targets = []

    def __call__(self, raw, target):
        self.raw_shapes.append(tuple(raw.shape))
        self.raw_dtypes.append(raw.dtype)
        self.targets.append(tuple(target))
        return torch.full(
            (raw.shape[0], raw.shape[1], target[0], target[1]),
            0.25,
            dtype=raw.dtype,
        )


@pytest.fixture
def fake_net(monkeypatch):
    """Install a fake network + the chosen adaptor in place of the real load.

    ``_load_net`` is what touches the disk, so replacing it leaves the whole
    strategy under test — adaptor math, target math, dtype — with no weights.
    """

    def _install(adaptor):
        net = _RecordingNet()
        monkeypatch.setattr(sesqui_mod, "_load_net", lambda *a, **k: (net, adaptor))
        return net

    return _install


def _upscaler(*, dtype: torch.dtype = torch.float32):
    """A Sesqui upscaler over the fake net (the format name is meaningless here)."""
    return SesquiLSRUpscaler("fake-format", device="cpu", dtype=dtype)


# --------------------------------------------------------------- the transform


def test_upscale_target_is_the_scaled_size_in_vae_coords(fake_net):
    """The network is asked for ``scale``x the latent, in *its* coordinate space.

    Sesqui takes a raw-VAE target while the engine thinks in canonical (pipeline)
    coords, so the adaptor's ``vae_target_size`` must sit between the two.
    """
    net = fake_net(make_wan21())  # spatial_scale 1: raw coords == canonical
    z_up = _upscaler()(torch.ones(1, 16, 8, 8))

    assert net.targets == [(16, 16)]  # 2x the 8x8 canonical latent
    assert z_up.shape == (1, 16, 16, 16)


def test_upscale_target_respects_the_adaptors_spatial_scale(fake_net):
    """A 2x-spatial raw latent (patchified Flux.2) must still come back at 2x.

    Without the adaptor's spatial scale the network would be asked for 4x and the
    canonical output would grow twice as much as the pipeline planned for.
    """
    net = fake_net(make_flux2())  # 128ch canonical <-> 32ch raw at 2x the size
    z_up = _upscaler()(torch.ones(1, 128, 8, 8))

    assert net.raw_shapes == [(1, 32, 16, 16)]  # canonical -> raw
    assert net.targets == [(32, 32)]  # 2x canonical = 2x the 16x16 raw latent
    assert z_up.shape == (1, 128, 16, 16)


def test_upscale_returns_a_canonical_latent(fake_net):
    """The result is back in canonical space, not the raw space the net produced."""
    adaptor = make_wan21()
    fake_net(adaptor)

    z_up = _upscaler()(torch.ones(1, 16, 4, 4))

    # The fake net returns a constant raw latent, so the canonical result is that
    # constant run back through ``from_vae_latent`` — a raw passthrough would differ.
    expected = adaptor.from_vae_latent(torch.full((1, 16, 8, 8), 0.25))
    assert torch.allclose(z_up, expected)


def test_scale_is_intrinsic_to_the_algorithm(fake_net):
    """Sesqui is 2x by construction, so nobody can configure it out of step.

    The reassembly head pixel-shuffles 2x, which is why the factor is a property of
    the strategy and not a request: ``DiffusionModel.UPSCALE_SCALE`` mirrors it.
    """
    fake_net(LatentFormatAdaptor(external_channels=16))
    upscaler = _upscaler()

    assert SesquiLSRUpscaler.scale == 2
    assert upscaler(torch.ones(1, 16, 4, 4)).shape == (1, 16, 8, 8)


def test_upscale_runs_and_returns_in_its_own_dtype(fake_net):
    """A bf16 upscaler feeds its net bf16 and hands back bf16, whatever came in."""
    net = fake_net(make_wan21())

    z_up = _upscaler(dtype=torch.bfloat16)(torch.ones(1, 16, 4, 4, dtype=torch.float32))

    assert net.raw_dtypes == [torch.bfloat16]
    assert z_up.dtype == torch.bfloat16


def test_upscaler_is_a_latent_upscaler(fake_net):
    """The pipeline's only handle on the strategy is the interface."""
    fake_net(make_wan21())
    assert isinstance(_upscaler(), LatentUpscaler)


def test_unknown_format_is_rejected_at_construction():
    """A format with no committed weights fails where the adapter named it."""
    with pytest.raises(ValueError, match="unknown latent format"):
        SesquiLSRUpscaler("not_a_real_format", device="cpu", dtype=torch.float32)
