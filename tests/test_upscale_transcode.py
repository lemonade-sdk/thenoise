"""The Qwen-Image 2.1 latent-transcode upscaler (fake VAE, trained weights).

The strategy has two halves: the bridge network, which is pure tensor math and is
exercised here on a toy-width instance, and the ``vae.decode_features`` call that
feeds it, which is exercised here against a fake VAE that records what it was
asked for. The VAE half of that — the decoder prefix itself — is covered in
``test_wan22_vae.py``; the pipeline drives every strategy through the same
``LatentUpscaler`` mock (see ``conftest``), and the adapter wiring is in
``test_qwen_image21.py``.

The committed weights are loaded in the last group of tests: they are ~6 M
parameters, cheap enough to check for real (a strict load is the only thing that
ties the vendored class to the file it was trained against).
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from thenoise.upscale import LatentUpscaler, LatentTranscodeNet, Qwen21TranscodeUpscaler
from thenoise.upscale import qwen21_transcode as transcode_mod
from thenoise.upscale.qwen21_transcode import WEIGHTS
from thenoise.utils.safetensors import load_safetensors, upscale_weight_path

# Toy widths: the same two-path shape as the trained bridge at a size that makes
# these instantiations free. The real ones are 1152 feature / 64 latent / 384 wide.
TOY = dict(feature_channels=6, latent_channels=4, width=8, depth=2)


class FakeVAE:
    """A stand-in for the Wan 2.2 VAE's decoder-prefix entry point.

    Only ``decode_features`` and ``z_dim`` exist, because that is all the strategy
    is allowed to reach for: it gets the canonical latent and hands back a feature
    map at 2x, recording what it saw so a test can assert the strategy did not
    denormalise the latent itself.
    """

    def __init__(self, *, z_dim: int = 4, feature_channels: int = 6):
        self.z_dim = z_dim
        self.feature_channels = feature_channels
        self.seen: list[tuple] = []

    def decode_features(self, latents, *, blocks=1):
        self.seen.append((tuple(latents.shape), latents.dtype, blocks))
        b, _, h, w = latents.shape
        return torch.full(
            (b, self.feature_channels, 2 * h, 2 * w),
            0.5,
            dtype=latents.dtype,
            device=latents.device,
        )


def _toy_net(**overrides) -> LatentTranscodeNet:
    return LatentTranscodeNet(**{**TOY, **overrides}).eval().requires_grad_(False)


def _toy_feature(latent: torch.Tensor, value: float = 0.0) -> torch.Tensor:
    return torch.full(
        (1, TOY["feature_channels"], 2 * latent.shape[-2], 2 * latent.shape[-1]),
        value,
    )


@pytest.fixture
def fake_load(monkeypatch):
    """Replace the weight load with a toy-width net, leaving the strategy intact."""

    def _install(**net_kwargs):
        def _load(device, dtype):
            return _toy_net(**net_kwargs).to(device=device, dtype=dtype)

        monkeypatch.setattr(transcode_mod, "_load_net", _load)

    return _install


def _upscaler(vae=None, *, fake_load, dtype: torch.dtype = torch.float32, **net_kwargs):
    fake_load(**net_kwargs)
    return Qwen21TranscodeUpscaler(vae or FakeVAE(), device="cpu", dtype=dtype)


# --------------------------------------------------------------- the strategy


def test_the_bridge_is_a_latent_upscaler_2x_by_construction(fake_load):
    """The pipeline's only handle on it is the interface, and the factor is fixed."""
    up = _upscaler(fake_load=fake_load)

    assert isinstance(up, LatentUpscaler)
    assert Qwen21TranscodeUpscaler.scale == 2 == up.scale


def test_the_bridge_conditions_on_the_canonical_latent_untouched(fake_load):
    """The VAE is handed the canonical latent, and asked for exactly one stage.

    The denormalisation is the VAE's job (``decode_features`` does it, the way
    ``decode_to_pixels`` does), so what arrives there must still be the latent the
    DiT produced — anything else silently feeds the decoder a different image.
    ``blocks=1`` is the 2x feature the bridge was trained against; a second block
    would ask for a 4x feature and the net would reject it.
    """
    vae = FakeVAE()
    z = torch.arange(1 * 4 * 3 * 5, dtype=torch.float32).reshape(1, 4, 3, 5)

    _upscaler(vae, fake_load=fake_load)(z)

    assert vae.seen == [(tuple(z.shape), torch.float32, 1)]


def test_the_bridge_returns_the_canonical_latent_at_2x(fake_load):
    """Canonical in, canonical out at double the resolution — the pipeline's contract."""
    up = _upscaler(fake_load=fake_load)

    assert up(torch.zeros(1, 4, 3, 5)).shape == (1, 4, 6, 10)


def test_the_bridge_runs_and_returns_in_its_own_dtype(fake_load):
    """A bf16 upscaler hands both the decoder prefix and the net bf16, and returns bf16.

    The feature is re-cast explicitly on the way back from the VAE, so a VAE sitting
    on another dtype cannot quietly upcast a whole feature map.
    """
    vae = FakeVAE()
    up = _upscaler(vae, fake_load=fake_load, dtype=torch.bfloat16)

    z_up = up(torch.ones(1, 4, 3, 5, dtype=torch.float32))

    assert vae.seen[0][1] == torch.bfloat16  # the VAE was handed a bf16 latent
    assert z_up.dtype == torch.bfloat16


def test_a_vae_without_a_decoder_prefix_is_rejected(fake_load):
    """Only the Wan 2.2 family exposes the prefix; a plain VAE fails where it's wired."""

    class NoPrefixVAE:
        z_dim = 4

    with pytest.raises(ValueError, match="decode_features"):
        _upscaler(NoPrefixVAE(), fake_load=fake_load)


def test_a_vae_whose_latent_does_not_fit_the_bridge_is_rejected(fake_load):
    """The bridge's latent width is the trained model's ``z_dim``, not a guess."""
    with pytest.raises(ValueError, match="does not fit this bridge"):
        _upscaler(FakeVAE(z_dim=48), fake_load=fake_load)


def test_a_decoder_prefix_of_the_wrong_width_fails_loudly(fake_load):
    """A VAE whose first upsample stage is not the bridge's feature width cannot pass.

    Not silently: the width is the one number the strategy cannot check up front
    (there is no feature map to measure until an upscale runs), so the failure the
    conv raises has to stay readable.
    """
    with pytest.raises(RuntimeError, match=r"expected input.*to have 6 channels"):
        _upscaler(FakeVAE(feature_channels=9), fake_load=fake_load)(torch.zeros(1, 4, 3, 5))


# ---------------------------------------------------------------- the network


def test_a_zeroed_trunk_degrades_to_the_corrected_bicubic_upsample():
    """With the detail path switched off, the net IS its skip: ``skip(bicubic(z))``.

    That is the property the two-path design buys — the trained trunk adds detail on
    top of an interpolation that already looks right, rather than the net having to
    synthesise the whole image.
    """
    net = _toy_net()
    for tensor in (net.head.weight, net.head.bias, net.out.weight, net.out.bias):
        torch.nn.init.zeros_(tensor)

    latent = torch.randn(1, TOY["latent_channels"], 3, 5)
    skip = F.interpolate(latent, size=(6, 10), mode="bicubic", align_corners=False)

    assert torch.allclose(net(_toy_feature(latent), latent), net.skip(skip), atol=1e-5)


def test_the_decoder_feature_actually_reaches_the_output():
    """The other path: holding the latent fixed, the feature must change the result.

    A bridge that ignored its conditioning would still pass every shape test above,
    and would behave exactly like a plain learned resize.
    """
    net = _toy_net()
    latent = torch.randn(1, TOY["latent_channels"], 3, 5)

    a = net(_toy_feature(latent, value=-1.0), latent)
    b = net(_toy_feature(latent, value=1.0), latent)

    assert not torch.allclose(a, b)


def test_the_feature_must_be_exactly_2x_the_latent_grid():
    """The coupling to the decoder's one upsample stage is asserted, not assumed."""
    net = _toy_net()
    latent = torch.zeros(1, TOY["latent_channels"], 4, 4)

    for size in [(4, 4), (3, 3), (8, 7), (16, 16)]:
        with pytest.raises(ValueError, match="exactly 2x the latent grid"):
            net(torch.zeros(1, TOY["feature_channels"], *size), latent)


def test_the_latent_must_have_the_trained_width():
    net = _toy_net()

    with pytest.raises(ValueError, match="expected a 4-channel latent"):
        net(torch.zeros(1, TOY["feature_channels"], 6, 6), torch.zeros(1, 6, 3, 5))


# ------------------------------------------------------- the committed weights


def test_the_committed_weights_match_the_vendored_architecture():
    """The file is the architecture: a strict load of the vendored class must be exact.

    Nothing reconstructs the network from checkpoint metadata, so this is the only
    thing tying ``LatentTranscodeNet``'s defaults to the shipped file — a rename, a
    depth change or a stray tensor fails here instead of at someone's first upscale.
    """
    state_dict = load_safetensors(upscale_weight_path(WEIGHTS), device="cpu")
    net = LatentTranscodeNet()
    net.load_state_dict(state_dict, strict=True)

    assert {k for k, v in state_dict.items() if v.dtype != torch.bfloat16} == set()
    # ``head``'s input width is the VAE side of the contract (``dec_dim *
    # dim_mult[-1]`` of the Qwen-Image 2.1 decoder), the latent width is its z_dim.
    assert net.head.weight.shape == (384, 1152, 1, 1)
    assert net.latent_channels == net.skip.weight.shape[0] == 64


def _committed_net(device, dtype) -> LatentTranscodeNet:
    """The real vendored bridge, loaded straight from the committed file."""
    net = LatentTranscodeNet()
    net.load_state_dict(
        load_safetensors(upscale_weight_path(WEIGHTS), device=device), strict=True
    )
    return net.to(device=device, dtype=dtype).eval().requires_grad_(False)


def test_the_committed_bridge_upscales_a_real_latent(monkeypatch):
    """The trained net, driven through the strategy against a toy decoder prefix.

    The real QI21 VAE is ~0.7 GB and out of bounds for the suite, so the feature map
    is faked at its true width (1152 ch at 2x) — enough to run every trained tensor.
    """
    monkeypatch.setattr(transcode_mod, "_load_net", _committed_net)
    up = Qwen21TranscodeUpscaler(
        FakeVAE(z_dim=64, feature_channels=1152), device="cpu", dtype=torch.float32
    )

    z_up = up(torch.randn(1, 64, 4, 6))

    assert z_up.shape == (1, 64, 8, 12)
    assert torch.isfinite(z_up).all()
