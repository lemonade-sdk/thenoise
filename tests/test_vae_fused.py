"""The kitchen-fused strip machinery, and the two Flux AEs running on it.

Past the gate, which is production policy, everything forces the fused path on — its
own answer for these fixtures is "no", the kernels being GPU ones — and shrinks
``STRIP_ELEMS`` until an image really is walked in strips. Every case compares against
the plain path of the SAME weights, with a spy on a fused kernel so a test cannot pass
by never having run the code.
"""
from __future__ import annotations

import contextlib
from typing import Iterator

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import thenoise.vae.flux as flux_module
import thenoise.vae.flux2 as flux2_module
import thenoise.vae.fused as fused_module
from comfy_kitchen.registry import registry
from thenoise.vae import AutoencoderKLFlux, AutoencoderKLFlux2
from thenoise.vae.fused import (
    STRIP_ELEMS,
    _strip_rows,
    fused_available,
    group_norm_only,
    interleaved_view,
    pointwise,
    region,
    strip_apply,
    strip_conv,
    strip_down,
)

# --------------------------------------------------------------- the capability gate


def test_the_gate_refuses_a_device_without_kernels():
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        assert not fused_available(torch.device("cpu"), dtype)


def test_the_gate_refuses_fp32_even_where_kernels_exist():
    assert not fused_available(torch.device("cuda"), torch.float32)


def test_the_gate_refuses_bf16_on_rocm(monkeypatch):
    """A BF16 conv on ROCm goes to MIOpen, which runs a ``channels_last`` conv through
    NCHW and gives the whole gain back."""
    monkeypatch.setattr(torch.version, "hip", "test-fake-hip")
    assert not fused_available(torch.device("cuda"), torch.bfloat16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="a GPU-backed kitchen backend")
def test_the_gate_on_this_device():
    device = torch.device("cuda")
    hip = torch.version.hip is not None
    backend = "hip" if hip else "cuda"
    if registry.get_constraints(backend, "group_norm_silu_pad3d") is None:
        pytest.skip(f"kitchen registers no group norm for {backend}")
    assert fused_available(device, torch.float16)
    assert fused_available(device, torch.bfloat16) is not hip


# ------------------------------------------------------------------- the strip driver


def _silu_normed(x: torch.Tensor, norm: nn.GroupNorm) -> torch.Tensor:
    return F.silu(norm(x))


@pytest.fixture
def conv_norm():
    torch.manual_seed(0)
    norm = nn.GroupNorm(32, 64).float().eval().requires_grad_(False)
    conv = nn.Conv2d(64, 96, 3, padding=1).float().eval().requires_grad_(False)
    return norm, conv


@pytest.mark.parametrize("strip_elems", [STRIP_ELEMS, 1 << 13, 1 << 11, 1 << 9])
def test_strip_conv_matches_the_plain_chain(conv_norm, strip_elems, monkeypatch):
    norm, conv = conv_norm
    monkeypatch.setattr(fused_module, "STRIP_ELEMS", strip_elems)
    x = torch.randn(1, 64, 17, 9)
    with torch.no_grad():
        assert torch.allclose(strip_conv(conv, x, norm), conv(_silu_normed(x, norm)), atol=1e-5)


@pytest.mark.parametrize("strip_elems", [STRIP_ELEMS, 1 << 12, 1 << 10])
def test_strip_conv_doubles_rows_like_an_upsample(conv_norm, strip_elems, monkeypatch):
    """The conv's one halo row still covers a doubled strip: doubling cannot make a row
    need a neighbour two rows away."""
    _, conv = conv_norm
    monkeypatch.setattr(fused_module, "STRIP_ELEMS", strip_elems)
    x = torch.randn(1, 64, 12, 7)
    with torch.no_grad():
        got = strip_conv(conv, x, None, scale=2)
        want = conv(F.interpolate(x, scale_factor=2, mode="nearest"))
    assert got.shape == want.shape == (1, 96, 24, 14)
    assert torch.allclose(got, want, atol=1e-5)
    # channels-last, which the next fused call and the interleaved halving both need
    assert got.is_contiguous(memory_format=torch.channels_last)


@pytest.mark.parametrize("strip_elems", [1 << 12, 1 << 10])
def test_strip_conv_writes_and_adds_into_a_given_output(conv_norm, strip_elems, monkeypatch):
    norm, conv = conv_norm
    monkeypatch.setattr(fused_module, "STRIP_ELEMS", strip_elems)
    x = torch.randn(1, 64, 12, 7)
    want = conv(_silu_normed(x, norm))
    with torch.no_grad():
        copied = strip_conv(conv, x, norm, out=torch.zeros(1, 96, 12, 7))
        added = strip_conv(conv, x, norm, out=torch.full((1, 96, 12, 7), 3.0), accumulate=True)
    assert torch.allclose(copied, want, atol=1e-5)
    assert torch.allclose(added, want + 3.0, atol=1e-5)


@pytest.mark.parametrize("strip_elems", [STRIP_ELEMS, 1 << 12, 1 << 9])
def test_strip_down_matches_the_manual_pad(strip_elems, monkeypatch):
    monkeypatch.setattr(fused_module, "STRIP_ELEMS", strip_elems)
    torch.manual_seed(0)
    conv = nn.Conv2d(32, 32, 3, stride=2, padding=0).float().eval().requires_grad_(False)
    x = torch.randn(1, 32, 13, 11)
    with torch.no_grad():
        got = strip_down(conv, x)
    assert torch.allclose(got, conv(F.pad(x, (0, 1, 0, 1))), atol=1e-5)


# ------------------------------------------- what a fused pass does instead of a conv


@pytest.mark.parametrize("channels_last", [True, False])
def test_pointwise_answers_the_1x1_conv_it_replaces(channels_last):
    torch.manual_seed(0)
    conv = nn.Conv2d(16, 8, 1).float().eval().requires_grad_(False)
    x = torch.randn(1, 16, 5, 4)
    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last)
    got = pointwise(conv, x)
    assert torch.allclose(got, conv(x), atol=1e-5)
    assert got.is_contiguous(memory_format=torch.channels_last)


def test_group_norm_only_normalises_without_activation_or_pad():
    norm = nn.GroupNorm(4, 16).float().eval().requires_grad_(False)
    x = torch.randn(1, 16, 5, 4).contiguous(memory_format=torch.channels_last)
    with torch.no_grad():
        got = group_norm_only(x, norm)
    assert torch.allclose(got, norm(x), atol=1e-5)
    assert got.is_contiguous(memory_format=torch.channels_last)


# --------------------------------------------------------------- the generic strip walk


@pytest.fixture
def chain():
    """Two 3x3 convs, each padding by itself, width preserved: ``halo=2`` keeps a strip honest."""
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Conv2d(6, 9, 3, padding=1), nn.SiLU(), nn.Conv2d(9, 6, 3, padding=1)
    ).eval().requires_grad_(False)


@pytest.fixture
def narrowing_chain():
    torch.manual_seed(1)
    return nn.Sequential(
        nn.Conv2d(6, 4, 3, padding=1), nn.SiLU(), nn.Conv2d(4, 3, 3, padding=1)
    ).eval().requires_grad_(False)


def test_strip_apply_asks_for_the_whole_image_below_the_threshold():
    seen = []
    strip_apply(lambda s: seen.append(s.shape[-2]) or s, torch.randn(1, 4, 8, 8))
    assert seen == [8]  # one whole call, no strips


@pytest.mark.parametrize("strip_elems", [STRIP_ELEMS, 1 << 9, 1 << 7, 1 << 5])
def test_strip_apply_matches_the_chain_applied_at_once(chain, strip_elems, monkeypatch):
    monkeypatch.setattr(fused_module, "STRIP_ELEMS", strip_elems)
    x = torch.randn(1, 6, 34, 5)
    with torch.no_grad():
        assert torch.allclose(strip_apply(chain, x, halo=2), chain(x), atol=1e-5)


@pytest.mark.parametrize("strip_elems", [STRIP_ELEMS, 1 << 9, 1 << 7])
def test_strip_apply_matches_the_upsample_inside_it(strip_elems, monkeypatch):
    monkeypatch.setattr(fused_module, "STRIP_ELEMS", strip_elems)
    up = nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest-exact"), nn.Conv2d(4, 4, 3, padding=1)
    ).eval().requires_grad_(False)
    x = torch.randn(1, 4, 24, 5)
    with torch.no_grad():
        assert torch.allclose(strip_apply(up, x, scale=2, halo=1), up(x), atol=1e-5)


# Down to one row per strip, where the driver has to widen a strip to the halo: a strip
# cannot hold back more rows than it made.
@pytest.mark.parametrize("strip_elems", [1 << 9, 1 << 7, 1 << 6, 1 << 4])
def test_strip_apply_accumulates_into_the_tensor_it_read(chain, strip_elems, monkeypatch):
    """The rows a strip writes are the next strip's halo, so they wait for it."""
    monkeypatch.setattr(fused_module, "STRIP_ELEMS", strip_elems)
    x = torch.randn(1, 6, 34, 5)
    with torch.no_grad():
        want = x + chain(x)
        buf = x.clone()
        got = strip_apply(chain, buf, halo=2, out=buf, accumulate=True)
    assert got is buf
    assert torch.allclose(got, want, atol=1e-5)


@pytest.mark.parametrize("strip_elems", [1 << 9, 1 << 7, 1 << 6, 1 << 4])
def test_strip_apply_lands_a_block_in_the_leading_channels_of_its_input(
    narrowing_chain, strip_elems, monkeypatch
):
    """A ``C -> C'`` block writing into the leading channels of the input it read."""
    monkeypatch.setattr(fused_module, "STRIP_ELEMS", strip_elems)
    x = torch.randn(1, 6, 30, 5)
    with torch.no_grad():
        want = narrowing_chain(x)
        pristine = x.clone()
        got = strip_apply(narrowing_chain, x, halo=2, out=x.narrow(1, 0, 3))
    assert torch.allclose(got, want, atol=1e-5)
    assert torch.equal(x.narrow(1, 3, 3), pristine.narrow(1, 3, 3))  # the rest is untouched


def test_the_destination_is_handed_over_only_where_nothing_reads_it(monkeypatch):
    """A fused kernel keeps its in-place write until the rows would land too early."""
    monkeypatch.setattr(fused_module, "STRIP_ELEMS", 1 << 7)
    seen: list[bool] = []

    def fn(rows, top, bottom, dst):
        seen.append(dst is not None)
        return rows + 1.0

    x = torch.randn(1, 3, 34, 5)
    assert _strip_rows(fn, x, 1 << 7, out=torch.empty_like(x)) is not None
    assert len(seen) > 1 and all(seen)
    seen.clear()
    assert _strip_rows(fn, x, 1 << 7, out=x) is x
    assert len(seen) > 1 and not any(seen)


# ------------------------------------------------------------- the shared buffer views


def test_interleaved_view_splits_the_storage_in_two_disjoint_images():
    x = torch.zeros(1, 16, 5, 4).contiguous(memory_format=torch.channels_last)
    front, back = interleaved_view(x, 8, 0), interleaved_view(x, 8, 1)
    assert (front.shape, back.shape) == ((1, 8, 5, 4), (1, 8, 5, 4))
    assert front.stride() == back.stride() == (5 * 4 * 16, 1, 4 * 16, 8)

    front.fill_(1.0)
    assert int((x == 1).sum()) == front.numel()  # no shared element,
    assert int((x == 0).sum()) == back.numel()  # and no gap between the halves


def test_region_is_the_channels_last_tail_of_a_buffer():
    buf = torch.arange(48, dtype=torch.float32)
    tail = region(buf, buf.numel(), 4, 3, 2)
    assert tail.shape == (1, 4, 3, 2)
    assert tail.is_contiguous(memory_format=torch.channels_last)
    assert torch.equal(tail.permute(0, 2, 3, 1).flatten(), buf[-24:])  # the buffer is NHWC
    head = region(buf, buf.numel() - 24, 4, 3, 2)
    assert torch.equal(torch.cat([head.permute(0, 2, 3, 1).flatten(), tail.permute(0, 2, 3, 1).flatten()]), buf)


# ------------------------------------------------------- the Flux.2 AE on the fused path


def _flux2_pair() -> tuple[AutoencoderKLFlux2, AutoencoderKLFlux2]:
    """A plain and a fused Flux.2 AE sharing one set of tiny weights, BN stats included."""
    torch.manual_seed(0)
    stock = AutoencoderKLFlux2(encoder_ch=32, decoder_ch=32, fused=False).eval().requires_grad_(False)
    fused = AutoencoderKLFlux2(encoder_ch=32, decoder_ch=32, fused=True).eval().requires_grad_(False)
    fused.load_state_dict(stock.state_dict())
    stock.bn.running_var.uniform_(0.5, 2.0)  # the decode un-normalises with these
    stock.bn.running_mean.normal_()
    fused.bn.load_state_dict(stock.bn.state_dict())
    return stock, fused


#: Every codec module that asks the gate before going fused.
_GATED = (flux2_module, flux_module)


@contextlib.contextmanager
def _fused_path(strip_elems: int) -> Iterator[list[torch.Size]]:
    """Force the fused path of every codec at a given strip size, counting the kernel calls."""
    calls: list[torch.Size] = []
    real_norm = fused_module.group_norm_silu_pad3d
    saved_gates = {module: module.fused_available for module in _GATED}
    saved_strip = fused_module.STRIP_ELEMS

    def counting(x, *args, **kwargs):
        calls.append(x.shape)
        return real_norm(x, *args, **kwargs)

    for module in _GATED:
        module.fused_available = lambda device, dtype: True
    fused_module.group_norm_silu_pad3d = counting
    fused_module.STRIP_ELEMS = strip_elems
    try:
        yield calls
    finally:
        for module, gate in saved_gates.items():
            module.fused_available = gate
        fused_module.group_norm_silu_pad3d = real_norm
        fused_module.STRIP_ELEMS = saved_strip


@pytest.mark.parametrize("strip_elems", [STRIP_ELEMS, 1 << 15, 1 << 11, 1 << 9])
def test_flux2_fused_decode_matches_the_plain_one(strip_elems):
    """Also the test of the shared buffer's aliasing: every level writes over the buffer
    while reading the previous level out of it.
    """
    stock, fused = _flux2_pair()
    latents = torch.randn(1, 128, 3, 5)
    with _fused_path(strip_elems) as calls:
        with torch.no_grad():
            want = stock.decode_to_pixels(latents)
            got = fused.decode_to_pixels(latents)
    assert calls, "the fused path never ran"
    assert got.shape == (1, 3, 48, 80)
    assert torch.allclose(got, want, atol=1e-4)


@pytest.mark.parametrize("strip_elems", [STRIP_ELEMS, 1 << 15, 1 << 11])
def test_flux2_fused_encode_matches_the_plain_one(strip_elems):
    stock, fused = _flux2_pair()
    pixels = torch.randn(1, 3, 48, 32)
    with _fused_path(strip_elems) as calls:
        with torch.no_grad():
            want = stock.encode_pixels_to_latents(pixels)
            got = fused.encode_pixels_to_latents(pixels)
    assert calls, "the fused path never ran"
    assert got.shape == want.shape == (1, 128, 3, 2)
    assert torch.allclose(got, want, atol=1e-4)


def test_flux2_fused_decode_rounds_a_batch_one_image_at_a_time():
    stock, fused = _flux2_pair()
    latents = torch.randn(2, 128, 2, 3)
    with _fused_path(1 << 12):
        with torch.no_grad():
            want = stock.decode_to_pixels(latents)
            got = fused.decode_to_pixels(latents)
    assert got.shape == want.shape == (2, 3, 32, 48)
    assert torch.allclose(got, want, atol=1e-4)


def test_flux2_leaves_the_pixels_alone_when_the_gate_says_no(monkeypatch):
    stock, fused = _flux2_pair()
    latents = torch.randn(1, 128, 2, 3)
    with _fused_path(STRIP_ELEMS) as calls:
        flux2_module.fused_available = fused_available  # the real gate, on CPU
        with torch.no_grad():
            want = stock.decode_to_pixels(latents)
            got = fused.decode_to_pixels(latents)
    assert not calls
    assert torch.equal(got, want)


# -------------------------------------------------------- the Flux AE on the fused path


def _flux_pair() -> tuple[AutoencoderKLFlux, AutoencoderKLFlux]:
    """A plain and a fused Flux AE sharing one set of tiny weights.

    The widths keep the shipped ``128/256/512/512`` ratios, so the two channel-halving
    blocks — the ones that run inside the residual stream — are the ones under test.
    """
    torch.manual_seed(0)
    kwargs = dict(block_out_channels=(32, 64, 128, 128), norm_num_groups=8, layers_per_block=1)
    stock = AutoencoderKLFlux(fused=False, **kwargs).eval().requires_grad_(False)
    fused = AutoencoderKLFlux(fused=True, **kwargs).eval().requires_grad_(False)
    fused.load_state_dict(stock.state_dict())
    return stock, fused


@pytest.mark.parametrize("strip_elems", [STRIP_ELEMS, 1 << 15, 1 << 12, 1 << 10])
def test_flux_fused_decode_matches_the_plain_one(strip_elems):
    stock, fused = _flux_pair()
    latents = torch.randn(1, 16, 3, 5)
    with _fused_path(strip_elems) as calls:
        with torch.no_grad():
            want = stock.decode_to_pixels(latents)
            got = fused.decode_to_pixels(latents)
    assert calls, "the fused path never ran"
    assert got.shape == (1, 3, 24, 40)
    assert torch.allclose(got, want, atol=1e-4)


def test_flux_fused_decode_rounds_a_batch_one_image_at_a_time():
    stock, fused = _flux_pair()
    latents = torch.randn(2, 16, 2, 3)
    with _fused_path(1 << 12):
        with torch.no_grad():
            want = stock.decode_to_pixels(latents)
            got = fused.decode_to_pixels(latents)
    assert got.shape == want.shape == (2, 3, 16, 24)
    assert torch.allclose(got, want, atol=1e-4)


def test_flux_fused_decode_gives_a_level_room_when_there_is_none():
    """A last level that does not halve fills the buffer with its own stream, so its
    blocks allocate instead of scratching into what they are reading.
    """
    torch.manual_seed(0)
    kwargs = dict(block_out_channels=(32, 32, 64, 64), norm_num_groups=8, layers_per_block=1)
    stock = AutoencoderKLFlux(fused=False, **kwargs).eval().requires_grad_(False)
    fused = AutoencoderKLFlux(fused=True, **kwargs).eval().requires_grad_(False)
    fused.load_state_dict(stock.state_dict())
    latents = torch.randn(1, 16, 2, 2)
    with _fused_path(1 << 12) as calls:
        with torch.no_grad():
            want = stock.decode_to_pixels(latents)
            got = fused.decode_to_pixels(latents)
    assert calls, "the fused path never ran"
    assert torch.allclose(got, want, atol=1e-4)


def test_flux_leaves_the_pixels_alone_when_the_gate_says_no():
    stock, fused = _flux_pair()
    latents = torch.randn(1, 16, 2, 3)
    with _fused_path(STRIP_ELEMS) as calls:
        flux_module.fused_available = fused_available  # the real gate, on CPU
        with torch.no_grad():
            want = stock.decode_to_pixels(latents)
            got = fused.decode_to_pixels(latents)
    assert not calls
    assert torch.equal(got, want)
