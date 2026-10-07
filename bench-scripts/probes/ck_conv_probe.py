#!/usr/bin/env python
"""Map the service gate and the throughput of comfy_kitchen's WMMA conv3d.

The kernel takes NDHWC views and writes NDHWC — the one thing MIOpen's conv refuses —
and fuses bias + residual with no im2col staging. It is fp16-only and both the Python
wrapper and the HIP launcher may decline a shape, so the gate has to be mapped and the
throughput measured before anything is built on it.

Sizes stay small: the dev box's driver does not like big launches. A FLOP cap keeps
every single call under ~0.1 TFLOP.
"""
import time
import torch
import torch.nn.functional as F

import comfy_kitchen as ck
from comfy_kitchen.backends.hip import _wmma_fp16_conv3d

dev = "cuda"
torch.manual_seed(0)
FLOP_CAP = 0.12e12


def served(c, k, hw):
    """Run (or fail to run) the WMMA kernel once on a pre-padded NDHWC input."""
    if 2 * hw * hw * c * k * 9 > FLOP_CAP:
        return None
    x = torch.randn(1, c, 1, hw + 2, hw + 2, device=dev, dtype=torch.float16)
    w = torch.randn(k, c, 1, 3, 3, device=dev, dtype=torch.float16)
    ok = _wmma_fp16_conv3d(x.to(memory_format=torch.channels_last_3d), w,
                           None, None, [1, 1, 1]) is not None
    del x, w
    return ok


def timeit(fn, iters=3, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / iters


print("device", torch.cuda.get_device_name(0))

# ---------------------------------------------------------------- the service gate
print("\n== fp16_conv3d service gate: C=K columns, spatial rows (3x3, stride 1) ==")
cs = (32, 64, 96, 128, 192, 256, 384, 512)
print("        " + "".join(f"{hw:>10d}" for hw in (32, 64, 128, 256)))
for c in cs:
    cells = []
    for hw in (32, 64, 128, 256):
        r = served(c, c, hw)
        cells.append({True: " served", False: "      -", None: "   cap"}[r])
    print(f"  C=K={c:<4d}" + "".join(cells))

print("\n== the same gate at the K the codecs actually use (hw=128) ==")
for c, k in ((96, 96), (96, 192), (128, 128), (128, 256), (128, 3), (256, 256),
             (256, 512), (512, 512), (512, 256), (512, 128), (768, 768), (1024, 1024)):
    r = served(c, k, 128)
    print(f"   C={c:<5d}K={k:<5d} {'served' if r else ('-' if r is False else 'cap')}")

# --------------------------------------------------------- throughput, per conv shape
print("\n== throughput: WMMA NDHWC vs torch NCHW vs torch NHWC-in, fp16 ==")
print("   (ck = comfy_kitchen fp16_conv3d; TF is analytic 2*HW*C*K*9 over measured ms)")
for c, hw in ((512, 128), (256, 256), (128, 512), (192, 128)):
    k = c
    xp = torch.randn(1, c, 1, hw + 2, hw + 2, device=dev, dtype=torch.float16)
    w = torch.randn(k, c, 1, 3, 3, device=dev, dtype=torch.float16)
    w2 = w.reshape(k, c, 3, 3).contiguous()
    x2 = xp[..., 1:-1, 1:-1].contiguous()
    flop = 2 * (hw * hw) * c * k * 9
    ms_ck = timeit(lambda: ck.fp16_conv3d(xp.to(memory_format=torch.channels_last_3d),
                                          w, None, None, [1, 1, 1]))
    ms_nchw = timeit(lambda: F.conv2d(x2.squeeze(2), w2, padding=1))
    ms_nhwc = timeit(lambda: F.conv2d(x2.squeeze(2).to(memory_format=torch.channels_last),
                                      w2.to(memory_format=torch.channels_last), padding=1))
    print(f"   C=K={c:<4d} {hw:4d}x{hw:<4d}  ck {ms_ck:8.2f} ms {flop/ms_ck*1e-9:5.1f} TF | "
          f"nchw {ms_nchw:8.2f} ms {flop/ms_nchw*1e-9:5.1f} TF | "
          f"nhwc-in {ms_nhwc:8.2f} ms {flop/ms_nhwc*1e-9:5.1f} TF")
    del xp, w, w2, x2

# ------------------------------------------------- one fused resblock, both ways
print("\n== norm+silu+pad+conv+bias+residual, fused (ck) vs eager (torch), fp16 ==")
for c, hw in ((256, 128), (128, 256)):
    x = torch.randn(1, c, 1, hw, hw, device=dev, dtype=torch.float16)
    g = torch.randn(c, device=dev, dtype=torch.float16) * 0.1 + 1
    b = torch.randn(c, device=dev, dtype=torch.float16) * 0.1
    w = torch.randn(c, c, 1, 3, 3, device=dev, dtype=torch.float16) / (c * 9) ** 0.5
    bias = torch.randn(c, device=dev, dtype=torch.float16) * 0.1

    def chain_ck():
        h = ck.group_norm_silu_pad3d(x, g, b, 32, 1e-6, [1, 1, 1, 1, 0], True)
        return ck.fp16_conv3d(h, w, bias, x, [1, 1, 1])

    def chain_torch():
        h = F.silu(F.group_norm(x, 32, g, b, 1e-6))
        h = F.conv2d(F.pad(h.squeeze(2), (1, 1, 1, 1)), w.reshape(c, c, 3, 3), bias)
        return h.unsqueeze(2) + x

    yc, yt = chain_ck(), chain_torch()
    ms_c, ms_t = timeit(chain_ck), timeit(chain_torch)
    rel = ((yc.float() - yt.float()).norm() / yt.float().norm()).item()
    nd = yc.is_contiguous(memory_format=torch.channels_last_3d)
    print(f"   C=K={c} {hw}x{hw}: ck {ms_c:7.2f} ms | torch {ms_t:7.2f} ms | "
          f"{ms_t/ms_c:4.2f}x | out {'NDHWC' if nd else yc.stride()} | relL2 {rel:.1e}")
    del x, g, b, w, bias, yc, yt
