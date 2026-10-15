#!/usr/bin/env python
"""Tiny-shape micro-probe: who can consume/produce a packed (channels-last) layout?

Runs the smallest cases that can answer a layout question, for the driver's sake.
Every row prints the memory format the op actually handed back, not what was asked.
"""
import sys, time, torch
import torch.nn.functional as F

torch.manual_seed(0)
dev = "cuda"
out = []


def fmt(t):
    if t.dim() == 4:
        return "NHWC" if t.is_contiguous(memory_format=torch.channels_last) else \
               ("NCHW" if t.is_contiguous() else "strided" + str(tuple(t.stride())))
    if t.dim() == 5:
        return "NDHWC" if t.is_contiguous(memory_format=torch.channels_last_3d) else \
               ("NCDHW" if t.is_contiguous() else "strided" + str(tuple(t.stride())))
    return "contig" if t.is_contiguous() else "strided" + str(tuple(t.stride()))


def timeit(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / iters


print("device", torch.cuda.get_device_name(0), torch.version.hip)

# ---------------------------------------------------------------- 1. conv layout
print("\n== 1. what layout does a conv hand back ==")
for cl in (False, True):
    for bench in (False, True):
        torch.backends.cudnn.benchmark = bench
        mf = torch.channels_last if cl else torch.contiguous_format
        x = torch.randn(1, 32, 64, 64, device=dev, dtype=torch.bfloat16).to(memory_format=mf)
        w = torch.randn(64, 32, 3, 3, device=dev, dtype=torch.bfloat16).to(memory_format=mf)
        y = F.conv2d(x, w, padding=1)
        print(f"  in={'NHWC' if cl else 'NCHW'} benchmark={bench}: out {fmt(y)}")
torch.backends.cudnn.benchmark = False

# ------------------------------------------- 2. comfy_kitchen WMMA conv3d (NDHWC)
print("\n== 2. comfy_kitchen fp16_conv3d: does the WMMA kernel serve? ==")
import comfy_kitchen as ck
from comfy_kitchen.backends.hip import _wmma_fp16_conv3d

# conv2d as conv3d: T=1 axis. The kernel has NO padding of its own, so the caller
# pre-pads (that is what group_norm_silu_pad3d's fused pad is for).
for (c, k) in ((32, 64), (128, 128), (512, 512)):
    x0 = torch.randn(1, c, 1, 64, 64, device=dev, dtype=torch.float16)
    x = F.pad(x0, (1, 1, 1, 1, 0, 0)).to(memory_format=torch.channels_last_3d)
    w = torch.randn(k, c, 1, 3, 3, device=dev, dtype=torch.float16)
    served = _wmma_fp16_conv3d(x, w, None, None, [1, 1, 1]) is not None
    y = ck.fp16_conv3d(x, w, None, None, [1, 1, 1])
    ref = F.conv3d(x.float(), w.float())
    err = (y.float() - ref).abs().max().item()
    rel = ((y.float() - ref).norm() / ref.norm()).item()
    print(f"  C={c}->K={k}: wmma_served={served} out {fmt(y)} max|d|={err:.2e} relL2={rel:.2e}")

# ------------------------------------- 3. fused norm+act+pad, and its output layout
print("\n== 3. comfy_kitchen group_norm_silu_pad3d ==")
x = torch.randn(1, 128, 1, 64, 64, device=dev, dtype=torch.float16)
g = torch.ones(128, device=dev, dtype=torch.float16)
b = torch.zeros(128, device=dev, dtype=torch.float16)
y = ck.group_norm_silu_pad3d(x, g, b, num_groups=32, eps=1e-6,
                             pad=[1, 1, 1, 1, 0], silu=True, zero_pad=True)
ref = F.silu(F.group_norm(x.float(), 32, g.float(), b.float(), 1e-6))
ref = F.pad(ref, (1, 1, 1, 1))
print("  out", fmt(y), tuple(y.shape),
      "max|d|=%.2e" % (y.float() - ref).abs().max().item())

# ------------------------------------------------- 4. SDPA at the VAE's head_dim
print("\n== 4. fused SDPA at head_dim = the whole channel count ==")
from torch.nn.attention import sdpa_kernel, SDPBackend
for hd in (128, 256, 512):
    q = torch.randn(1, 1, 1024, hd, device=dev, dtype=torch.bfloat16)
    for be in (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION,
               SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH):
        try:
            with sdpa_kernel(be):
                y = F.scaled_dot_product_attention(q, q, q)
            print(f"  head_dim={hd:4d} {be.name:20s} ok  out {fmt(y)}")
        except Exception as e:
            print(f"  head_dim={hd:4d} {be.name:20s} REFUSED: {str(e)[:70]}")

print("\n== 5. bmm penalty for channel-major vs row-major operands ==")
for name, mk in (("row", lambda b, n, c: torch.randn(b, n, c, device=dev, dtype=torch.bfloat16)),
                 ("chan", lambda b, n, c: torch.randn(b, c, n, device=dev,
                                                      dtype=torch.bfloat16).transpose(1, 2))):
    n, c = 4096, 512
    q, k, v = (mk(1, n, c) for _ in range(3))
    ms1 = timeit(lambda: q @ k.transpose(-2, -1), iters=5)
    s = q @ k.transpose(-2, -1)
    ms2 = timeit(lambda: s.softmax(-1) @ v, iters=5)
    print(f"  {name}: qk^T {ms1:6.2f} ms   softmax  {ms2:6.2f} ms   "
          f"TF={4*n*n*c/((ms1+ms2)/1000)/1e12:5.1f}")
