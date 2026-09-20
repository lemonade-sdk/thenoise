"""Qwen-Image 2.1 — single-stream DiT with a causal text/reference prefix.

``models`` is the transformer (and the per-run sequence it consumes), ``sampling``
the flow schedule, ``utils`` the checkpoint geometry/loader and ``encoder`` the
Qwen3-VL-8B conditioner (vision tokens in, image slots out).
"""
