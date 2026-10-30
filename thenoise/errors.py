"""Errors that outlive the request that raised them. No imports, so every layer can
name them: the adapters that raise, the runtime that recovers, the wire that answers.
"""
from __future__ import annotations


class ModelReloadNeeded(RuntimeError):
    """The loaded instance is in a state only rebuilding it from disk can clear.

    An out-of-memory LoRA switch is the case: layers merged, nothing recorded to undo
    them, so another request would blend the half-merge into its result.
    """
