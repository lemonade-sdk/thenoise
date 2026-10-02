"""Shared logging setup, based on sd-scripts' ``library/utils.py``."""
from __future__ import annotations

import logging
import sys


def setup_logging(args=None, log_level=None, reset=False):
    """Configure the root logger; a no-op once one is configured unless ``reset``."""
    if logging.root.handlers:
        if reset:
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
        else:
            return

    if log_level is None and args is not None:
        log_level = args.console_log_level
    if log_level is None:
        log_level = "INFO"
    log_level = getattr(logging, log_level)

    handler = None
    if args is not None and args.console_log_file:
        handler = logging.FileHandler(args.console_log_file, mode="w")
    else:
        if not args or not args.console_log_simple:
            try:
                from rich.logging import RichHandler
                from rich.console import Console

                handler = RichHandler(console=Console(stderr=True))
            except ImportError:
                pass
        if handler is None:
            handler = logging.StreamHandler(sys.stdout)
            handler.propagate = False

    formatter = logging.Formatter(
        fmt="%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logging.root.setLevel(log_level)
    logging.root.addHandler(handler)
