"""The plumbing behind a declared knob: one dataclass field carrying its own words,
bounds and step, so that the CLI flags, the request schema, the PNG record and the
web UI form all come from the same line.

Two surfaces are built on it: the per-kind control-map options
(:mod:`thenoise.controlmap.options`) and the post-decode filter chain
(:mod:`thenoise.postprocess.options`). It is torch-free on purpose — the CLI derives
its flags from a declaration and must not need the GPU stack to parse them.

A knob is declared with :func:`option`, which is an ordinary ``dataclass`` field plus
metadata:

* ``label`` — the words the UI shows, ``help`` — the words ``--help`` shows;
* ``ge``/``le``/``step`` — what the knob accepts (``step`` is only ever a control
  granularity, never a validation rule);
* ``choices``/``pool`` — a pick-list instead of a number, the values being names the
  request sends and the pool naming the list the UI fills from;
* ``hint``/``ui_min``/``ui_max`` — what the control shows when the honest declared
  range is wider than a useful slider;
* anything else is stored untouched for the owning surface to interpret (the
  post-decode chain reads ``filter`` and ``stage``).

Only ``label`` is required, so a knob nobody may leave at an unusable value cannot
forget to say what it accepts.
"""
from __future__ import annotations

import argparse
from dataclasses import field, fields
from typing import Any, Dict, Optional, Sequence, get_type_hints

#: The JSON type the web UI builds each knob's control from.
_JSON_TYPE = {bool: "boolean", int: "integer", float: "number", str: "string"}

#: ``dataclasses`` metadata key -> the word the schema and the UI use for it.
_WIRE_KEYS = (
    ("ge", "min"),
    ("le", "max"),
    ("step", "step"),
    ("choices", "choices"),
    ("pool", "pool"),
    ("hint", "hint"),
    ("ui_min", "ui_min"),
    ("ui_max", "ui_max"),
)


def option(
    default,
    *,
    label: str,
    help: str = "",
    ge: Optional[float] = None,
    le: Optional[float] = None,
    step: Optional[float] = None,
    choices: Optional[Sequence[str]] = None,
    pool: Optional[str] = None,
    hint: Optional[str] = None,
    ui_min: Optional[float] = None,
    ui_max: Optional[float] = None,
    **extra: Any,
):
    """One tunable: its default, the words shown for it, and what it accepts.

    Every knob on both surfaces has a *simple* default — the declaration doubles as
    the class attribute a flat request reads its default off.
    """
    meta: Dict[str, Any] = {"label": label}
    if help:
        meta["help"] = help
    for key, value in (
        ("ge", ge), ("le", le), ("step", step), ("choices", choices),
        ("pool", pool), ("hint", hint), ("ui_min", ui_min), ("ui_max", ui_max),
    ):
        if value is not None:
            meta[key] = value
    meta.update(extra)
    return field(default=default, metadata=meta)


def option_names(cls: type) -> tuple:
    """Every declared knob name, in declaration order."""
    return tuple(entry.name for entry in fields(cls))


def option_values(options) -> Dict[str, Any]:
    """A declaration's knob values as a flat ``{name: value}`` record."""
    return {entry.name: getattr(options, entry.name) for entry in fields(options)}


def option_meta(cls: type, name: str) -> Dict[str, Any]:
    """What one knob declared — used by surfaces that need a single field's bounds."""
    for entry in fields(cls):
        if entry.name == name:
            return entry.metadata
    raise KeyError(f"{cls.__name__} declares no knob {name!r}")


def declared_knobs(source: type):
    """A class decorator appending another class's declared knobs as ordinary fields.

    The target gains a real field per knob — same name, annotation and default — so a
    request object can carry the knobs flat without restating any of them:

    .. code-block:: python

        @dataclass
        @declared_knobs(PostprocessOptions)
        class GenerateRequest:
            prompt: str

    ``@dataclass`` must sit *above* it, so the fields exist by the time the dataclass
    collects them.
    """
    hints = get_type_hints(source)

    def decorate(target):
        declared = dict(getattr(target, "__annotations__", {}))
        for entry in fields(source):
            if entry.name in declared:
                raise ValueError(
                    f"{target.__name__} already declares {entry.name}, which "
                    f"{source.__name__} declares too"
                )
            declared[entry.name] = hints[entry.name]
            setattr(target, entry.name, entry.default)
        target.__annotations__ = declared
        return target

    return decorate


def wire_fields(source: type) -> Dict[str, Any]:
    """A declaration as ``pydantic.create_model`` field specs: the same flat wire name,
    the declared default, the bounds the request accepts and the words the API docs
    show for it.
    """
    from pydantic import Field  # imported here: the CLI reads declarations, not models

    hints = get_type_hints(source)
    specs: Dict[str, Any] = {}
    for entry in fields(source):
        meta = entry.metadata
        specs[entry.name] = (
            hints[entry.name],
            Field(
                default=entry.default,
                **{key: meta[key] for key in ("ge", "le") if key in meta},
                description=meta.get("help", ""),
            ),
        )
    return specs


def validate(options) -> None:
    """Reject a value outside ``ge``/``le``, or a choice that is not offered."""
    kind = getattr(options, "kind", "")
    for entry in fields(options):
        meta, value = entry.metadata, getattr(options, entry.name)
        low, high = meta.get("ge"), meta.get("le")
        if (low is not None and value < low) or (high is not None and value > high):
            bound = (
                f">= {low}" if high is None
                else f"<= {high}" if low is None
                else f"in [{low}, {high}]"
            )
            raise ValueError(f"{entry.name} must be {bound}, got {value}")
        choices = meta.get("choices")
        if choices is not None and value not in choices:
            named = f"{kind} " if kind else ""
            raise ValueError(
                f"unknown {named}{entry.name} {value!r}; expected one of {tuple(choices)}"
            )


def options_schema(cls: type) -> list:
    """One class's knobs as a form description: label, control, default, bounds."""
    hints = get_type_hints(cls)
    described = []
    for entry in fields(cls):
        meta = entry.metadata
        item: Dict[str, Any] = {
            "name": entry.name,
            "label": meta.get("label", entry.name),
            "type": _JSON_TYPE[hints[entry.name]],
            "default": entry.default,
        }
        for key, wire in _WIRE_KEYS:
            if key in meta:
                item[wire] = tuple(meta[key]) if key == "choices" else meta[key]
        described.append(item)
    return described


def _bounded(kind: type, name: str, meta: Dict[str, Any]):
    """A number parser that refuses a value outside the declared bounds."""
    low, high = meta.get("ge"), meta.get("le")
    if low is None and high is None:
        return kind

    def parse(text: str):
        value = kind(text)
        if (low is not None and value < low) or (high is not None and value > high):
            bound = (
                f">= {low}" if high is None
                else f"<= {high}" if low is None
                else f"in [{low}, {high}]"
            )
            raise argparse.ArgumentTypeError(f"{name} must be {bound}, got {value}")
        return value

    return parse


def add_flags(
    parser: argparse.ArgumentParser,
    cls: type,
    *,
    prefix: str = "",
    dest_prefix: str = "",
    suppress_unset: bool = False,
    bounds: bool = False,
) -> None:
    """One flag per declared knob, named, typed and described from the declaration.

    ``prefix``/``dest_prefix`` namespace a shared parser (the control maps use
    ``--canny-low`` for the ``canny`` kind's ``low``). ``suppress_unset`` leaves a
    flag nobody passed off the namespace instead of stamping the declared default on
    it, which is what a parser holding several kinds' flags needs. ``bounds`` refuses
    an out-of-range number where argparse parses it rather than letting the
    :func:`validate` of the options dataclass catch it later.
    """
    hints = get_type_hints(cls)
    for entry in fields(cls):
        meta = entry.metadata
        flag = f"--{prefix}-" if prefix else "--"
        flag += entry.name.replace("_", "-")
        help = f"{meta.get('help', '')} (default: {entry.default})".strip()
        kind = hints[entry.name]
        default = argparse.SUPPRESS if suppress_unset else entry.default
        if kind is bool:
            parser.add_argument(
                flag, dest=f"{dest_prefix}{entry.name}", action=argparse.BooleanOptionalAction,
                default=default, help=help,
            )
            continue
        kwargs: Dict[str, Any] = {
            "dest": f"{dest_prefix}{entry.name}",
            "type": _bounded(kind, entry.name, meta) if bounds else kind,
            "default": default,
            "help": help,
        }
        if meta.get("choices"):
            kwargs["choices"] = list(meta["choices"])
        parser.add_argument(flag, **kwargs)


__all__ = [
    "add_flags",
    "declared_knobs",
    "option",
    "option_meta",
    "option_names",
    "option_values",
    "options_schema",
    "validate",
    "wire_fields",
]
