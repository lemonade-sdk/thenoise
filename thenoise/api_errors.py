"""The error envelopes this server answers with.

A route answers a refusal in the format it speaks: one that hands back bytes answers with
the reason as plain text, the job API answers with `{"error": ...}`. The status code
carries the class of the failure either way.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from fastapi import Response
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


def error(status_code: int, reason: str) -> Response:
    return Response(status_code=status_code, content=reason, media_type="text/plain")


def json_error(status_code: int, reason: str) -> Response:
    return JSONResponse({"error": reason}, status_code=status_code)


def guarded(failure: str, action: Callable[[], Any], present: Callable[[Any], Any], *,
            envelope: Callable[[int, str], Response],
            statuses: dict[type[Exception], int] | None = None) -> Any:
    """Run one request, answering with what ``present`` makes of the result or with a failure.

    ``ValueError`` is the request's fault: 400 echoing the reason. A type named in
    ``statuses`` answers with the status it names. Anything else is 500 and logged.
    """
    try:
        return present(action())
    except ValueError as e:
        return envelope(400, str(e))
    except Exception as e:
        for kind, status in (statuses or {}).items():
            if isinstance(e, kind):
                return envelope(status, str(e))
        logger.exception("%s failed", failure)
        return envelope(500, f"{failure} failed: {e}")
