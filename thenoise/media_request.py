"""The HTTP form of a media request."""
from __future__ import annotations

from typing import ClassVar, Optional

from .utils.media import verified_picture


class MediaRequest:
    """Base of a medium's HTTP body: the spec's fields, plus ``model`` and base64 media."""

    SPEC: ClassVar[type]
    MEDIA_FIELDS: ClassVar[tuple[str, ...]] = ()

    model: Optional[str] = None

    def to_spec(self):
        data = self.model_dump(exclude={"model", *self.MEDIA_FIELDS})
        for name in self.MEDIA_FIELDS:
            encoded = getattr(self, name)
            if encoded is not None:
                data[name] = verified_picture(name, encoded)
        return self.SPEC(**data)


__all__ = ["MediaRequest"]
