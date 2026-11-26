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
        data.update(self.decode_media())
        return self.SPEC(**data)

    def decode_media(self) -> dict:
        """What the spec is handed for the request's media fields: base64 become bytes.

        A medium whose media is more than a bare picture overrides this and returns that
        field's spec form itself; the base covers pictures, one or several.
        """
        decoded = {}
        for name in self.MEDIA_FIELDS:
            encoded = getattr(self, name)
            if isinstance(encoded, list):
                decoded[name] = [verified_picture(name, one) for one in encoded]
            elif encoded is not None:
                decoded[name] = verified_picture(name, encoded)
        return decoded


__all__ = ["MediaRequest"]
