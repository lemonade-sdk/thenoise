"""The mesh medium's HTTP body and routes."""
from __future__ import annotations

from typing import Callable, ClassVar, Optional, Union

from fastapi import FastAPI

from ..jobs import JobPlan, JobQueue
from ..media_request import MediaRequest
from ..media_routes import install_media_routes
from .spec import MeshSpec


class MeshRequest(MediaRequest, MeshSpec):
    """The mesh spec plus what only the wire carries: base64, and several images are views."""

    SPEC: ClassVar[type] = MeshSpec
    MEDIA_FIELDS: ClassVar[tuple[str, ...]] = ("image",)

    image: Optional[Union[str, list[str]]] = None


def install_mesh_routes(app: FastAPI, queue: JobQueue,
                        planner: Callable[[MeshSpec], JobPlan],
                        model_name: str) -> None:
    install_media_routes(app, queue, planner, model_name, "mesh", MeshRequest)


__all__ = ["MeshRequest", "install_mesh_routes"]
