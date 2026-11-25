"""The mesh medium's HTTP body and routes."""
from __future__ import annotations

from typing import Callable, ClassVar, Optional, Union

from fastapi import FastAPI

from ..jobs import JobPlan, JobQueue
from ..media_request import MediaRequest
from ..media_routes import install_knobs_route, install_media_routes
from .finish.options import MeshFinishOptions
from .shapes import ShapeStore
from .spec import MeshSpec


class MeshRequest(MediaRequest, MeshSpec):
    """The mesh spec plus what only the wire carries: base64, and several images are views."""

    SPEC: ClassVar[type] = MeshSpec
    MEDIA_FIELDS: ClassVar[tuple[str, ...]] = ("image",)

    image: Optional[Union[str, list[str]]] = None


def install_mesh_routes(app: FastAPI, queue: JobQueue,
                        planner: Callable[[MeshSpec], JobPlan],
                        model_name: str,
                        shapes: Optional[ShapeStore] = None) -> None:
    install_media_routes(app, queue, planner, model_name, "mesh", MeshRequest)
    install_knobs_route(app, "/finish", MeshFinishOptions)

    if shapes is not None:
        @app.get("/shapes")
        def kept() -> dict:
            """Newest first: offering more than this offers shapes the server has let go."""
            return {"shapes": shapes.kept()}


__all__ = ["MeshRequest", "install_mesh_routes"]
