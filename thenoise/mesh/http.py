"""The mesh medium's HTTP body and routes."""
from __future__ import annotations

from typing import Callable, ClassVar, Optional, Union

from fastapi import FastAPI

from ..jobs import JobPlan, JobQueue
from ..media_request import MediaRequest
from ..media_routes import install_media_routes
from ..utils.options import options_schema
from .finish.options import MeshFinishOptions
from .spec import MeshSpec


class MeshRequest(MediaRequest, MeshSpec):
    """The mesh spec, plus what only the wire carries: a model name, base64 media.

    ``image`` takes one base64 picture or several, and several are views of one object.
    """

    SPEC: ClassVar[type] = MeshSpec
    MEDIA_FIELDS: ClassVar[tuple[str, ...]] = ("image",)

    image: Optional[Union[str, list[str]]] = None


def install_mesh_routes(app: FastAPI, queue: JobQueue,
                        planner: Callable[[MeshSpec], JobPlan],
                        model_name: str) -> None:
    install_media_routes(app, queue, planner, model_name, "mesh", MeshRequest)

    @app.get("/finish")
    def finish():
        """The shape-finish knobs, described for the web UI to build its controls from."""
        return {"options": options_schema(MeshFinishOptions)}


__all__ = ["MeshRequest", "install_mesh_routes"]
