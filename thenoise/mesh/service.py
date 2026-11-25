"""Turn a mesh request into a job the queue can run against the loaded model."""
from __future__ import annotations

from typing import Callable, Optional

from ..jobs import JobPlan
from .shapes import ShapeStore
from .spec import MeshSpec


def mesh_planner(runtime, shapes: Optional[ShapeStore] = None) -> Callable[[MeshSpec], JobPlan]:
    """Nothing of the runtime is held: a reload can swap it under a running planner."""
    def render(spec, token, progress, output):
        runtime.run(lambda controller: controller.render(spec, token, progress, output,
                                                          shapes))

    def plan(spec: MeshSpec) -> JobPlan:
        controller = runtime.pipeline
        if controller is None:
            raise RuntimeError("no mesh model is loaded")
        resolved = controller.resolve(spec)
        return JobPlan(kind="mesh", model=runtime.model_name, payload=resolved,
                       public=resolved.public(), suffix=f".{resolved.output_format}",
                       render=render)

    return plan


__all__ = ["mesh_planner"]
