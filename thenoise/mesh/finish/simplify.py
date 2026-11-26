from __future__ import annotations
import numpy as np
import torch
from ..extract import Mesh

# A tetrahedron is the least a closed shape can have.
TETRAHEDRON = 4


def reduce_to(mesh: Mesh, keep: float) -> Mesh:
    """``mesh`` with ``keep`` of its triangles."""
    cut = mesh.faces.shape[0]
    # Nothing can come off a tetrahedron without opening it, and the reducer refuses a
    # target at or above the count it is given -- which rounding a fraction does.
    if keep >= 1.0 or cut <= TETRAHEDRON:
        return mesh

    import fast_simplification

    local = mesh.cpu()
    points, faces = fast_simplification.simplify(
        local.vertices.to(torch.float64).numpy(),
        local.faces.to(torch.int32).numpy(),
        target_count=min(cut - 1, max(TETRAHEDRON, round(cut * keep))),
    )
    return Mesh(
        torch.as_tensor(points, dtype=mesh.vertices.dtype, device=mesh.vertices.device),
        torch.as_tensor(np.asarray(faces, dtype=np.int64), dtype=mesh.faces.dtype,
                        device=mesh.faces.device),
    )


__all__ = ["reduce_to"]
