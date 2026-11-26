from __future__ import annotations
import numpy as np
import torch
from ..extract import Mesh


def reduce_to(mesh: Mesh, keep: float) -> Mesh:
    """``mesh`` with ``keep`` of its triangles."""
    if keep >= 1.0 or mesh.is_empty:
        return mesh

    import fast_simplification

    local = mesh.cpu()
    points, faces = fast_simplification.simplify(
        local.vertices.to(torch.float64).numpy(),
        local.faces.to(torch.int32).numpy(),
        target_count=max(4, round(len(local.faces) * keep)),
    )
    return Mesh(
        torch.as_tensor(points, dtype=mesh.vertices.dtype, device=mesh.vertices.device),
        torch.as_tensor(np.asarray(faces, dtype=np.int64), dtype=mesh.faces.dtype,
                        device=mesh.faces.device),
    )


__all__ = ["reduce_to"]
