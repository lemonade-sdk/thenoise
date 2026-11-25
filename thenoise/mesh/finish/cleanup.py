import torch

from ..extract import Mesh


def drop_islands(mesh: Mesh, share: float) -> Mesh:
    """Every piece of ``mesh`` holding at least ``share`` of its area; 0 keeps all."""
    if share <= 0.0 or mesh.is_empty:
        return mesh

    faces, corners = mesh.faces, mesh.vertices[mesh.faces]
    area = torch.linalg.cross(corners[:, 1] - corners[:, 0],
                              corners[:, 2] - corners[:, 0]).norm(dim=-1) * 0.5
    labels = _components(faces, mesh.vertices.shape[0])
    piece = labels[faces[:, 0]]
    totals = torch.zeros(int(labels.max()) + 1, dtype=area.dtype, device=area.device)
    totals.scatter_reduce_(0, piece, area, reduce="sum")
    whole = float(totals.sum())
    wanted = totals >= share * whole

    if not bool(wanted[piece].any()):
        raise ValueError(f"every piece of the cut holds less than {share} of its area: "
                         f"there would be no shape left to write")

    keep = wanted[labels]
    lookup = torch.full_like(labels, -1)
    lookup[keep] = torch.arange(int(keep.sum()), device=labels.device)
    return Mesh(mesh.vertices[keep], lookup[faces[wanted[piece]]])


def _components(faces: torch.Tensor, vertices: int) -> torch.Tensor:
    """One label per vertex, shared by everything reachable along an edge."""
    label = torch.arange(vertices, device=faces.device)
    corners = faces.reshape(-1)
    while True:
        before = label.clone()
        smallest = label[faces].amin(dim=1, keepdim=True).expand(-1, 3).reshape(-1)
        label.scatter_reduce_(0, corners, smallest, reduce="amin")
        label = label[label]
        if torch.equal(label, before):
            return label
