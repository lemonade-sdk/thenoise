import logging
import torch
from ..extract import Mesh

logger = logging.getLogger(__name__)


def drop_islands(mesh: Mesh, size: float) -> Mesh:
    """Every piece of ``mesh`` reaching at least ``size`` as far as the widest; 1 keeps that one."""
    if size <= 0.0 or mesh.is_empty:
        return mesh

    labels = _components(mesh.faces, mesh.vertices.shape[0])
    roots = torch.unique(labels)
    where = torch.searchsorted(roots, labels)
    piece = where[mesh.faces[:, 0]]
    corner = where[:, None].expand(-1, 3)
    low = mesh.vertices.new_full((roots.numel(), 3), float("inf"))
    high = mesh.vertices.new_full((roots.numel(), 3), float("-inf"))
    low.scatter_reduce_(0, corner, mesh.vertices, reduce="amin")
    high.scatter_reduce_(0, corner, mesh.vertices, reduce="amax")

    reach = (high - low).norm(dim=-1)
    farthest = float(reach.max())
    wanted = reach >= size * farthest
    if bool((~wanted[piece]).any()):
        area = mesh.face_areas()
        logger.info("floaters: dropped %d of %d pieces (under %d%% of the widest piece, "
                    "largest %d%%), %.2f%% of the area",
                    int((~wanted).sum()), len(roots), round(100 * size),
                    round(100 * float(reach[~wanted].max()) / farthest),
                    100 * float(area[~wanted[piece]].sum()) / float(area.sum()))
    else:
        logger.info("floaters: %d pieces, all reaching %d%% of the widest",
                    len(roots), round(100 * size))
    return mesh.select(wanted[where], wanted[piece])


#: One edge crossed per round, so a spiked cut needs the hand merge rather than more rounds.
ROUNDS = 32


def _components(faces: torch.Tensor, vertices: int) -> torch.Tensor:
    """One label per vertex, shared by everything reachable along an edge."""
    label = torch.arange(vertices, device=faces.device)
    corners = faces.reshape(-1)
    for _round in range(ROUNDS):
        before = label.clone()
        smallest = label[faces].amin(dim=1, keepdim=True).expand(-1, 3).reshape(-1)
        label.scatter_reduce_(0, corners, smallest, reduce="amin")
        label = label[label]
        if not bool(before.ne(label).any()):
            return label
    return _settled_by_hand(label, faces)


def _settled_by_hand(label: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Merge the labels the rounds left disagreeing, in the number of pieces rather than vertices."""
    a, b, c = (label[faces[:, i]] for i in (0, 1, 2))
    joined = faces[(a != b) | (b != c) | (c != a)]
    x, y, z = (label[joined[:, i]] for i in (0, 1, 2))
    pairs = torch.unique(torch.cat([torch.stack([x, y], 1), torch.stack([y, z], 1),
                                    torch.stack([z, x], 1)]), dim=0)
    logger.info("components: %d rounds did not settle it, %d label pairs merged by hand",
                ROUNDS, len(pairs))

    roots = torch.unique(label)
    parent = {int(one): int(one) for one in roots}

    def find(one: int) -> int:
        while parent[one] != one:
            parent[one] = parent[parent[one]]
            one = parent[one]
        return one

    for near, far in pairs.tolist():
        parent[find(near)] = find(far)
    return torch.tensor([find(int(one)) for one in roots],
                        device=label.device)[torch.searchsorted(roots, label)]
