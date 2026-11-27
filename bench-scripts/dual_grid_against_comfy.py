"""The same structured latent through our shape decoder and mesher, and ComfyUI's.

The other half lives in ComfyUI, which is not a dependency of this repository: point the script
at a checkout — one line, so no continuation can be lost in a paste.

    .venv/bin/python bench-scripts/dual_grid_against_comfy.py --comfy /path/to/ComfyUI

ComfyUI's compiled memory pool is invented rather than installed, because these decoders never
reach it. TRELLIS.2 support is recent: an old checkout has nothing to compare against, and the
script says so rather than comparing against nothing.

Both sides get one deterministic latent, so a difference is the model's or the extraction's,
not the input's. Positions are compared by voxel index, which only lines up when both sides
grew the lattice row for row, and again as surface points, so a reordered lattice shows up as
a chamfer distance instead of a crash.
"""
from __future__ import annotations

import argparse
import importlib.abc
import importlib.machinery
import pathlib
import sys
import types

import torch


def ours(path: str, device: torch.device, dtype: torch.dtype, coords: torch.Tensor,
         feats: torch.Tensor, grid: int):
    import torch.nn.functional as functional

    from thenoise.mesh import dual_grid_mesh
    from thenoise.sparse import SparseTensor
    from thenoise.vae import dual_grid_fields, load_trellis2_shape_vae

    decoder = load_trellis2_shape_vae(path, device, dtype=dtype)
    lattice = SparseTensor(feats, coords, (1, grid, grid, grid))
    with torch.no_grad():
        # The level walk, spelled out rather than called, so each level's lattice is in hand:
        # the reference hands back its subdivisions with coordinates, and only a walk can say
        # which level the two stop agreeing on.
        h = lattice.replace(decoder.from_latent(lattice.feats))
        subs, levels = [], []
        for level in decoder.blocks:
            levels.append(h.coords.long())
            for block in level[:-1]:
                h = block(h)
            h, subdiv = level[-1](h)
            subs.append(subdiv)
        read_out = h.replace(decoder.output_layer(functional.layer_norm(
            h.feats, h.feats.shape[-1:])))
        whole, whole_subs = decoder(lattice)
    walked = float((read_out.feats.float() - whole.feats.float()).abs().max())
    print(f"our decoder's own loop leaves the same read-out: {walked:.1e}")
    vertex, crossed, split = dual_grid_fields(read_out.feats)
    mesh = dual_grid_mesh(read_out, vertex, crossed, split)[0]
    del decoder
    return mesh, subs, levels


def _rows(coords: torch.Tensor) -> set:
    return {tuple(row) for row in coords.tolist()}


def _reference():
    """ComfyUI's own shape decoder, imported with its memory pool stubbed out."""
    try:
        import comfy_aimdo  # noqa: F401
    except ImportError:
        sys.meta_path.append(_Fabricator())

    from comfy.ldm.trellis2.vae import ShapeVae, SparseTensor
    return ShapeVae, SparseTensor


class _Fabricator(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Hands out any ``comfy_aimdo`` module, and those invent any attribute.

    ComfyUI imports its compiled memory pool at module scope; the decoders do not touch it, so
    the comparison does not need it installed.
    """

    def find_spec(self, name, path=None, target=None):
        if name != "comfy_aimdo" and not name.startswith("comfy_aimdo."):
            return None
        return importlib.machinery.ModuleSpec(name, self, is_package=True)

    def create_module(self, spec):
        made = _Invented(spec.name)
        made.__path__ = []
        return made

    def exec_module(self, module):
        return None


class _Invented(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        made = _Invented(f"{self.__name__}.{name}")
        setattr(self, name, made)
        return made

    def __call__(self, *args, **kwargs):
        return None


def theirs(path: str, device: torch.device, dtype: torch.dtype, coords: torch.Tensor,
           feats: torch.Tensor, grid: int):
    from safetensors.torch import load_file

    ShapeVae, SparseTensor = _reference()

    model = ShapeVae()
    state = load_file(path)
    missing = [key for key in state if key.startswith("shape_dec.")
               and key not in model.state_dict()]
    if missing:
        raise SystemExit(f"the reference model has no {missing[:3]} to load into")
    model.load_state_dict(state, strict=False)
    model.to(device, dtype).eval()

    slat = SparseTensor(feats=feats.clone(), coords=coords.clone())
    with torch.no_grad():
        mesh, subs = model.decode_shape_slat(slat, grid * 16)
    vertex, faces = mesh[0]
    del model
    return vertex.float(), faces.long(), [(sub.feats, sub.coords.long()) for sub in subs]


def vertex_gap(mine: torch.Tensor, theirs: torch.Tensor, within: float,
               chunk: int = 2048) -> tuple:
    """Every vertex to the nearest vertex the other side has.

    Both sides carry a vertex per voxel, so this is the comparison that means something. Two
    sampled point clouds would measure their own spacing instead: mean nearest distance between
    N points on one surface is about sqrt(area / N), which at 20 000 samples and area 0.77 is
    6e-3 — a whole voxel of noise floor, and larger than any difference worth reporting.
    """
    def one_way(source: torch.Tensor, target: torch.Tensor):
        near = torch.empty(source.shape[0], device=source.device)
        for start in range(0, source.shape[0], chunk):
            part = source[start:start + chunk]
            # The default cdist goes through a matmul, which reports a point's distance to
            # *itself* as up to 7e-4 by cancellation — the floor has to sit well under the gap.
            near[start:start + chunk] = torch.cdist(
                part, target, compute_mode="donot_use_mm_for_euclid_dist").min(dim=-1).values
        return near

    forward, backward = one_way(mine, theirs), one_way(theirs, mine)
    return (float(forward.mean()), float(forward.max()),
            int((forward > within).sum()), int((backward > within).sum()))


def main() -> int:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("vae", nargs="?",
                        default="models/pixal3d/vae/trellis_2_shape_vae_bf16.safetensors",
                        help="the shape VAE file, read by both sides")
    parser.add_argument("--grid", type=int, default=8, help="the latent's own grid")
    parser.add_argument("--voxels", type=int, default=300, help="how many carry a latent")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--comfy", default=None,
                        help="a ComfyUI checkout to read the reference decoders from")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16")
    arguments = parser.parse_args()

    if not arguments.vae.strip() or not pathlib.Path(arguments.vae).is_file():
        parser.error(f"no shape VAE file at {arguments.vae!r}")

    if arguments.comfy:
        sys.path.insert(0, str(pathlib.Path(arguments.comfy).resolve()))
    device = torch.device(arguments.device)
    dtype = getattr(torch, arguments.dtype)
    generator = torch.Generator(device=device).manual_seed(arguments.seed)
    cells = torch.cartesian_prod(*[torch.arange(arguments.grid, device=device)] * 3)
    chosen = torch.randperm(cells.shape[0], device=device,
                            generator=generator)[:arguments.voxels]
    coords = torch.cat([torch.zeros(len(chosen), 1, device=device), cells[chosen]],
                       dim=1).to(torch.int32)
    feats = torch.randn(len(chosen), 32, device=device, dtype=dtype, generator=generator)
    print(f"{len(coords)} voxels of a {arguments.grid}^3 grid, in [-0.5, 0.5], {dtype}")

    from thenoise.mesh import Mesh

    mine, my_subs, my_levels = ours(arguments.vae, device, dtype, coords, feats,
                                    arguments.grid)
    try:
        their_vertices, their_faces, their_subs = theirs(arguments.vae, device, dtype, coords,
                                                     feats, arguments.grid)
    except ImportError as missing:
        raise SystemExit(f"{missing}: pass --comfy a checkout recent enough to carry "
                         f"comfy/ldm/trellis2") from missing
    theirs_mesh = Mesh(their_vertices.to(mine.vertices.device),
                       their_faces.to(mine.faces.device))

    for name, left, right in (("faces", mine.faces.shape[0], theirs_mesh.faces.shape[0]),
                              ("vertices", len(mine.vertices), len(theirs_mesh.vertices))):
        print(f"{name:<10}  ours {left:>9d}   theirs {right:>9d}")
    print(f"area        ours {float(mine.face_areas().sum()):>9.5f}   "
          f"theirs {float(theirs_mesh.face_areas().sum()):>9.5f}")
    print(f"volume      ours {float(mine.signed_volume()):>9.5f}   "
          f"theirs {float(theirs_mesh.signed_volume()):>9.5f}")

    if mine.vertices.shape == theirs_mesh.vertices.shape:
        gap = (mine.vertices - theirs_mesh.vertices).abs().max()
        print(f"faces identical by index: {torch.equal(mine.faces, theirs_mesh.faces)}; "
              f"positions agree to {float(gap):.2e}")

    # Where the two ladders part company. A level whose lattice is the same rows in the same
    # order can be compared value for value: a wrong conv layout or child order shows up there
    # as a delta no rounding explains, while accumulated noise stays at a few units of bf16.
    for level, (ours_here, (theirs_logits, theirs_here)) in enumerate(zip(my_levels,
                                                                          their_subs)):
        ours_here, theirs_here = ours_here.cpu(), theirs_here.cpu()
        same = (ours_here.shape == theirs_here.shape
                and torch.equal(ours_here, theirs_here))
        if not same:
            shared = len(_rows(ours_here) & _rows(theirs_here))
            print(f"level {level}: lattice {len(ours_here)} vs {len(theirs_here)} voxels, "
                  f"{shared} of them the same — the logits are no longer comparable by row")
            continue
        mine_here = my_subs[level].float().cpu()
        theirs_here_logits = theirs_logits.float().cpu()
        gap = (mine_here - theirs_here_logits).abs()
        flipped = int(((mine_here > 0) != (theirs_here_logits > 0)).sum())
        # The scale matters as much as the gap: 0.25 on a logit of 20 is rounding, on a logit
        # of 1 it is not. All eight channels differing by the same large amount says the
        # subdivision columns are ordered differently; one channel says a head is misread.
        print(f"level {level}: {len(ours_here)} voxels alike; |logit| mean "
              f"{float(mine_here.abs().mean()):.2f}, |gap| mean {float(gap.mean()):.2f} "
              f"({float(gap.mean()) / max(float(mine_here.abs().mean()), 1e-9):.1%} of it), "
              f"max {float(gap.max()):.2f}; {flipped} of {gap.numel()} cross zero")
        print(f"  per channel |gap|: " + " ".join(f"{float(v):.2f}" for v in gap.mean(0)))

    voxel = 1.0 / float(arguments.grid * 16)
    mean, worst, missing, extra = vertex_gap(mine.vertices, theirs_mesh.vertices, voxel)
    print(f"vertices to nearest vertex: mean {mean:.2e}, worst {worst:.2e}; "
          f"{missing} of ours and {extra} of theirs have no counterpart within a voxel "
          f"({voxel:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
