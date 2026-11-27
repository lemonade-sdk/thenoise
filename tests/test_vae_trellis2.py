import pytest
import torch

from thenoise.sparse import SparseTensor
from thenoise.utils.safetensors import Header
from thenoise.vae.trellis2 import (SlatDecoder, _geometry, dual_grid_fields)

LATENT = 32
#: width, MLP hidden, NeXt blocks, landing width of the refinement, per level: as shipped.
LEVELS = ((1024, 4096, 4, 512), (512, 2048, 16, 256), (256, 1024, 8, 128), (128, 512, 4, 64))


def file_keys(out_channels: int, pred_subdiv: bool, prefix: str = "shape_dec.") -> dict:
    """The names and shapes a decoder's subtree carries, in the file's own conv layout."""
    keys = {f"{prefix}from_latent.weight": (LEVELS[0][0], LATENT),
            f"{prefix}from_latent.bias": (LEVELS[0][0],),
            f"{prefix}output_layer.weight": (out_channels, LEVELS[-1][3]),
            f"{prefix}output_layer.bias": (out_channels,)}
    for level, (width, hidden, count, up) in enumerate(LEVELS):
        for index in range(count):
            keys.update({
                f"{prefix}blocks.{level}.{index}.norm.weight": (width,),
                f"{prefix}blocks.{level}.{index}.norm.bias": (width,),
                f"{prefix}blocks.{level}.{index}.conv.weight": (width, 3, 3, 3, width),
                f"{prefix}blocks.{level}.{index}.conv.bias": (width,),
                f"{prefix}blocks.{level}.{index}.mlp.0.weight": (hidden, width),
                f"{prefix}blocks.{level}.{index}.mlp.0.bias": (hidden,),
                f"{prefix}blocks.{level}.{index}.mlp.2.weight": (width, hidden),
                f"{prefix}blocks.{level}.{index}.mlp.2.bias": (width,),
            })
        last = f"{prefix}blocks.{level}.{count}"
        keys.update({
            f"{last}.norm1.weight": (width,), f"{last}.norm1.bias": (width,),
            f"{last}.conv1.weight": (up * 8, 3, 3, 3, width),
            f"{last}.conv1.bias": (up * 8,),
            f"{last}.conv2.weight": (up, 3, 3, 3, up),
            f"{last}.conv2.bias": (up,),
        })
        if pred_subdiv:
            keys[f"{last}.to_subdiv.weight"] = (8, width)
            keys[f"{last}.to_subdiv.bias"] = (8,)
    return keys


def synthetic_header(keys: dict) -> Header:
    return Header("synthetic.safetensors", dict(keys),
                  {name: "BF16" for name in keys}, {}, {})


def test_the_decoder_is_the_shape_its_file_describes():
    from thenoise.vae.trellis2 import _decoder_state

    keys = file_keys(7, True)
    geometry = _geometry(synthetic_header(keys), "shape_dec.")
    with torch.device("meta"):
        model = SlatDecoder(**geometry)
        taken = _decoder_state("shape_dec.")(
            {name: torch.empty(shape) for name, shape in keys.items()}, model)
    want = {name: tuple(value.shape) for name, value in model.state_dict().items()}

    assert set(taken) == set(want)
    for name, value in taken.items():
        assert tuple(value.shape) == want[name], name


def test_a_texture_file_has_no_heads_to_predict_with():
    geometry = _geometry(synthetic_header(file_keys(6, False, "txt_dec.")), "txt_dec.")
    assert geometry["pred_subdiv"] is False
    assert geometry["out_channels"] == 6
    assert geometry["latent_channels"] == LATENT
    assert geometry["channels"] == [1024, 512, 256, 128]
    assert geometry["level_blocks"] == [4, 16, 8, 4]
    assert geometry["up_channels"] == [512, 256, 128, 64]


def small_decoder(pred_subdiv: bool) -> SlatDecoder:
    return SlatDecoder(latent_channels=4, channels=(64, 32), mlp_hidden=(128, 64),
                       level_blocks=(1, 1), up_channels=(32, 8), out_channels=6,
                       pred_subdiv=pred_subdiv)


def latent(rows: int = 24, grid: int = 4) -> SparseTensor:
    torch.manual_seed(0)
    cells = torch.cartesian_prod(torch.arange(grid), torch.arange(grid),
                                 torch.arange(grid)).int()
    coords = torch.cat([torch.zeros(len(cells), 1), cells], dim=1).to(torch.int32)
    if rows < len(cells):
        coords = coords[torch.randperm(len(cells))[:rows]]
    return SparseTensor(torch.randn(len(coords), 4), coords, (1, grid, grid, grid))


def test_a_decoder_refines_once_per_level_and_reads_out_per_voxel():
    out, subs = small_decoder(True).eval()(latent())
    assert out.feats.shape[1] == 6
    assert out.shape == (1, 16, 16, 16)
    assert len(subs) == 2
    assert all(sub.shape[1] == 8 for sub in subs)
    assert torch.isfinite(out.feats).all()


def test_a_guided_decoder_lands_on_the_lattice_the_shape_decoder_chose():
    x = latent()
    shape, subs = small_decoder(True).eval()(x)
    guided, again = small_decoder(False).eval()(x, guide_subs=subs)
    assert torch.equal(guided.coords, shape.coords)
    assert again == []


def test_a_decoder_without_heads_refuses_to_pick_its_own_lattice():
    with pytest.raises(ValueError, match="no subdivision heads"):
        small_decoder(False).eval()(latent())


def test_a_subdivision_of_negative_logits_bears_nothing():
    x = latent().replace(torch.randn(24, 64))
    dead = torch.full((24, 8), -1.0)
    born, _ = small_decoder(False).blocks[0][-1](x, subdiv=dead)
    assert len(born) == 0


def test_a_read_out_of_seven_is_a_dual_grid():
    feats = torch.randn(12, 7)
    vertices, intersected, split = dual_grid_fields(feats, voxel_margin=0.5)
    assert vertices.shape == (12, 3) and intersected.dtype == torch.bool
    assert vertices.min() > -0.5 and vertices.max() < 1.5
    assert (split >= 0).all()
