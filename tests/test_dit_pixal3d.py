import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from thenoise.dit.pixal3d import (
    PIXAL_SIGNATURE,
    Pixal3DStage,
    Pixal3DStageParams,
    load_pixal3d_stage,
    rope_matrices,
    stage_params,
    stage_prefix,
    stages_in,
)
from thenoise.utils.rope import apply_rope
from thenoise.utils.safetensors import as_header

MINI = Pixal3DStageParams(
    hidden_size=12, depth=2, heads=2, head_dim=6, mlp_dim=24, in_channels=4, out_channels=4,
    cond_channels=8, proj_channels=10, timestep_features=8, shared_modulation=True,
)


def stage(stage_name: str, **overrides) -> Pixal3DStage:
    params = Pixal3DStageParams(**{**MINI.__dict__, **overrides})
    torch.manual_seed(0)
    return Pixal3DStage(params)


def coords(tokens: int) -> torch.Tensor:
    return torch.arange(tokens).repeat(3, 1).t().contiguous()


#: The module puts the QK gains on the shared slots; a file still names them the old way.
CKPT_NAMES = {".qk_norm.query_norm.weight": ".q_rms_norm.gamma",
              ".qk_norm.key_norm.weight": ".k_rms_norm.gamma"}


def checkpoint_state(model: Pixal3DStage, stage_name: str) -> dict:
    state = {}
    for key, value in model.state_dict().items():
        for shared, legacy in CKPT_NAMES.items():
            key = key.replace(shared, legacy)
        state[stage_prefix(stage_name) + key] = value
    return state


def test_params_round_trip_through_a_state_dict():
    built = stage("shape")
    shapes = {key: tuple(value.shape) for key, value in checkpoint_state(built, "shape").items()}
    assert stage_params(shapes, "shape") == MINI


def test_the_texture_stage_reads_a_second_latent_in_alongside():
    texture = stage("texture", in_channels=2 * MINI.in_channels)
    shapes = {key: tuple(value.shape) for key, value in checkpoint_state(texture, "texture").items()}
    assert stage_params(shapes, "texture").in_channels == 2 * MINI.in_channels


def test_an_unsizable_file_is_refused():
    shapes = {key: tuple(value.shape) for key, value in checkpoint_state(stage("shape"), "shape").items()}
    shapes.pop(stage_prefix("shape") + "input_layer.weight")
    with pytest.raises(ValueError, match="input_layer"):
        stage_params(shapes, "shape")


def test_one_file_carrying_two_stages_is_read_as_two():
    shapes = {}
    for name, width in (("structure", 10), ("texture", 20)):
        shapes |= {key: tuple(value.shape) for key, value in
                   checkpoint_state(stage(name, proj_channels=width,
                                          in_channels=8 if name == "texture" else 4), name).items()}
    assert stages_in(shapes) == ["structure", "texture"]
    assert stage_params(shapes, "texture").proj_channels == 20
    assert all(key in shapes for key in PIXAL_SIGNATURE.keys)


def test_rope_gives_each_axis_its_own_rotation():
    one_axis, other_axis = rope_matrices(torch.tensor([[3, 0, 0], [0, 3, 0]]), MINI.head_dim)[0]
    assert not torch.allclose(one_axis, other_axis)


def test_a_head_wider_than_the_axes_leaves_the_spare_pairs_unrotated():
    matrices = rope_matrices(torch.tensor([[5, 4, 2]]), head_dim=8)
    assert matrices.shape == (1, 1, 4, 2, 2)
    q, k = torch.randn(1, 2, 1, 8), torch.randn(1, 2, 1, 8)
    rotated_q, rotated_k = apply_rope(q, k, matrices)
    assert torch.equal(rotated_q[..., 6:], q[..., 6:])
    assert torch.equal(rotated_k[..., 6:], k[..., 6:])


def test_a_stage_runs_and_keeps_its_final_norm_in_fp32():
    built = stage("shape")
    out = built(torch.randn(2, 6, 4), coords(6), torch.rand(2),
                torch.randn(2, 5, 8), torch.randn(2, 6, 10))
    assert out.shape == (2, 6, 4)
    assert torch.isfinite(out).all()

    # The norm before the output layer is unaffine, so it owns no parameters.
    assert not any("out_norm" in key for key in built.state_dict())
    assert built.out_layer.out_features == MINI.out_channels


def test_loading_one_stage_out_of_a_shared_file(tmp_path):
    tensors = {}
    for name, width, channels in (("structure", 10, MINI.in_channels),
                                  ("texture", 20, 2 * MINI.in_channels)):
        built = stage(name, proj_channels=width, in_channels=channels)
        tensors |= checkpoint_state(built, name)
    path = tmp_path / "pixal3d_stages.safetensors"
    save_file(tensors, str(path))
    assert as_header(path).identifies(PIXAL_SIGNATURE)

    loaded = load_pixal3d_stage(str(path), "structure", device="cpu", dtype=torch.bfloat16)
    assert loaded.params == MINI
    assert all(value.dtype == torch.bfloat16 for value in loaded.state_dict().values())
    out = loaded(*(x.bfloat16() for x in (torch.randn(1, 3, MINI.in_channels), coords(3),
                                          torch.rand(1), torch.randn(1, 4, MINI.cond_channels),
                                          torch.randn(1, 3, MINI.proj_channels))))
    assert out.dtype == torch.bfloat16 and out.shape == (1, 3, 4)
