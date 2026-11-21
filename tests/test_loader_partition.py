"""Reading one checkpoint into several components: partition the file, fill a module."""
from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file

from thenoise.utils.loader import load_selected_state, partition_keys
from thenoise.utils.safetensors import read_header


def _saved(tmp_path, tensors, name="one.safetensors"):
    save_file({key: tensor.clone() for key, tensor in tensors.items()}, str(tmp_path / name))
    return read_header(str(tmp_path / name))


def _routing(**owners):
    """A ``part_of`` routing keys by prefix, as in ``part_of(dit="model.", vae="vae.")``."""

    def route(raw):
        for name, prefix in owners.items():
            if raw.startswith(prefix):
                return name, raw[len(prefix):]
        return None, None

    return route


def test_one_file_answers_several_parts(tmp_path):
    header = _saved(tmp_path, {"model.weight": torch.ones(2, 2), "vae.weight": torch.ones(3, 1)})
    picked, rejected = partition_keys(
        header, {"dit": {"weight": (2, 2)}, "vae": {"weight": (3, 1)}},
        part_of=_routing(dit="model.", vae="vae."))
    assert picked == {"dit": {"model.weight": "weight"}, "vae": {"vae.weight": "weight"}}
    assert not any(rejected.values())


def test_a_part_the_request_named_is_held_to_fitting_it(tmp_path):
    """Shape, name and duplicate faults refuse the file; found parts only report."""
    wrong_shape = _saved(tmp_path, {"vae.weight": torch.ones(9, 9)})
    with pytest.raises(ValueError, match="shape"):
        partition_keys(wrong_shape, {"vae": {"weight": (3, 1)}},
                       part_of=_routing(vae="vae."), strict=frozenset({"vae"}))
    picked, rejected = partition_keys(wrong_shape, {"vae": {"weight": (3, 1)}},
                                      part_of=_routing(vae="vae."))
    assert picked["vae"] == {}
    assert rejected["vae"] == {"vae.weight": "weight"}

    wrong_name = _saved(tmp_path, {"vae.encoder.gone.weight": torch.ones(1)})
    with pytest.raises(ValueError, match="unexpected vae tensor"):
        partition_keys(wrong_name, {"vae": {"weight": (1,)}},
                       part_of=_routing(vae="vae."), strict=frozenset({"vae"}))

    twice = _saved(tmp_path, {"vae.a.weight": torch.ones(1), "vae.b.weight": torch.ones(1)})
    with pytest.raises(ValueError, match="duplicate vae weight"):
        partition_keys(twice, {"vae": {"weight": (1,)}},
                       part_of=lambda raw: ("vae", "weight") if raw.endswith("weight") else (None, None),
                       strict=frozenset({"vae"}))


def test_quantization_metadata_follows_the_weight_it_describes(tmp_path):
    header = _saved(tmp_path, {"dit.layer.weight": torch.ones(2, 2, dtype=torch.int8),
                               "dit.layer.weight_scale": torch.ones(2, 1)})
    picked, rejected = partition_keys(header, {"dit": {"layer.weight": (2, 2)}},
                                      part_of=_routing(dit="dit."))
    assert picked["dit"] == {"dit.layer.weight": "layer.weight",
                             "dit.layer.weight_scale": "layer.weight_scale"}
    assert not rejected["dit"]

    no_such_weight = _saved(tmp_path, {"dit.other.weight_scale": torch.ones(2, 1)},
                            "orphan.safetensors")
    picked, rejected = partition_keys(no_such_weight, {"dit": {"layer.weight": (2, 2)}},
                                      part_of=_routing(dit="dit."))
    assert picked["dit"] == {}
    assert rejected["dit"] == {"dit.other.weight_scale": "other.weight_scale"}
    with pytest.raises(ValueError, match="quantization metadata"):
        partition_keys(no_such_weight, {"dit": {"layer.weight": (2, 2)}},
                       part_of=_routing(dit="dit."), strict=frozenset({"dit"}))


def test_a_part_is_filled_from_several_files_but_filled_once(tmp_path):
    weights = tmp_path / "weights.safetensors"
    save_file({"a.weight": torch.ones(2, 2)}, str(weights))
    biases = tmp_path / "biases.safetensors"
    save_file({"b.bias": torch.zeros(2)}, str(biases))
    with torch.device("meta"):
        module = torch.nn.Linear(2, 2)
    load_selected_state(module, [(str(weights), {"a.weight": "weight"}),
                                 (str(biases), {"b.bias": "bias"})],
                        device="cpu", dtype=torch.float32, what="dit")
    torch.testing.assert_close(module.weight, torch.ones(2, 2))
    torch.testing.assert_close(module.bias, torch.zeros(2))

    again = tmp_path / "again.safetensors"
    save_file({"a.weight": torch.zeros(2, 2)}, str(again))
    with torch.device("meta"):
        module = torch.nn.Linear(2, 2)
    with pytest.raises(ValueError, match="duplicate dit weight"):
        load_selected_state(module, [(str(weights), {"a.weight": "weight"}),
                                     (str(again), {"a.weight": "weight"})],
                            device="cpu", dtype=torch.float32, what="dit")


def test_a_part_that_ends_up_short_names_the_files_that_came_up_short(tmp_path):
    path = tmp_path / "half.safetensors"
    save_file({"weight": torch.ones(2, 2)}, str(path))
    with torch.device("meta"):
        module = torch.nn.Linear(2, 2)
    with pytest.raises(ValueError, match=r"leave missing=\['bias'\]"):
        load_selected_state(module, [(str(path), {"weight": "weight"})],
                            device="cpu", dtype=torch.float32, what="vae")


def test_the_keys_it_was_given_are_the_keys_it_reads(tmp_path):
    """A file carrying more than this component still fills it."""
    path = tmp_path / "crowded.safetensors"
    save_file({"want.weight": torch.full((2, 2), 3.), "bias": torch.ones(2),
               "other.giant": torch.ones(4, 4)}, str(path))
    with torch.device("meta"):
        module = torch.nn.Linear(2, 2, bias=False)
    load_selected_state(module, [(str(path), {"want.weight": "weight"})],
                        device="cpu", dtype=torch.float32, what="vae")
    torch.testing.assert_close(module.weight, torch.full((2, 2), 3.))


def test_int8_without_its_scale_is_refused(tmp_path):
    from thenoise.dit.quantized import QuantizedLinear

    path = tmp_path / "missing_scale.safetensors"
    save_file({"layer.weight": torch.ones(2, 4, dtype=torch.int8)}, str(path))
    with torch.device("meta"):
        module = torch.nn.ModuleDict({"layer": QuantizedLinear(4, 2, bias=False)})
    with pytest.raises(RuntimeError, match="missing its"):
        load_selected_state(module, [(str(path), {"layer.weight": "layer.weight"})],
                            device="cpu", dtype=torch.bfloat16, what="dit")


def test_buffers_and_parameters_both_arrive(tmp_path):
    path = tmp_path / "buffered.safetensors"
    save_file({"weight": torch.ones(3, 2), "bias": torch.ones(3),
               "running": torch.full((3,), 5.)}, str(path))
    with torch.device("meta"):
        module = torch.nn.Linear(2, 3)
        module.register_buffer("running", torch.zeros(3))
    load_selected_state(module, [(str(path), {"weight": "weight", "bias": "bias",
                                              "running": "running"})],
                        device="cpu", dtype=torch.float32, what="dit")
    assert not any(x.is_meta for x in (*module.parameters(), *module.buffers()))
    torch.testing.assert_close(module(torch.ones(2)), torch.tensor([3., 3., 3.]))
    torch.testing.assert_close(module.running, torch.full((3,), 5.))
