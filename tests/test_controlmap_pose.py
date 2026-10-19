"""Pose extraction: skeleton painting, the two-stage runner, and the crop convention.

The renderer is tested with hand-built keypoints — no weights, no network. The runner
runs on fakes. One golden test then pins the box convention the two halves of the
processor have to agree on: the crop it feeds the model, and the way back from a
heatmap peak to a joint in the input image.
"""
from __future__ import annotations

import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import BatchFeature

from thenoise.controlmap import pose as pose_module
from thenoise.controlmap.manager import ControlMapManager
from thenoise.controlmap.pose import Person, PoseOptions, PoseControlMap
from thenoise.controlmap.render import COCO_POSE_PALETTE, skeleton_edges, skeleton_to_pil
from thenoise.controlmap_controller import ControlMapController

#: The COCO-17 skeleton exactly as the ViTPose checkpoints declare it in their
#: ``config.json``, so the renderer is tested against the real topology.
COCO_LABELS = {
    str(index): name
    for index, name in enumerate(
        [
            "Nose", "L_Eye", "R_Eye", "L_Ear", "R_Ear", "L_Shoulder", "R_Shoulder",
            "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hip", "R_Hip", "L_Knee",
            "R_Knee", "L_Ankle", "R_Ankle",
        ]
    )
}
COCO_EDGES = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6],
    [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5],
    [4, 6],
]
JOINT = {name: int(index) for index, name in COCO_LABELS.items()}


def person(positions: dict, scores: dict | None = None) -> Person:
    """A COCO-17 person.

    ``positions`` maps joint name to normalised coordinates and every joint named in it
    is confident; a joint in ``scores`` takes that confidence instead, and a joint in
    neither is culled (score 0), which is how the runner reports one it did not believe.
    """
    keypoints = torch.zeros(len(JOINT), 2)
    confidence = torch.zeros(len(JOINT))
    for name, (x, y) in positions.items():
        keypoints[JOINT[name]] = torch.tensor([x, y])
    for name in positions:
        confidence[JOINT[name]] = 1.0
    for name, score in (scores or {}).items():
        confidence[JOINT[name]] = score
    return Person(keypoints=keypoints, scores=confidence)


def drawn_pixels(image: Image.Image) -> set:
    return set(image.get_flattened_data()) - {(0, 0, 0)}


# ------------------------------------------------------------------ options


def test_a_score_has_to_be_a_probability():
    with pytest.raises(ValueError, match=r"kp_threshold must be in \[0, 1\]"):
        PoseOptions(kp_threshold=1.5)
    with pytest.raises(ValueError, match=r"threshold must be in \[0, 1\]"):
        PoseOptions(threshold=-0.1)


def test_the_canvas_has_to_be_drawable():
    with pytest.raises(ValueError, match="line_width must be >= 1"):
        PoseOptions(line_width=0)
    with pytest.raises(ValueError, match="radius must be >= 1"):
        PoseOptions(radius=0)


def test_an_unknown_canvas_is_rejected_by_name():
    with pytest.raises(ValueError, match="unknown pose background 'purple'"):
        PoseOptions(background="purple")


# ----------------------------------------------------------------- renderer


def test_the_neck_is_derived_and_the_torso_and_head_rewired_through_it():
    """COCO-17 declares no neck; the openpose canvas has one at the shoulder midpoint.
    The shoulder line becomes two limbs meeting at it and the head hangs from it, which
    is what turns the checkpoint's 19 edges into the 20 the ControlNet was trained on."""
    edges, parents = skeleton_edges(COCO_EDGES, COCO_LABELS)
    assert parents == (5, 6)
    assert edges.count((5, 17)) == 1 and edges.count((17, 6)) == 1
    assert (5, 6) not in edges
    assert (17, 0) in edges  # head, through the neck
    assert len(edges) == len(COCO_EDGES) + 2


def test_the_derived_neck_cannot_collide_with_a_declared_joint():
    """The neck is a new joint, so it takes the index after the last one the checkpoint
    declares — even when that checkpoint declares its joints sparsely."""
    sparse = {"0": "Nose", "5": "L_Shoulder", "6": "R_Shoulder"}
    edges, _ = skeleton_edges([[0, 5], [5, 6]], sparse)
    assert {index for edge in edges for index in edge} <= {0, 5, 6, 7}
    assert (5, 7) in edges and (7, 6) in edges


def test_a_skeleton_that_declares_no_shoulders_is_drawn_as_declared():
    """A ViTPose checkpoint for another dataset — hands, animals — has no COCO shoulders
    to derive a neck from, and its own edges are still a perfectly good skeleton."""
    edges, parents = skeleton_edges(
        [[0, 1], [1, 2]], {"0": "wrist", "1": "elbow", "2": "shoulder"}
    )
    assert parents is None
    assert edges == [(0, 1), (1, 2)]


def test_a_joint_is_painted_at_its_own_fraction_of_the_image():
    """Joints are normalised when measured and scaled when painted, so a joint at the
    centre of the image is painted at the centre of the map, at whatever size that is."""
    nose = person({"Nose": (0.5, 0.5)})
    small = skeleton_to_pil((21, 11), [nose], edges=[], labels=COCO_LABELS, radius=1)
    large = skeleton_to_pil((41, 21), [nose], edges=[], labels=COCO_LABELS, radius=1)
    assert small.getpixel((10, 5)) == COCO_POSE_PALETTE[JOINT["Nose"]]
    assert large.getpixel((20, 10)) == COCO_POSE_PALETTE[JOINT["Nose"]]
    assert large.getpixel((0, 0)) == (0, 0, 0)


def test_the_map_is_the_size_of_the_image_it_was_measured_on():
    image = skeleton_to_pil((37, 13), [], edges=COCO_EDGES, labels=COCO_LABELS)
    assert image.size == (37, 13)


def test_the_canvas_colour_is_a_choice():
    """SD's original openpose controlmap painted the green canvas; everything newer,
    black. The map has to match whichever ControlNet consumes it."""
    def canvas(background):
        return skeleton_to_pil(
            (11, 9), [], edges=COCO_EDGES, labels=COCO_LABELS, background=background
        )

    assert canvas("black").getpixel((5, 1)) == (0, 0, 0)
    assert canvas("green").getpixel((5, 1)) == (36, 255, 12)


def test_the_torso_is_painted_through_the_derived_neck():
    """Two live shoulders and nothing else: what gets drawn is the shoulder line meeting
    at the midpoint between them, which only exists if the neck was derived."""
    shoulders = {"L_Shoulder": (0.2, 0.5), "R_Shoulder": (0.8, 0.5)}
    image = skeleton_to_pil(
        (41, 21), [person(shoulders)], edges=COCO_EDGES, labels=COCO_LABELS, radius=1
    )
    assert image.getpixel((20, 10)) != (0, 0, 0)  # the midpoint: the derived neck
    assert image.getpixel((20, 2)) == (0, 0, 0)  # nothing else was measured


def test_a_culled_shoulder_takes_the_neck_and_the_head_with_it():
    """The neck is a guess about two shoulders, so it is only as good as the worse one:
    a culled shoulder must remove the limbs through the neck, not invent a joint."""
    shoulders = {"L_Shoulder": (0.2, 0.5), "R_Shoulder": (0.8, 0.5)}

    def midpoint(scores):
        return skeleton_to_pil(
            (41, 21),
            [person(shoulders, scores)],
            edges=COCO_EDGES,
            labels=COCO_LABELS,
            radius=1,
        ).getpixel((20, 10))

    assert midpoint({}) != (0, 0, 0)
    assert midpoint({"R_Shoulder": 0.0}) == (0, 0, 0)


def test_a_culled_joint_loses_its_limbs_but_not_its_index():
    """An index always names the same joint, so culling silences a score rather than
    removing a row: the elbow goes, its limb with it, the surviving shoulder stays."""
    pair = {"L_Shoulder": (0.2, 0.5), "L_Elbow": (0.6, 0.5)}

    def drawn(scores):
        return skeleton_to_pil(
            (41, 21),
            [person(pair, scores)],
            edges=COCO_EDGES,
            labels=COCO_LABELS,
            radius=1,
        )

    whole = drawn({})
    culled = drawn({"L_Elbow": 0.0})
    # L_Shoulder at 0.2 of 40 is x=8; L_Elbow at 0.6 is x=24; both at y=10.
    assert whole.getpixel((8, 10)) != (0, 0, 0)  # the shoulder
    assert whole.getpixel((24, 10)) != (0, 0, 0)  # the elbow
    assert culled.getpixel((8, 10)) != (0, 0, 0)  # the joint that survived is still there
    assert culled.getpixel((24, 10)) == (0, 0, 0)  # and nothing painted where it was
    assert drawn_pixels(culled) == {COCO_POSE_PALETTE[JOINT["L_Shoulder"]]}


def test_limbs_are_painted_in_the_palette_and_as_thick_as_asked():
    live = Person(
        keypoints=torch.tensor([[0.1, 0.5], [0.9, 0.5]]),
        scores=torch.tensor([1.0, 1.0]),
    )
    thin = skeleton_to_pil(
        (41, 21), [live], edges=[[0, 1]], labels={}, line_width=1, radius=1
    )
    thick = skeleton_to_pil(
        (41, 21), [live], edges=[[0, 1]], labels={}, line_width=5, radius=1
    )
    assert thin.getpixel((20, 10)) == COCO_POSE_PALETTE[0]
    assert thin.getpixel((20, 8)) == (0, 0, 0)
    assert thick.getpixel((20, 8)) == COCO_POSE_PALETTE[0]


# ------------------------------------------------------------------- runner


class FakePoseProcessor:
    """Stands in for ViTPose's processor, keeping the real one's conventions: it is
    handed absolute COCO boxes, one list per image, and answers with joints in pixels."""

    def __init__(self):
        self.crop_boxes = None
        self.post_boxes = None
        self.heatmaps = None

    def __call__(self, images, boxes, return_tensors):
        self.crop_boxes = boxes
        return BatchFeature(
            {"pixel_values": torch.zeros(len(boxes[0]), 3, 256, 192)},
            tensor_type=return_tensors,
        )

    def post_process_pose_estimation(self, outputs, boxes):
        self.post_boxes = boxes
        self.heatmaps = outputs.heatmaps
        # The real one maps each crop's peak back to the centre of that crop's box, in
        # the input image's pixels, and returns one entry per person for one image.
        centres = boxes[0][:, :2] + boxes[0][:, 2:] / 2
        people = [
            {
                "keypoints": torch.tensor([[x, y]], dtype=torch.float32),
                "scores": torch.tensor([score], dtype=torch.float32),
            }
            for (x, y), score in zip(centres, [0.9, 0.2])
        ]
        return [people]


class FakeDetectorProcessor:
    """Stands in for RT-DETR's: corners in, corners out, plus the labels to filter on."""

    def __init__(self, boxes, labels):
        self.boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        self.labels = torch.tensor(labels).reshape(-1)
        self.target_sizes = None
        self.threshold = None

    def __call__(self, images, return_tensors):
        return BatchFeature(
            {"pixel_values": torch.zeros(1, 3, 64, 64)}, tensor_type=return_tensors
        )

    def post_process_object_detection(self, outputs, threshold, target_sizes):
        self.target_sizes = target_sizes
        self.threshold = threshold
        return [
            {
                "boxes": self.boxes,
                "labels": self.labels,
                "scores": torch.full((len(self.boxes),), 0.9),
            }
        ]


class FakePoseModel(torch.nn.Module):
    """One heatmap per crop, in the dtype the weights were loaded at."""

    def __init__(self, dtype):
        super().__init__()
        self.dtype = dtype
        self.crop_dtype = None
        # A single declared joint: enough to paint, and no neck to derive.
        self.config = type("Config", (), {"edges": [], "id2label": {"0": "Nose"}})()

    def forward(self, pixel_values, **kwargs):
        self.crop_dtype = pixel_values.dtype
        crops = pixel_values.shape[0]
        return SimpleNamespace(heatmaps=torch.zeros(crops, 1, 8, 6, dtype=self.dtype))


class FakeDetectorModel(torch.nn.Module):
    def forward(self, pixel_values, **kwargs):
        return SimpleNamespace(logits=None, pred_boxes=None)


def checkpoint_dir(tmp_path, name, model_type):
    directory = tmp_path / name
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps({"model_type": model_type}))
    for filename in ("model.safetensors", "preprocessor_config.json"):
        (directory / filename).write_text("{}")
    return directory


def pose_pool(tmp_path, monkeypatch, boxes=None, labels=None, detectors=("rtdetr_r18vd",)):
    """A controlmap pool holding a ViTPose estimator and ``detectors``, with the
    loaders replaced by fakes. Returns the pool plus what the fakes recorded."""
    estimator = FakePoseProcessor()
    detector = FakeDetectorProcessor(
        # Corners, as RT-DETR answers: two people, 30x20 and 20x25 pixels.
        [[10, 5, 40, 25], [40, 10, 60, 35]] if boxes is None else boxes,
        [0, 0] if labels is None else labels,
    )
    model = FakePoseModel(torch.bfloat16)
    detectors_loaded = []

    def load_processor(path):
        # One loader for both halves, exactly as the real one resolves by model_type.
        return detector if "rtdetr" in str(path) else estimator

    monkeypatch.setattr(pose_module, "load_processor", load_processor)
    monkeypatch.setattr(pose_module, "load_model", lambda path, dtype: model)
    monkeypatch.setattr(
        pose_module,
        "load_detector",
        lambda path, dtype: detectors_loaded.append(str(path)) or FakeDetectorModel(),
    )

    checkpoint_dir(tmp_path, "vitpose_base_simple", "vitpose")
    for name in detectors:
        checkpoint_dir(tmp_path, name, "rt_detr")

    manager = ControlMapManager(device="cpu", controlmap_dir=str(tmp_path))
    return type(
        "Rig",
        (),
        {
            "manager": manager,
            "controlmap": manager.get("pose"),
            "estimator": estimator,
            "detector": detector,
            "model": model,
            "detectors_loaded": detectors_loaded,
        },
    )


def test_a_pose_map_is_painted_at_the_input_size(tmp_path, monkeypatch):
    rig = pose_pool(tmp_path, monkeypatch)
    image = Image.new("RGB", (64, 40), "black")
    out = rig.controlmap.run(image, PoseOptions())
    assert (out.mode, out.size) == ("RGB", image.size)


def test_the_detector_answers_the_pose_model_in_coco_boxes(tmp_path, monkeypatch):
    """RT-DETR returns corners; ViTPose wants a corner and a size — and both halves of
    the keypoint model have to be handed the very same box."""
    rig = pose_pool(tmp_path, monkeypatch)
    rig.controlmap.detect(Image.new("RGB", (64, 40), "black"), PoseOptions())
    assert rig.estimator.crop_boxes[0].tolist() == [[10, 5, 30, 20], [40, 10, 20, 25]]
    assert rig.estimator.post_boxes[0] is rig.estimator.crop_boxes[0]
    assert rig.detector.target_sizes == [(40, 64)]  # (height, width) of the input
    assert rig.detector.threshold == PoseOptions().threshold


def test_only_people_get_a_skeleton(tmp_path, monkeypatch):
    """Class 2 is a bicycle, and a bicycle has no skeleton: drawing one would paint a
    person where none is standing."""
    rig = pose_pool(tmp_path, monkeypatch, boxes=[[10, 5, 40, 25]], labels=[2])
    detected = rig.controlmap.detect(
        Image.new("RGB", (64, 40), "black"), PoseOptions()
    )
    assert detected.people == []


def test_no_people_is_a_blank_canvas_of_the_input_size(tmp_path, monkeypatch):
    rig = pose_pool(tmp_path, monkeypatch, boxes=[], labels=[])
    image = Image.new("RGB", (16, 12), "black")
    blank = rig.controlmap.run(image, PoseOptions(background="green"))
    assert blank.size == image.size
    assert drawn_pixels(blank) == {(36, 255, 12)}


def test_joints_are_normalised_into_the_image(tmp_path, monkeypatch):
    """A map is drawn at whatever size it is painted at, so what is measured has to be a
    fraction of the image rather than a pixel in it."""
    rig = pose_pool(tmp_path, monkeypatch)
    detected = rig.controlmap.detect(
        Image.new("RGB", (64, 40), "black"), PoseOptions()
    )
    assert detected.size == (64, 40)
    # Box [10, 5, 30, 20] is centred at (25, 15) in a 64x40 image.
    assert detected.people[0].keypoints[0].tolist() == pytest.approx([25 / 63, 15 / 39])


def test_a_faint_joint_is_silenced_rather_than_removed(tmp_path, monkeypatch):
    """Index 5 is the left shoulder whatever else happened, so a joint the model was not
    sure about keeps its place with a score of 0."""
    rig = pose_pool(tmp_path, monkeypatch)
    detected = rig.controlmap.detect(
        Image.new("RGB", (64, 40), "black"), PoseOptions()
    )
    assert [float(one.scores[0]) for one in detected.people] == pytest.approx([0.9, 0.0])
    assert all(one.keypoints.shape[0] == 1 for one in detected.people)


def test_the_joint_cutoff_is_the_request_s(tmp_path, monkeypatch):
    rig = pose_pool(tmp_path, monkeypatch)
    image = Image.new("RGB", (64, 40), "black")
    strict = rig.controlmap.detect(image, PoseOptions(kp_threshold=0.95))
    assert [float(one.scores[0]) for one in strict.people] == [0.0, 0.0]


def test_the_heatmaps_are_handed_over_in_fp32(tmp_path, monkeypatch):
    """The joint refinement that follows is numpy work and numpy has no bf16, so the
    model's dtype must not be allowed to leak into it."""
    rig = pose_pool(tmp_path, monkeypatch)
    rig.controlmap.detect(Image.new("RGB", (64, 40), "black"), PoseOptions())
    assert rig.model.crop_dtype == torch.bfloat16  # the model itself still runs reduced
    assert rig.estimator.heatmaps.dtype == torch.float32


def test_the_person_detector_is_loaded_once_and_swapped_when_another_is_asked_for(
    tmp_path, monkeypatch
):
    rig = pose_pool(tmp_path, monkeypatch, detectors=("rtdetr_r18vd", "rtdetr_r50vd"))
    image = Image.new("RGB", (64, 40), "black")
    assert rig.detectors_loaded == []  # loading the estimator is not loading a detector
    rig.controlmap.detect(image, PoseOptions())
    assert rig.detectors_loaded == [str(tmp_path / "rtdetr_r18vd")]  # the first one
    rig.controlmap.detect(image, PoseOptions())
    assert len(rig.detectors_loaded) == 1  # and it stays resident between runs
    rig.controlmap.detect(image, PoseOptions(detector="rtdetr_r50vd"))
    assert rig.detectors_loaded[-1] == str(tmp_path / "rtdetr_r50vd")
    with pytest.raises(ValueError, match="no person detector named 'rtdetr_none'"):
        rig.controlmap.detect(image, PoseOptions(detector="rtdetr_none"))


def test_a_pose_map_is_self_describing(tmp_path, monkeypatch):
    """The PNG has to say what made it, like every other map."""
    rig = pose_pool(tmp_path, monkeypatch)
    controller = ControlMapController(rig.manager)
    out = controller.run(
        Image.new("RGB", (64, 40), "black"),
        "pose",
        PoseOptions(radius=2),
        "vitpose_base_simple",
    )
    buffer = io.BytesIO()
    out.save(buffer, format="PNG", pnginfo=out._pnginfo)
    buffer.seek(0)
    record = json.loads(Image.open(buffer).info["controlmap_data"])
    assert record == {
        "type": "pose",
        "model": "vitpose_base_simple",
        "options": {"radius": 2},
    }


# ------------------------------------------------------- the crop convention
#
# The riskiest detail of the two-stage pipeline: ViTPose's processor crops by a box and
# its post-processing maps a heatmap peak back through that box. If the boxes handed to
# the two halves meant different things, every skeleton would be drawn in the wrong
# place — quietly, and with plausible-looking numbers. These tests pin the convention
# the runner relies on, with the real processor and no weights.


@pytest.fixture
def vitpose_processor():
    from thenoise.controlmap.pose_ops import PoseProcessor

    return PoseProcessor()


def test_a_box_is_cropped_around_its_own_centre(vitpose_processor):
    """A point at the centre of a COCO box ends up at the centre of the model's crop,
    wherever in the image that box is."""
    for width, height, x, y in (
        (400, 200, 250, 70),
        (200, 400, 30, 350),
        (640, 480, 320, 240),
    ):
        pixels = np.zeros((height, width, 3), np.uint8)
        pixels[max(y - 2, 0) : y + 3, max(x - 2, 0) : x + 3] = 255
        crop = vitpose_processor(
            images=Image.fromarray(pixels),
            boxes=[[[x - 20, y - 30, 40, 60]]],
            return_tensors="pt",
        ).pixel_values[0]
        brightness = crop.float().mean(0)
        rows, columns = torch.nonzero(brightness > brightness.max() / 2, as_tuple=True)
        assert float(columns.float().mean()) == pytest.approx(95.5, abs=1.5)
        assert float(rows.float().mean()) == pytest.approx(127.5, abs=1.5)


def test_a_joint_at_the_centre_of_a_crop_is_the_centre_of_its_box(vitpose_processor):
    """The way back: the peak of a crop's heatmaps is the centre of that box, in the
    input image's pixels — which is what the runner then normalises."""
    from thenoise.controlmap.pose_ops import PoseProcessor

    boxes = [[230.0, 40.0, 40.0, 60.0], [10.0, 100.0, 30.0, 50.0]]
    rows, columns = torch.meshgrid(torch.arange(64.0), torch.arange(48.0), indexing="ij")
    peak = torch.exp(-(((rows - 31.5) ** 2 + (columns - 23.5) ** 2) / 4.0))
    # Every joint peaking at the crop centre, so every joint must land on the box centre.
    heatmaps = peak.repeat(2, 17, 1, 1)
    poses = vitpose_processor.post_process_pose_estimation(
        SimpleNamespace(heatmaps=heatmaps), boxes=[boxes]
    )[0]
    assert poses[0]["keypoints"].numpy() == pytest.approx(
        np.full((17, 2), [250.0, 70.0]), abs=0.5
    )
    assert poses[1]["keypoints"].numpy() == pytest.approx(
        np.full((17, 2), [25.0, 125.0]), abs=0.5
    )


#: The COCO-17 joint names, for the fake estimator's declared skeleton.
COCO_NAMES = [
    "Nose", "L_Eye", "R_Eye", "L_Ear", "R_Ear", "L_Shoulder", "R_Shoulder", "L_Elbow",
    "R_Elbow", "L_Wrist", "R_Wrist", "L_Hip", "R_Hip", "L_Knee", "R_Knee", "L_Ankle",
    "R_Ankle",
]


def test_the_two_stages_compose_into_joints_where_the_people_are(
    tmp_path, monkeypatch
):
    """The whole chain, with the real processors and fake weights: RT-DETR's boxes ->
    COCO boxes -> a crop per person -> heatmaps -> joints back in the input image ->
    normalised. Two people whose joints are all at their own centre must come out at
    their own centre, in the order the detector reported — a skeleton drawn on the wrong
    person is the failure this pipeline can produce most quietly."""
    from transformers.models.rt_detr.image_processing_pil_rt_detr import (
        RTDetrImageProcessorPil,
    )
    from thenoise.controlmap.pose_ops import PoseProcessor

    width, height = 320, 200
    corners = [[20.0, 30.0, 100.0, 190.0], [200.0, 40.0, 300.0, 160.0]]

    class Detector(torch.nn.Module):
        """Two confident people, in the normalised centre-and-size form RT-DETR emits."""

        def forward(self, pixel_values, **kwargs):
            logits = torch.full((1, len(corners), 80), -12.0)
            boxes = torch.zeros(1, len(corners), 4)
            for index, (x1, y1, x2, y2) in enumerate(corners):
                # Distinct scores: which of two equally confident boxes comes first is
                # the detector's business, and the joints follow whichever order it picks.
                logits[0, index, 0] = 8.0 - index
                boxes[0, index] = torch.tensor(
                    [
                        (x1 + x2) / 2 / width, (y1 + y2) / 2 / height,
                        (x2 - x1) / width, (y2 - y1) / height,
                    ]
                )
            return SimpleNamespace(logits=logits, pred_boxes=boxes)

    class Estimator(torch.nn.Module):
        """Every joint peaking at the middle of its crop."""

        config = type(
            "Config",
            (),
            {
                "edges": [[5, 6], [0, 1]],
                "id2label": {str(i): n for i, n in enumerate(COCO_NAMES)},
            },
        )()

        def forward(self, pixel_values, **kwargs):
            rows, columns = torch.meshgrid(
                torch.arange(64.0), torch.arange(48.0), indexing="ij"
            )
            peak = torch.exp(-(((rows - 31.5) ** 2 + (columns - 23.5) ** 2) / 4.0))
            return SimpleNamespace(
                heatmaps=torch.zeros(
                    pixel_values.shape[0], len(COCO_NAMES), 64, 48
                )
                + peak
            )

    checkpoint_dir(tmp_path, "vitpose_base_simple", "vitpose")
    checkpoint_dir(tmp_path, "rtdetr_r18vd", "rt_detr")
    monkeypatch.setattr(
        pose_module,
        "load_processor",
        lambda path: (
            PoseProcessor() if "vitpose" in str(path) else RTDetrImageProcessorPil()
        ),
    )
    monkeypatch.setattr(pose_module, "load_model", lambda path, dtype: Estimator())
    monkeypatch.setattr(pose_module, "load_detector", lambda path, dtype: Detector())

    pre = PoseControlMap()
    pre.load(
        "cpu",
        str(tmp_path / "vitpose_base_simple"),
        detectors={"rtdetr_r18vd": str(tmp_path / "rtdetr_r18vd")},
    )
    image = Image.new("RGB", (width, height), "black")
    detected = pre.detect(image, PoseOptions())

    assert len(detected.people) == len(corners)
    extent = np.array([width, height]) - 1
    centres = np.array([[(x1 + x2) / 2, (y1 + y2) / 2] for x1, y1, x2, y2 in corners])
    # Which person comes first is the detector's choice, so the check is that every joint
    # sits within a pixel of the centre it was cropped from, and that both centres are
    # occupied: not one person drawn twice, and none drawn on the other's body.
    for person in detected.people:
        error = np.abs(
            person.keypoints.numpy()[:, None] * extent - centres
        ).max(axis=2).min(axis=1)
        assert (error < 1.0).all()
    for centre in centres:
        nearest = min(
            float(np.abs(person.keypoints[0].numpy() * extent - centre).max())
            for person in detected.people
        )
        assert nearest < 1.0
    assert pre.render(detected, PoseOptions()).size == image.size
