"""Pose maps: an RT-DETR detector finds the people, ViTPose measures their joints,
and :mod:`render` draws the skeleton. ViTPose is top-down, so a pose run needs both."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from thenoise.utils.image_tensor import flatten_alpha
from .base import PreprocessOptions, Preprocessor, require_checkpoint
from .pose_ops import PoseProcessor
from .render import CANVASES, skeleton_to_pil

#: RT-DETR's ``person`` is class 0 in both COCO checkpoints.
PERSON_CLASS = 0


@dataclass
class Person:
    """``[K, 2]`` normalised keypoints and their ``[K]`` scores; a culled joint keeps its place."""

    keypoints: torch.Tensor
    scores: torch.Tensor


@dataclass
class Pose:
    """The people a run measured, and the size of the image they were measured on."""

    size: Tuple[int, int]
    people: List[Person]


@dataclass
class PoseOptions(PreprocessOptions):
    """Which person detector to crop with, and how the skeleton is painted."""

    detector: str = ""  # a detector from the pool; "" is the first one
    threshold: float = 0.5
    kp_threshold: float = 0.3
    line_width: int = 4
    radius: int = 4
    background: str = "black"

    def __post_init__(self) -> None:
        for name in ("threshold", "kp_threshold"):
            score = getattr(self, name)
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {score}")
        if self.line_width < 1:
            raise ValueError(f"line_width must be >= 1, got {self.line_width}")
        if self.radius < 1:
            raise ValueError(f"radius must be >= 1, got {self.radius}")
        if self.background not in CANVASES:
            raise ValueError(
                f"unknown pose background {self.background!r}; "
                f"expected one of {tuple(CANVASES)}"
            )


class PosePreprocessor(Preprocessor):
    """ViTPose's COCO-17 joints over RT-DETR's person boxes, painted as COCO-18."""

    name: ClassVar[str] = "vitpose-base-simple"
    kind: ClassVar[str] = "pose"
    OPTIONS: ClassVar[type] = PoseOptions
    DTYPE: ClassVar[torch.dtype] = torch.bfloat16
    #: Deformable attention and the topk that picks the boxes are less happy reduced.
    DETECTOR_DTYPE: ClassVar[torch.dtype] = torch.float32
    DOWNLOAD: ClassVar[str] = "scripts/download.py --model preprocess --variant pose"

    def __init__(self) -> None:
        super().__init__()
        self._detectors: Dict[str, str] = {}
        self._detector_name: Optional[str] = None
        self._detector_model: Any = None
        self._detector_processor: Any = None

    def load(
        self,
        device: Any,
        path: Optional[str] = None,
        detectors: Optional[Dict[str, str]] = None,
    ) -> None:
        """Load the keypoint checkpoint ``path``; ``detectors`` is the pool of person
        detectors it may crop with, named by their directory names."""
        self.device = device
        self.name = os.path.basename(path) if path else self.name
        require_checkpoint(self, path)
        self._detectors = dict(detectors or {})
        self._processor = load_processor(path)
        self._model = load_model(path, self.DTYPE).to(device).eval()

    def unload(self) -> None:
        self._release_detector()
        del self._model
        del self._processor

    def detect(self, image: Image.Image, options: PoseOptions) -> Pose:
        """The people in ``image``, as normalised keypoints, measured on its geometry."""
        rgb = flatten_alpha(image)  # the processors want RGB, not a 4th channel
        boxes = self._person_boxes(rgb, options)
        if not len(boxes):
            return Pose(size=image.size, people=[])
        # One crop per person, with the same COCO [x, y, w, h] boxes on the way back.
        inputs = self._processor(images=rgb, boxes=[boxes], return_tensors="pt").to(
            self.device, self.DTYPE
        )
        with torch.no_grad():
            outputs = self._model(**inputs)
        outputs.heatmaps = outputs.heatmaps.float()  # numpy has no bf16
        detections = self._processor.post_process_pose_estimation(
            outputs, boxes=[boxes]
        )[0]
        # Pixels -> normalised: the last pixel's centre is the image size minus one.
        extent = torch.tensor(image.size, dtype=torch.float32) - 1.0
        return Pose(
            size=image.size,
            people=[
                Person(
                    keypoints=detection["keypoints"].float() / extent,
                    scores=_culled(detection["scores"].float(), options.kp_threshold),
                )
                for detection in detections
            ],
        )

    def render(self, pose: Pose, options: Optional[PoseOptions] = None) -> Image.Image:
        options = options or PoseOptions()
        config = self._model.config
        return skeleton_to_pil(
            pose.size,
            pose.people,
            edges=config.edges,
            labels=config.id2label,
            line_width=options.line_width,
            radius=options.radius,
            background=options.background,
        )

    def _person_boxes(self, image: Image.Image, options: PoseOptions) -> np.ndarray:
        """Absolute COCO ``[x, y, w, h]`` boxes for the people in ``image``."""
        model, processor = self._detector(options.detector)
        inputs = processor(images=image, return_tensors="pt").to(
            self.device, self.DETECTOR_DTYPE
        )
        with torch.no_grad():
            outputs = model(**inputs)
        detections = processor.post_process_object_detection(
            outputs,
            threshold=options.threshold,
            target_sizes=[image.size[::-1]],
        )[0]
        boxes = detections["boxes"][detections["labels"] == PERSON_CLASS]
        # RT-DETR answers in corners, ViTPose wants a corner and a size.
        boxes = torch.cat([boxes[:, :2], boxes[:, 2:] - boxes[:, :2]], dim=1)
        return boxes.cpu().numpy()

    def _detector(self, wanted: str):
        """The person detector's weights, loaded on first use and swapped on request."""
        if not self._detectors:
            raise ValueError(
                "pose needs a person detector beside the keypoint model; put one in the "
                f"preprocessor directory and run '{self.DOWNLOAD}' to fetch it"
            )
        available = sorted(self._detectors)
        name = wanted or available[0]
        if name not in self._detectors:
            raise ValueError(
                f"no person detector named {name!r}; available: {', '.join(available)}"
            )
        if name != self._detector_name:
            self._release_detector()
            path = self._detectors[name]
            require_checkpoint(self, path)
            detector = load_detector(path, self.DETECTOR_DTYPE)
            self._detector_model = detector.to(self.device).eval()
            self._detector_processor = load_processor(path)
            self._detector_name = name
        return self._detector_model, self._detector_processor

    def _release_detector(self) -> None:
        self._detector_model = None
        self._detector_processor = None
        self._detector_name = None


def _culled(scores: torch.Tensor, threshold: float) -> torch.Tensor:
    """Zero joints under ``threshold``: they keep their places so the renderer can skip them."""
    return scores * (scores > threshold)


def load_processor(path: str):
    """The processor a checkpoint needs: PoseProcessor for ViTPose, AutoImageProcessor for RT-DETR."""
    with open(os.path.join(path, "config.json"), encoding="utf-8") as f:
        model_type = json.load(f).get("model_type", "")
    if model_type == "vitpose":
        return PoseProcessor()
    from transformers import AutoImageProcessor

    return AutoImageProcessor.from_pretrained(path, backend="pil")


def load_model(path: str, dtype) -> torch.nn.Module:
    """The ViTPose keypoint model of a checkpoint directory."""
    from transformers import VitPoseForPoseEstimation

    return VitPoseForPoseEstimation.from_pretrained(path, dtype=dtype)


def load_detector(path: str, dtype) -> torch.nn.Module:
    """The RT-DETR person detector of a checkpoint directory."""
    from transformers import RTDetrForObjectDetection

    return RTDetrForObjectDetection.from_pretrained(path, dtype=dtype)


__all__ = [
    "Person",
    "Pose",
    "PoseOptions",
    "PosePreprocessor",
    "load_detector",
    "load_model",
    "load_processor",
]
