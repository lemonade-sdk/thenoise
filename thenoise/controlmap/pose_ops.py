"""Scipy-free pose operations: the affine crop warp and DARK joint refinement that
the transformers ViTPose processor used scipy for, reimplemented in numpy/torch.
"""
from __future__ import annotations

import itertools
import math
from typing import Any, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers.image_processing_utils import BatchFeature

#: The VitPose input size and ImageNet normalisation the crop warp and refinement key off.
SIZE = {"height": 256, "width": 192}
NORMALIZE_FACTOR = 200.0
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def box_to_center_scale(
    box: Sequence[float],
    image_width: float,
    image_height: float,
    normalize_factor: float = NORMALIZE_FACTOR,
    padding_factor: float = 1.25,
) -> tuple[np.ndarray, np.ndarray]:
    """A COCO ``[x, y, w, h]`` box -> ``(center, scale)`` for the pose crop, with
    ``scale`` normalised by ``normalize_factor`` and padded."""
    top_left_x, top_left_y, width, height = box[:4]
    aspect_ratio = image_width / image_height
    center = np.array(
        [top_left_x + width * 0.5, top_left_y + height * 0.5], dtype=np.float32
    )
    if width > aspect_ratio * height:
        height = width * 1.0 / aspect_ratio
    elif width < aspect_ratio * height:
        width = height * aspect_ratio
    scale = np.array([width / normalize_factor, height / normalize_factor], dtype=np.float32)
    return center, scale * padding_factor


def warp_affine(
    image: np.ndarray, center: np.ndarray, scale: np.ndarray, size: Sequence[int]
) -> np.ndarray:
    """Bilinearly warp ``image`` (``H, W, C``) to ``size`` ``(width, height)``, the
    rotation-free affine crop the ViTPose processor applies."""
    matrix = get_warp_matrix(
        0.0, center * 2.0, np.array(size, dtype=np.float32) - 1.0, scale * NORMALIZE_FACTOR
    )
    return _affine_transform(image, matrix, (size[1], size[0]))


def get_warp_matrix(
    theta: float, size_input: np.ndarray, size_dst: np.ndarray, size_target: np.ndarray
) -> np.ndarray:
    """The unbiased affine transform matrix (rotation ``theta``, no shear)."""
    theta = np.deg2rad(theta)
    scale_x = size_dst[0] / size_target[0]
    scale_y = size_dst[1] / size_target[1]
    matrix = np.zeros((2, 3), dtype=np.float32)
    matrix[0, 0] = math.cos(theta) * scale_x
    matrix[0, 1] = -math.sin(theta) * scale_x
    matrix[0, 2] = scale_x * (
        -0.5 * size_input[0] * math.cos(theta)
        + 0.5 * size_input[1] * math.sin(theta)
        + 0.5 * size_target[0]
    )
    matrix[1, 0] = math.sin(theta) * scale_y
    matrix[1, 1] = math.cos(theta) * scale_y
    matrix[1, 2] = scale_y * (
        -0.5 * size_input[0] * math.sin(theta)
        - 0.5 * size_input[1] * math.cos(theta)
        + 0.5 * size_target[1]
    )
    return matrix


def _affine_transform(src: np.ndarray, matrix: np.ndarray, output_shape: Sequence[int]) -> np.ndarray:
    """``scipy.ndimage.affine_transform(src, M, output_shape, order=1)``."""
    homogeneous = np.vstack([matrix, [0.0, 0.0, 1.0]])
    inverse = np.linalg.inv(homogeneous)
    inverse[0, 0], inverse[0, 1], inverse[1, 0], inverse[1, 1] = (
        inverse[1, 1], inverse[1, 0], inverse[0, 1], inverse[0, 0],
    )
    inverse[0, 2], inverse[1, 2] = inverse[1, 2], inverse[0, 2]
    return _affine_sample(src, inverse[:2], output_shape)


def _affine_sample(src: np.ndarray, matrix: np.ndarray, output_shape: Sequence[int]) -> np.ndarray:
    """Bilinear sample ``src`` at ``matrix``-mapped output coordinates, 0 outside."""
    height, width, _ = src.shape
    out_rows, out_cols = output_shape
    rows, cols = np.meshgrid(np.arange(out_rows), np.arange(out_cols), indexing="ij")
    coords = np.stack([rows, cols, np.ones_like(rows)], axis=-1)
    y, x = (coords @ matrix.T)[..., 0], (coords @ matrix.T)[..., 1]
    return _bilinear_sample(src, y, x)


def _bilinear_sample(src: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    height, width, channels = src.shape
    y0, x0 = np.floor(y), np.floor(x)
    y1, x1 = y0 + 1, x0 + 1
    wy, wx = y - y0, x - x0

    def gather(gy, gx):
        gy = np.clip(gy, 0, height - 1).astype(np.int64)
        gx = np.clip(gx, 0, width - 1).astype(np.int64)
        return src[gy, gx]

    sampled = (
        gather(y0, x0) * (1 - wy)[..., None] * (1 - wx)[..., None]
        + gather(y0, x1) * (1 - wy)[..., None] * wx[..., None]
        + gather(y1, x0) * wy[..., None] * (1 - wx)[..., None]
        + gather(y1, x1) * wy[..., None] * wx[..., None]
    )
    inside = (y >= 0) & (y <= height - 1) & (x >= 0) & (x <= width - 1)
    return np.where(inside[..., None], sampled, 0.0)


def get_keypoint_predictions(heatmaps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Argmax joint locations and their scores from ``[B, K, H, W]`` heatmaps."""
    batch, keypoints, _, width = heatmaps.shape
    flat = heatmaps.reshape(batch, keypoints, -1)
    index = flat.argmax(dim=2).reshape(batch, keypoints, 1)
    scores = flat.max(dim=2).values.reshape(batch, keypoints, 1)
    preds = index.repeat(1, 1, 2).float()
    preds[:, :, 0] = preds[:, :, 0] % width
    preds[:, :, 1] = preds[:, :, 1] // width
    return torch.where(scores.repeat(1, 1, 2) > 0, preds, -1.0), scores


def post_dark(coords: torch.Tensor, heatmaps: torch.Tensor, kernel: int = 11) -> torch.Tensor:
    """DARK joint refinement: modulate the heatmap, then fit a parabola at each peak.
    The blur is torch; the fit stays in numpy to match scipy's numerics exactly."""
    batch, keypoints, height, width = heatmaps.shape
    radius = (kernel - 1) // 2
    coords_cpu = coords.detach().cpu()
    blurred = _gaussian_blur(heatmaps.detach().cpu(), 0.8, radius).numpy()
    blurred = np.log(np.clip(blurred, 0.001, 50))
    padded = np.pad(blurred, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="edge").flatten()
    index = (coords_cpu[..., 0] + 1 + (coords_cpu[..., 1] + 1) * (width + 2)).numpy()
    index += (width + 2) * (height + 2) * np.arange(batch * keypoints).reshape(-1, keypoints)
    index = index.astype(int).reshape(-1, 1)
    i_ = padded[index]
    ix1 = padded[index + 1]
    iy1 = padded[index + width + 2]
    ix1y1 = padded[index + width + 3]
    ix1_ = padded[index - 1]
    iy1_ = padded[index - 2 - width]
    ix1_y1_ = padded[index - width - 3]
    derivative = np.concatenate([0.5 * (ix1 - ix1_), 0.5 * (iy1 - iy1_)], axis=1)
    derivative = derivative.reshape(batch, keypoints, 2, 1)
    dxx = ix1 - 2 * i_ + ix1_
    dyy = iy1 - 2 * i_ + iy1_
    dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
    hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1).reshape(batch, keypoints, 2, 2)
    hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
    refined = coords_cpu.numpy() - np.einsum("ijmn,ijnk->ijmk", hessian, derivative).squeeze()
    return torch.tensor(refined, dtype=torch.float32)


def _gaussian_blur(x: torch.Tensor, sigma: float, radius: int) -> torch.Tensor:
    """Separable Gaussian blur matching ``scipy.ndimage.gaussian_filter`` (reflect edge)."""
    axis = torch.arange(-radius, radius + 1, dtype=x.dtype, device=x.device)
    kernel = torch.exp(-0.5 * (axis / sigma) ** 2)
    kernel = kernel / kernel.sum()
    kernel = (kernel[:, None] * kernel[None, :])[None, None]
    batch, keypoints, height, width = x.shape
    padded = _pad_edge_repeat(x.reshape(batch * keypoints, 1, height, width), radius)
    return F.conv2d(padded, kernel).reshape(batch, keypoints, height, width)


def _pad_edge_repeat(x: torch.Tensor, radius: int) -> torch.Tensor:
    """Reflect-pad the last two dims, repeating the edge pixel (scipy's ``reflect``)."""
    rows = torch.flip(x[..., :radius, :], dims=[-2])
    x = torch.cat([rows, x, torch.flip(x[..., -radius:, :], dims=[-2])], dim=-2)
    cols = torch.flip(x[..., :radius], dims=[-1])
    return torch.cat([cols, x, torch.flip(x[..., -radius:], dims=[-1])], dim=-1)


def transform_preds(
    coords: torch.Tensor, center: np.ndarray, scale: np.ndarray, output_size: Sequence[int]
) -> torch.Tensor:
    """Map a crop's heatmap peak back into the input image's pixels."""
    scale = scale * NORMALIZE_FACTOR
    scale_y = scale[1] / (output_size[0] - 1.0)
    scale_x = scale[0] / (output_size[1] - 1.0)
    target = torch.ones_like(coords)
    target[:, 0] = coords[:, 0] * scale_x + center[0] - scale[0] * 0.5
    target[:, 1] = coords[:, 1] * scale_y + center[1] - scale[1] * 0.5
    return target


def coco_to_pascal_voc(boxes: np.ndarray) -> np.ndarray:
    """COCO ``[x, y, w, h]`` -> Pascal VOC ``[x1, y1, x2, y2]`` corners."""
    boxes = boxes.copy()
    boxes[:, 2] = boxes[:, 2] + boxes[:, 0] - 1
    boxes[:, 3] = boxes[:, 3] + boxes[:, 1] - 1
    return boxes


class PoseProcessor:
    """Stands in for ``VitPoseImageProcessorPil``: ``__call__`` crops each box and
    warps it to the model's input, ``post_process_pose_estimation`` maps the peaks back."""

    def __call__(self, images: Any, boxes: Any, return_tensors: str = "pt"):
        if not isinstance(images, (list, tuple)):
            images = [images]
        transformed = []
        for image, image_boxes in zip(images, boxes):
            rgb = np.asarray(image, dtype=np.float32)
            for box in image_boxes:
                center, scale = box_to_center_scale(box, SIZE["width"], SIZE["height"])
                warped = warp_affine(rgb, center, scale, (SIZE["width"], SIZE["height"]))
                transformed.append((warped / 255.0 - MEAN) / STD)
        pixels = torch.tensor(np.stack(transformed, axis=0).transpose(0, 3, 1, 2))
        return BatchFeature({"pixel_values": pixels}, tensor_type=return_tensors)

    def post_process_pose_estimation(
        self, outputs: Any, boxes: Any, kernel_size: int = 11
    ) -> List[dict]:
        heatmaps = outputs.heatmaps
        batch_size, keypoints, height, width = heatmaps.shape
        flattened = list(itertools.chain(*boxes))
        centers = np.zeros((batch_size, 2), dtype=np.float32)
        scales = np.zeros((batch_size, 2), dtype=np.float32)
        for i in range(batch_size):
            center, scale = box_to_center_scale(flattened[i], SIZE["width"], SIZE["height"])
            centers[i], scales[i] = center, scale
        coords, scores = get_keypoint_predictions(heatmaps)
        coords = post_dark(coords, heatmaps, kernel=kernel_size)
        scores = scores.detach().cpu()
        for i in range(batch_size):
            coords[i] = transform_preds(coords[i], centers[i], scales[i], [height, width])
        boxes_xyxy = torch.tensor(coco_to_pascal_voc(np.concatenate([centers, scales], axis=1)))
        return [
            [
                {
                    "keypoints": coords[i],
                    "scores": scores[i].squeeze(-1),
                    "labels": torch.arange(keypoints),
                    "bbox": boxes_xyxy[i],
                }
                for i in range(batch_size)
            ]
        ]


__all__ = [
    "PoseProcessor",
    "box_to_center_scale",
    "coco_to_pascal_voc",
    "get_keypoint_predictions",
    "post_dark",
    "transform_preds",
    "warp_affine",
]
