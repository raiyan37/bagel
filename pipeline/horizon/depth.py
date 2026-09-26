"""Monocular depth (Depth Anything V2) and its metric alignment to the calibrated court plane.

Depth Anything V2 relative checkpoints predict affine-invariant disparity (larger = closer). On pixels that
certainly show the court, the calibrated camera tells us the true depth, so we fit
scale * disparity + shift = 1 / depth robustly and apply it to the whole frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from horizon.camera import PinholeCamera
from horizon.court import court_rect, net_quad

DEFAULT_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


class DepthEstimator(Protocol):
    def __call__(self, rgb: np.ndarray) -> np.ndarray: ...


class DepthAnythingV2:
    """Hugging Face `depth-estimation` pipeline wrapper returning full-resolution disparity."""

    def __init__(self, model_id: str = DEFAULT_DEPTH_MODEL, device: str | None = None):
        import torch
        from transformers import pipeline

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._pipe = pipeline(task="depth-estimation", model=model_id, device=device)

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        from PIL import Image

        predicted = self._pipe(Image.fromarray(rgb))["predicted_depth"]
        if hasattr(predicted, "detach"):
            predicted = predicted.detach().float().cpu().numpy()
        disparity = np.squeeze(np.asarray(predicted, dtype=np.float32))
        h, w = rgb.shape[:2]
        if disparity.shape != (h, w):
            disparity = cv2.resize(disparity, (w, h), interpolation=cv2.INTER_LINEAR)
        return disparity


def pixel_grid(width: int, height: int, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    vs, us = np.mgrid[0:height:stride, 0:width:stride]
    return us.astype(np.float64), vs.astype(np.float64)


def ground_depth_map(camera: PinholeCamera) -> np.ndarray:
    us, vs = pixel_grid(camera.width, camera.height)
    _, depth = camera.ground_intersection(us.ravel(), vs.ravel())
    return depth.reshape(camera.height, camera.width)


def _fill(mask: np.ndarray, camera: PinholeCamera, polygon: np.ndarray, value: int) -> None:
    uv, z = camera.project(polygon)
    if (z > 0).all():
        cv2.fillPoly(mask, [np.round(uv).astype(np.int32).reshape(-1, 1, 2)], value)


def court_fit_mask(
    camera: PinholeCamera, exclude: np.ndarray | None = None, margin_x: float = 2.0, margin_y: float = 3.0
) -> np.ndarray:
    """Pixels that certainly show the court surface: court + run-off, minus the net and `exclude` (e.g. players)."""
    mask = np.zeros((camera.height, camera.width), np.uint8)
    _fill(mask, camera, court_rect(margin_x, margin_y), 1)
    _fill(mask, camera, net_quad(), 0)
    result = mask.astype(bool)
    if exclude is not None:
        result &= ~exclude
    return result


@dataclass(frozen=True)
class DisparityAlignment:
    scale: float
    shift: float
    residual: float  # RMS residual (1/m) on inliers

    def depth(self, disparity: np.ndarray, max_depth: float = 120.0) -> np.ndarray:
        inverse = self.scale * np.asarray(disparity, dtype=np.float64) + self.shift
        return 1.0 / np.maximum(inverse, 1.0 / max_depth)


def fit_disparity_alignment(
    disparity: np.ndarray,
    target_depth: np.ndarray,
    mask: np.ndarray,
    max_samples: int = 20000,
    inlier_tol: float = 0.1,
    iterations: int = 200,
    min_inlier_fraction: float = 0.3,
    seed: int = 0,
) -> DisparityAlignment:
    """Robust fit of scale * disparity + shift = 1 / depth on the mask pixels.

    RANSAC over pixel pairs (positive scales only) finds the consensus line; a pixel is an inlier when its
    predicted inverse depth is within `inlier_tol` (relative) of the truth. Least squares on the consensus set,
    one re-selection of inliers with the refined line and a final refit give the alignment.
    """
    sel = mask & np.isfinite(target_depth) & (target_depth > 0)
    d = disparity[sel].astype(np.float64)
    y = 1.0 / target_depth[sel].astype(np.float64)
    if d.size < 50:
        raise ValueError(f"Only {d.size} ground pixels available for depth alignment (need >= 50)")
    rng = np.random.default_rng(seed)
    if d.size > max_samples:
        idx = rng.choice(d.size, max_samples, replace=False)
        d, y = d[idx], y[idx]
    not_disparity = ValueError(
        "Depth model output does not behave like disparity (no positive-scale fit explains the court); "
        "use a relative Depth Anything V2 checkpoint"
    )
    best = None
    for _ in range(iterations):
        i, j = rng.choice(d.size, 2, replace=False)
        if d[i] == d[j]:
            continue
        scale = (y[i] - y[j]) / (d[i] - d[j])
        if scale <= 0:
            continue
        inliers = np.abs(scale * d + (y[i] - scale * d[i]) - y) <= inlier_tol * y
        if best is None or inliers.sum() > best.sum():
            best = inliers
    if best is None or best.mean() < min_inlier_fraction:
        raise not_disparity
    A = np.stack([d, np.ones_like(d)], axis=1)
    coef, *_ = np.linalg.lstsq(A[best], y[best], rcond=None)
    inliers = np.abs(A @ coef - y) <= inlier_tol * y
    if inliers.sum() >= 2:
        coef, *_ = np.linalg.lstsq(A[inliers], y[inliers], rcond=None)
    else:
        inliers = best
    scale, shift = float(coef[0]), float(coef[1])
    if scale <= 0:
        raise not_disparity
    rms = float(np.sqrt(np.mean((A[inliers] @ coef - y[inliers]) ** 2)))
    return DisparityAlignment(scale=scale, shift=shift, residual=rms)


def anchor_player_depth(depth: np.ndarray, mask: np.ndarray, foot_depth: float, band: float = 0.6) -> np.ndarray:
    """Depths of the mask pixels, shifted so the lowest 10 % of rows sit at foot_depth, clamped to +-band m."""
    values = depth[mask].astype(np.float64)
    rows = np.nonzero(mask.any(axis=1))[0]
    if rows.size == 0:
        return values
    top, bottom = int(rows[0]), int(rows[-1])
    cutoff = bottom - max(1, int(round(0.1 * (bottom - top + 1)))) + 1
    feet = mask.copy()
    feet[:cutoff] = False
    offset = foot_depth - float(np.median(depth[feet]))
    return np.clip(values + offset, foot_depth - band, foot_depth + band)
