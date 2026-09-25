"""Find the tennis ball in a broadcast clip and lift it into 3D.

The ball is a few pixels wide and is not on the ground plane, so neither the depth model nor the ground
intersection can place it. Between contacts it is a projectile, and that model is affine in its six unknowns,
so the whole 3D fit is a linear least-squares problem (see the POV ball spec §2).
"""

from __future__ import annotations

import cv2
import numpy as np

from horizon.players import Players
from horizon.tracking import Detections
from horizon.video import iter_frames

BALL_RADIUS_M = 0.0335  # ITF


def player_exclusion(detections: Detections, players: Players, frame: int, dilate_px: int = 21) -> np.ndarray:
    """Mask of the players, grown so a racket and the contact-frame ball fall inside it."""
    from horizon.reconstruct import player_detection  # local: keeps horizon.render free of this import chain

    union = np.zeros((detections.height, detections.width), np.uint8)
    for track in players.tracks.values():
        det = player_detection(detections, track, frame)
        if det is not None:
            union |= det.mask(detections.width, detections.height).astype(np.uint8)
    if dilate_px <= 1:
        return union.astype(bool)
    return cv2.dilate(union, np.ones((dilate_px, dilate_px), np.uint8)).astype(bool)


def detect_candidates(
    rgb: np.ndarray,
    plate: np.ndarray,
    exclude: np.ndarray,
    diff_threshold: int = 28,
    yellow_threshold: int = 25,
    min_area: int = 2,
    max_area: int = 120,
) -> list[tuple[float, float]]:
    """Small yellow-green blobs that differ from the clean plate: the ball, plus some noise.

    `G - B` is the discriminator. A blue hard court and its white lines both give values at or below zero,
    while a yellow ball gives a large positive one.
    """
    difference = np.abs(rgb.astype(np.int16) - plate.astype(np.int16)).sum(axis=2)
    yellowness = rgb[..., 1].astype(np.int16) - rgb[..., 2].astype(np.int16)
    mask = ((difference > diff_threshold) & (yellowness > yellow_threshold) & ~exclude).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    found = []
    for i in range(1, count):
        if min_area <= int(stats[i, cv2.CC_STAT_AREA]) <= max_area:
            found.append((float(centroids[i, 0]), float(centroids[i, 1])))
    return found


def link_track(
    candidates: list[list[tuple[float, float]]],
    max_step_px: float = 180.0,
    max_gap: int = 3,
    min_length: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Longest tracklet through the per-frame candidates, using a constant-velocity prediction.

    One clip is one rally, so the longest consistent chain is the ball. Candidates consumed by an accepted
    tracklet are not offered to later ones, which keeps this linear in the number of candidates.
    """
    used: list[set[int]] = [set() for _ in candidates]
    best_frames: list[int] = []
    best_points: list[np.ndarray] = []
    for start in range(len(candidates)):
        for index in range(len(candidates[start])):
            if index in used[start]:
                continue
            frames = [start]
            points = [np.asarray(candidates[start][index], dtype=np.float64)]
            picked = [(start, index)]
            misses = 0
            for frame in range(start + 1, len(candidates)):
                step = frame - frames[-1]
                if len(points) >= 2:
                    velocity = (points[-1] - points[-2]) / (frames[-1] - frames[-2])
                    gate = 0.5 * max_step_px * step  # the prediction is good, so allow half the raw travel
                else:
                    velocity = np.zeros(2)
                    gate = max_step_px * step
                predicted = points[-1] + velocity * step
                choice, best_distance = None, gate
                for j, uv in enumerate(candidates[frame]):
                    if j in used[frame]:
                        continue
                    distance = float(np.hypot(*(np.asarray(uv, dtype=np.float64) - predicted)))
                    if distance < best_distance:
                        choice, best_distance = j, distance
                if choice is None:
                    misses += 1
                    if misses > max_gap:
                        break
                    continue
                misses = 0
                frames.append(frame)
                points.append(np.asarray(candidates[frame][choice], dtype=np.float64))
                picked.append((frame, choice))
            if len(frames) >= min_length:
                for frame, j in picked:
                    used[frame].add(j)
                if len(frames) > len(best_frames):
                    best_frames, best_points = frames, points
    if not best_frames:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 2), dtype=np.float64)
    return np.array(best_frames, dtype=np.int64), np.array(best_points, dtype=np.float64).reshape(-1, 2)
