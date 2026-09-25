"""Find the tennis ball in a broadcast clip and lift it into 3D.

The ball is a few pixels wide and is not on the ground plane, so neither the depth model nor the ground
intersection can place it. Between contacts it is a projectile, and that model is affine in its six unknowns,
so the whole 3D fit is a linear least-squares problem (see the POV ball spec §2).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

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


GRAVITY = np.array([0.0, 0.0, -9.81])

COURT_BOX_X = 20.0  # metres either side of the centre line; wider than this is not this rally
COURT_BOX_Y = 30.0
COURT_BOX_Z = 15.0


def ballistic_positions(a: np.ndarray, v: np.ndarray, t: np.ndarray) -> np.ndarray:
    """X(t) = a + v*t + 0.5*G*t^2 for each t, as (N, 3) world metres."""
    t = np.atleast_1d(np.asarray(t, dtype=np.float64))
    return a[None, :] + v[None, :] * t[:, None] + 0.5 * GRAVITY[None, :] * (t**2)[:, None]


def segment_times(frames: np.ndarray, fps: float) -> np.ndarray:
    """Seconds since the first observation. Every caller must derive t this way or the fit shifts."""
    frames = np.asarray(frames, dtype=np.int64)
    return (frames - frames[0]) / float(fps)


def fit_ballistic(camera, frames, uv, fps: float, iterations: int = 2):
    """Least-squares projectile through monocular observations. Returns (a, v).

    X(t) is affine in (a, v), so clearing the projection denominator gives two linear equations per
    observation (spec §2). That makes this one lstsq rather than a non-linear optimisation. Clearing the
    denominator minimises algebraic error, which over-weights distant observations, so the solve is repeated
    with w = 1/z from the previous estimate to approximate true reprojection error.
    """
    frames = np.asarray(frames, dtype=np.int64)
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    if len(frames) < 3:
        raise ValueError(f"a ballistic fit needs at least 3 observations, got {len(frames)}")
    t = segment_times(frames, fps)
    projection = camera.projection_matrix
    M, c = projection[:, :3], projection[:, 3]
    quadratic = 0.5 * GRAVITY[None, :] * (t**2)[:, None]  # (N, 3), the known part of X(t)
    known = quadratic @ M.T + c[None, :]  # (N, 3), k_i(t)

    rows = np.empty((2 * len(t), 6))
    rhs = np.empty(2 * len(t))
    for axis, numerator in enumerate((0, 1)):  # axis 0 is u against row 0, axis 1 is v against row 1
        coefficient = uv[:, axis][:, None] * M[2][None, :] - M[numerator][None, :]  # (N, 3)
        rows[axis::2, :3] = coefficient
        rows[axis::2, 3:] = coefficient * t[:, None]
        rhs[axis::2] = known[:, numerator] - uv[:, axis] * known[:, 2]

    weights = np.ones(2 * len(t))
    solution = np.zeros(6)
    for _ in range(iterations + 1):
        solution, _residual, rank, _singular = np.linalg.lstsq(rows * weights[:, None], rhs * weights, rcond=None)
        if rank < 6:
            raise ValueError(
                "rank-deficient fit: these observations do not determine a trajectory "
                "(the ball barely moves in the image)"
            )
        a, v = solution[:3], solution[3:]
        depth = ballistic_positions(a, v, t) @ M[2] + c[2]
        if not np.all(np.isfinite(depth)) or np.any(np.abs(depth) < 1e-6):
            break
        weights = np.repeat(1.0 / np.abs(depth), 2)
    return solution[:3], solution[3:]


def reprojection_residuals(camera, a: np.ndarray, v: np.ndarray, t: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Pixel distance between each observation and where the fitted trajectory says it should be."""
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    projected, _depth = camera.project(ballistic_positions(a, v, t))
    return np.linalg.norm(np.nan_to_num(projected, nan=1e9, posinf=1e9, neginf=1e9) - uv, axis=1)


def is_plausible(a: np.ndarray, v: np.ndarray, t: np.ndarray, max_speed_ms: float = 90.0) -> bool:
    """Reject fits that no tennis ball could produce, including a near-degenerate solve's wild output."""
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(v))):
        return False
    points = ballistic_positions(a, v, t)
    speeds = np.linalg.norm(v[None, :] + GRAVITY[None, :] * t[:, None], axis=1)
    return bool(
        np.all(np.isfinite(points))
        and np.all(points[:, 2] > -0.25)
        and np.all(points[:, 2] < COURT_BOX_Z)
        and np.all(np.abs(points[:, 0]) < COURT_BOX_X)
        and np.all(np.abs(points[:, 1]) < COURT_BOX_Y)
        and np.all(speeds < max_speed_ms)
    )


@dataclass(frozen=True)
class BallSegment:
    """One flight between contacts. `start` and `end` are inclusive frame indices."""

    start: int
    end: int
    a: np.ndarray
    v: np.ndarray


def split_flights(camera, frames, uv, fps: float, max_rms_px: float = 3.0, min_length: int = 4) -> list[BallSegment]:
    """Cut the tracklet wherever one parabola cannot explain it: those cuts are the bounces and racket hits.

    A span is accepted when a single fit explains it to within `max_rms_px` and is physically possible.
    Otherwise it is cut at its worst observation and both halves are retried. Spans that are too short, or
    that do not determine a trajectory at all, are dropped, so unexplained frames simply have no ball.
    """
    frames = np.asarray(frames, dtype=np.int64)
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    accepted: list[BallSegment] = []
    pending = [(0, len(frames))]
    while pending:
        low, high = pending.pop()
        if high - low < min_length:
            continue
        try:
            a, v = fit_ballistic(camera, frames[low:high], uv[low:high], fps)
        except ValueError:
            continue  # too few observations, or they do not determine a trajectory
        t = segment_times(frames[low:high], fps)
        residuals = reprojection_residuals(camera, a, v, t, uv[low:high])
        rms = float(np.sqrt(np.mean(residuals**2)))
        if rms <= max_rms_px and is_plausible(a, v, t):
            accepted.append(BallSegment(int(frames[low]), int(frames[high - 1]), a, v))
            continue
        if high - low < 2 * min_length:
            continue  # cannot cut into two halves that are both long enough to fit
        cut = int(np.clip(low + int(np.argmax(residuals)), low + min_length, high - min_length))
        pending.append((low, cut))
        pending.append((cut, high))
    return sorted(accepted, key=lambda s: s.start)


@dataclass
class BallTrack:
    fps: float
    frame_count: int
    xyz: np.ndarray  # (T, 3) world metres, NaN on frames with no ball
    reprojection_rms_px: float
    segments: list[tuple[int, int]]

    @property
    def detected_frames(self) -> int:
        return int(np.isfinite(self.xyz[:, 0]).sum())

    def position(self, frame: int) -> np.ndarray | None:
        if not 0 <= frame < len(self.xyz):
            return None
        point = self.xyz[frame]
        return None if not np.isfinite(point).all() else point

    def save(self, path: Path) -> None:
        data = {
            "fps": self.fps,
            "frame_count": self.frame_count,
            "reprojection_rms_px": round(self.reprojection_rms_px, 4),
            "segments": [{"start": s, "end": e} for s, e in self.segments],
            "xyz": [None if not np.isfinite(p).all() else [round(float(c), 4) for c in p] for p in self.xyz],
        }
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path: Path) -> "BallTrack":
        data = json.loads(Path(path).read_text())
        rows = [[np.nan] * 3 if p is None else [float(c) for c in p] for p in data["xyz"]]
        return cls(
            fps=float(data["fps"]),
            frame_count=int(data["frame_count"]),
            xyz=np.array(rows, dtype=np.float64).reshape(-1, 3),
            reprojection_rms_px=float(data["reprojection_rms_px"]),
            segments=[(int(s["start"]), int(s["end"])) for s in data["segments"]],
        )


def build_ball_track(
    camera,
    candidates: list[list[tuple[float, float]]],
    fps: float,
    frame_count: int,
    max_step_px: float = 180.0,
    max_gap: int = 3,
    min_length: int = 5,
    max_rms_px: float = 3.0,
) -> BallTrack:
    """Candidates -> one world position per frame, plus the segments and the fit quality."""
    xyz = np.full((frame_count, 3), np.nan)
    frames, uv = link_track(candidates, max_step_px=max_step_px, max_gap=max_gap, min_length=min_length)
    if len(frames) < 3:
        return BallTrack(fps=fps, frame_count=frame_count, xyz=xyz, reprojection_rms_px=0.0, segments=[])
    flights = split_flights(camera, frames, uv, fps, max_rms_px=max_rms_px)
    squared, observations = 0.0, 0
    for segment in flights:
        span = np.arange(segment.start, min(segment.end, frame_count - 1) + 1)
        xyz[span] = ballistic_positions(segment.a, segment.v, (span - segment.start) / float(fps))
        inside = (frames >= segment.start) & (frames <= segment.end)
        if inside.any():
            residuals = reprojection_residuals(
                camera, segment.a, segment.v, (frames[inside] - segment.start) / float(fps), uv[inside]
            )
            squared += float(np.sum(residuals**2))
            observations += int(inside.sum())
    return BallTrack(
        fps=fps,
        frame_count=frame_count,
        xyz=xyz,
        reprojection_rms_px=float(np.sqrt(squared / observations)) if observations else 0.0,
        segments=[(s.start, s.end) for s in flights],
    )
