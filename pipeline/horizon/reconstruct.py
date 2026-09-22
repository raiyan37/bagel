"""Point clouds from the clip: one static background cloud plus per-frame player clouds.

Pipeline: masked-median clean plate -> Depth Anything disparity aligned to the calibrated court plane ->
pixels near the plane become the ground texture, the rest become background points. Every frame, each player's
mask pixels are lifted with that frame's aligned depth, anchored so their feet sit at the calibrated foot depth.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from horizon.camera import PinholeCamera, camera_from_dict, camera_to_dict
from horizon.depth import (
    DepthEstimator,
    DisparityAlignment,
    anchor_player_depth,
    court_fit_mask,
    fit_disparity_alignment,
    ground_depth_map,
    pixel_grid,
)
from horizon.ground import DEFAULT_EXTENT, GroundTexture, build_ground_texture
from horizon.players import PlayerTrack, Players
from horizon.tracking import Detection, Detections
from horizon.video import iter_frames

ROLE_CODES = {"near": 0, "far": 1}


@dataclass(frozen=True, eq=False)
class Scene:
    background_points: np.ndarray  # (N, 3) float32 world
    background_colors: np.ndarray  # (N, 3) uint8
    background_radii: np.ndarray  # (N,) float32 world footprint radius in metres
    player_points: np.ndarray  # (M, 3) float32, all frames concatenated
    player_colors: np.ndarray  # (M, 3) uint8
    player_radii: np.ndarray  # (M,) float32
    player_roles: np.ndarray  # (M,) uint8, see ROLE_CODES
    frame_offsets: np.ndarray  # (T + 1,) int64: frame i owns [offsets[i], offsets[i + 1])
    ground: GroundTexture
    camera: PinholeCamera  # the calibrated broadcast camera
    fps: float

    @property
    def frame_count(self) -> int:
        return len(self.frame_offsets) - 1

    def players_at(self, frame: int, exclude_role: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        start, end = int(self.frame_offsets[frame]), int(self.frame_offsets[frame + 1])
        points = self.player_points[start:end]
        colors = self.player_colors[start:end]
        radii = self.player_radii[start:end]
        if exclude_role is not None:
            keep = self.player_roles[start:end] != ROLE_CODES[exclude_role]
            points, colors, radii = points[keep], colors[keep], radii[keep]
        return points, colors, radii

    def save(self, path: Path) -> None:
        cam = camera_to_dict(self.camera)
        np.savez_compressed(
            path,
            background_points=self.background_points,
            background_colors=self.background_colors,
            background_radii=self.background_radii,
            player_points=self.player_points,
            player_colors=self.player_colors,
            player_radii=self.player_radii,
            player_roles=self.player_roles,
            frame_offsets=self.frame_offsets,
            ground_image=self.ground.image,
            ground_extent=np.array([self.ground.x_min, self.ground.x_max, self.ground.y_min, self.ground.y_max]),
            ground_fill=np.array(self.ground.fill, dtype=np.uint8),
            camera_K=np.array(cam["K"]),
            camera_R=np.array(cam["R"]),
            camera_t=np.array(cam["t"]),
            camera_size=np.array([cam["width"], cam["height"]]),
            fps=np.array(self.fps),
        )

    @classmethod
    def load(cls, path: Path) -> "Scene":
        with np.load(path) as d:
            x_min, x_max, y_min, y_max = (float(v) for v in d["ground_extent"])
            ground = GroundTexture(d["ground_image"], x_min, x_max, y_min, y_max, tuple(int(v) for v in d["ground_fill"]))
            width, height = (int(v) for v in d["camera_size"])
            camera = camera_from_dict(
                {"K": d["camera_K"], "R": d["camera_R"], "t": d["camera_t"], "width": width, "height": height}
            )
            return cls(
                background_points=d["background_points"],
                background_colors=d["background_colors"],
                background_radii=d["background_radii"],
                player_points=d["player_points"],
                player_colors=d["player_colors"],
                player_radii=d["player_radii"],
                player_roles=d["player_roles"],
                frame_offsets=d["frame_offsets"],
                ground=ground,
                camera=camera,
                fps=float(d["fps"]),
            )


def median_plate(frames: list[np.ndarray], masks: list[np.ndarray], dilate_px: int = 9) -> np.ndarray:
    """Per-pixel median over frames, ignoring (dilated) player pixels; pixels never visible are inpainted."""
    kernel = np.ones((dilate_px, dilate_px), np.uint8)
    stack = []
    for frame, mask in zip(frames, masks):
        grown = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
        values = frame.astype(np.float32)
        values[grown] = np.nan
        stack.append(values)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN pixels are handled below
        plate = np.nanmedian(np.stack(stack), axis=0)
    holes = np.isnan(plate[..., 0])
    plate = np.nan_to_num(plate, nan=0.0).clip(0, 255).astype(np.uint8)
    if holes.any():
        plate = cv2.inpaint(plate, holes.astype(np.uint8), 5, cv2.INPAINT_TELEA)
    return plate


def player_detection(detections: Detections, track: PlayerTrack, frame: int) -> Detection | None:
    track_id = track.track_ids[frame] if frame < len(track.track_ids) else None
    return None if track_id is None else detections.by_track(frame, track_id)


def _player_masks(detections: Detections, players: Players, frame: int) -> dict[str, tuple[Detection, np.ndarray]]:
    out = {}
    for role, track in players.tracks.items():
        det = player_detection(detections, track, frame)
        if det is not None:
            out[role] = (det, det.mask(detections.width, detections.height))
    return out


def background_cloud(
    plate: np.ndarray,
    disparity: np.ndarray,
    camera: PinholeCamera,
    ground_depth: np.ndarray,
    extent: tuple[float, float, float, float],
    stride: int = 2,
    ground_tolerance: float = 0.12,
    max_depth: float = 120.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, DisparityAlignment]:
    fit_mask = court_fit_mask(camera)
    alignment = fit_disparity_alignment(disparity, ground_depth, fit_mask)
    depth = alignment.depth(disparity, max_depth)
    x_min, x_max, y_min, y_max = extent
    us, vs = pixel_grid(camera.width, camera.height)
    hits, _ = camera.ground_intersection(us.ravel(), vs.ravel())
    gx = hits[:, 0].reshape(depth.shape)
    gy = hits[:, 1].reshape(depth.shape)
    in_extent = np.isfinite(ground_depth) & (gx >= x_min) & (gx <= x_max) & (gy >= y_min) & (gy <= y_max)
    with np.errstate(invalid="ignore"):
        near_plane = np.abs(depth - ground_depth) <= ground_tolerance * ground_depth
    ground = (in_extent & near_plane) | fit_mask
    keep = np.zeros_like(ground)
    keep[::stride, ::stride] = True
    keep &= ~ground & (depth < max_depth * 0.999)
    v_idx, u_idx = np.nonzero(keep)
    z = depth[v_idx, u_idx]
    points = camera.backproject(u_idx, v_idx, z).astype(np.float32)
    radii = (0.5 * stride * z / camera.fy).astype(np.float32)
    return points, plate[v_idx, u_idx], radii, ground, alignment


def reconstruct_scene(
    video_path: Path,
    camera: PinholeCamera,
    players: Players,
    detections: Detections,
    estimator: DepthEstimator,
    stride: int = 2,
    samples: int = 24,
    extent: tuple[float, float, float, float] = DEFAULT_EXTENT,
    resolution: float = 0.025,
    log: Callable[[str], None] | None = None,
) -> Scene:
    log = log or (lambda _msg: None)
    total = players.frame_count
    sample_ids = set(np.linspace(0, total - 1, min(samples, total)).round().astype(int).tolist())
    frames, masks = [], []
    for index, rgb in iter_frames(video_path, stop=total):
        if index in sample_ids:
            frames.append(rgb)
            player_masks = _player_masks(detections, players, index)
            union = np.zeros(rgb.shape[:2], bool)
            for _, mask in player_masks.values():
                union |= mask
            masks.append(union)
    plate = median_plate(frames, masks)
    log(f"clean plate from {len(frames)} frames")

    ground_depth = ground_depth_map(camera)
    bg_points, bg_colors, bg_radii, ground_mask, alignment = background_cloud(
        plate, estimator(plate), camera, ground_depth, extent, stride=stride
    )
    ground = build_ground_texture(plate, camera, ground_mask, extent, resolution)
    log(f"background: {len(bg_points)} points, alignment residual {alignment.residual:.2e} 1/m")

    erode = np.ones((3, 3), np.uint8)
    grow = np.ones((9, 9), np.uint8)
    pts_list, col_list, rad_list, role_list = [], [], [], []
    offsets = [0]
    for index, rgb in iter_frames(video_path, stop=total):
        player_masks = _player_masks(detections, players, index)
        count = 0
        if player_masks:
            disparity = estimator(rgb)
            union = np.zeros(rgb.shape[:2], np.uint8)
            for _, mask in player_masks.values():
                union |= mask.astype(np.uint8)
            exclude = cv2.dilate(union, grow).astype(bool)
            depth = fit_disparity_alignment(disparity, ground_depth, court_fit_mask(camera, exclude=exclude)).depth(disparity)
            for role, (det, mask) in player_masks.items():
                inner = cv2.erode(mask.astype(np.uint8), erode).astype(bool)
                if inner.sum() < 10:
                    continue
                _, foot_depth = camera.ground_intersection(np.array([det.foot[0]]), np.array([det.foot[1]]))
                if not np.isfinite(foot_depth[0]):
                    continue
                values = anchor_player_depth(depth, inner, float(foot_depth[0]))
                v_idx, u_idx = np.nonzero(inner)
                pts_list.append(camera.backproject(u_idx, v_idx, values).astype(np.float32))
                col_list.append(rgb[v_idx, u_idx])
                rad_list.append((0.5 * values / camera.fy).astype(np.float32))
                role_list.append(np.full(len(values), ROLE_CODES[role], np.uint8))
                count += len(values)
        offsets.append(offsets[-1] + count)
        if index % 50 == 0:
            log(f"  players: frame {index}/{total}")
    while len(offsets) < total + 1:  # fewer decoded frames than expected: empty tail
        offsets.append(offsets[-1])

    def cat(parts: list[np.ndarray], shape: tuple[int, ...], dtype) -> np.ndarray:
        return np.concatenate(parts) if parts else np.zeros(shape, dtype)

    return Scene(
        background_points=bg_points,
        background_colors=bg_colors,
        background_radii=bg_radii,
        player_points=cat(pts_list, (0, 3), np.float32),
        player_colors=cat(col_list, (0, 3), np.uint8),
        player_radii=cat(rad_list, (0,), np.float32),
        player_roles=cat(role_list, (0,), np.uint8),
        frame_offsets=np.array(offsets, dtype=np.int64),
        ground=ground,
        camera=camera,
        fps=players.fps,
    )
