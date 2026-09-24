"""Per-frame metric trajectories (court metres) for the near and far players."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from horizon.camera import PinholeCamera
from horizon.identify import ROLES, Identity, court_xy, on_half
from horizon.tracking import Detection, Detections

EYE_HEIGHT_RATIO = 0.94  # eye height / stature for adults


@dataclass
class PlayerTrack:
    role: str
    name: str
    description: str
    track_ids: list[int | None]
    bboxes: list[tuple[float, float, float, float] | None]
    foot_xy: np.ndarray  # (T, 2) smoothed court position in metres
    stature_m: float
    speed_kmh: np.ndarray  # (T,)
    distance_m: np.ndarray  # (T,) cumulative

    @property
    def eye_height_m(self) -> float:
        return EYE_HEIGHT_RATIO * self.stature_m

    @property
    def visible(self) -> list[bool]:
        return [b is not None for b in self.bboxes]

    def eye(self, frame: int) -> np.ndarray:
        x, y = self.foot_xy[frame]
        return np.array([x, y, self.eye_height_m])

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "stature_m": round(self.stature_m, 4),
            "track_ids": self.track_ids,
            "bboxes": [None if b is None else [round(v, 2) for v in b] for b in self.bboxes],
            "foot_xy": np.round(self.foot_xy, 4).tolist(),
            "speed_kmh": np.round(self.speed_kmh, 3).tolist(),
            "distance_m": np.round(self.distance_m, 3).tolist(),
        }

    @classmethod
    def from_json(cls, role: str, data: dict) -> "PlayerTrack":
        return cls(
            role=role,
            name=data["name"],
            description=data["description"],
            track_ids=list(data["track_ids"]),
            bboxes=[None if b is None else tuple(b) for b in data["bboxes"]],
            foot_xy=np.array(data["foot_xy"], dtype=np.float64).reshape(-1, 2),
            stature_m=float(data["stature_m"]),
            speed_kmh=np.array(data["speed_kmh"], dtype=np.float64),
            distance_m=np.array(data["distance_m"], dtype=np.float64),
        )


@dataclass
class Players:
    fps: float
    frame_count: int
    tracks: dict[str, PlayerTrack]

    def opponent(self, role: str) -> PlayerTrack:
        return self.tracks["far" if role == "near" else "near"]

    def save(self, path: Path) -> None:
        data = {
            "fps": self.fps,
            "frame_count": self.frame_count,
            "players": {role: track.to_json() for role, track in self.tracks.items()},
        }
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path: Path) -> "Players":
        data = json.loads(Path(path).read_text())
        tracks = {role: PlayerTrack.from_json(role, t) for role, t in data["players"].items()}
        return cls(fps=float(data["fps"]), frame_count=int(data["frame_count"]), tracks=tracks)


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def associate(
    detections: Detections,
    camera: PinholeCamera,
    role: str,
    start_track_id: int,
    max_base_jump: float = 1.5,
    max_speed: float = 10.0,
):
    """Follow one player through the clip on the court plane, surviving ByteTrack ID switches.

    Prefers the identified track whenever it is on the player's half, then the current one; otherwise takes
    the closest person on that half if they are within reach (max_base_jump + max_speed * elapsed seconds).
    Preferring the identified track lets the player be re-acquired after a ball kid has taken the id over.
    """
    current = start_track_id
    last_xy: tuple[float, float] | None = None
    last_frame: int | None = None
    for i, frame in enumerate(detections.frames):
        det = next((d for d in frame if d.track_id == current), None)
        if det is not None:
            last_xy, last_frame = court_xy(camera, det.foot), i
            break
    ids: list[int | None] = []
    chosen: list[Detection | None] = []
    xys: list[tuple[float, float] | None] = []
    for i, frame in enumerate(detections.frames):
        options = []
        for det in frame:
            xy = court_xy(camera, det.foot)
            if on_half(role, xy):
                options.append((det, xy))
        pick = next(((d, xy) for d, xy in options if d.track_id == start_track_id), None)
        if pick is not None:
            current = start_track_id
        else:
            pick = next(((d, xy) for d, xy in options if d.track_id == current), None)
        if pick is None and options and last_xy is not None:
            elapsed = abs(i - last_frame) / detections.fps if last_frame is not None else 0.0
            det, xy = min(options, key=lambda o: _distance(o[1], last_xy))
            if _distance(xy, last_xy) <= max_base_jump + max_speed * elapsed:
                pick = (det, xy)
                current = det.track_id
        if pick is None:
            ids.append(None)
            chosen.append(None)
            xys.append(None)
        else:
            det, xy = pick
            ids.append(det.track_id)
            chosen.append(det)
            xys.append(xy)
            last_xy, last_frame = xy, i
    return ids, chosen, xys


def fill_gaps(xys: list[tuple[float, float] | None]) -> np.ndarray:
    known = [i for i, v in enumerate(xys) if v is not None]
    if not known:
        raise ValueError("player was never detected on their half of the court")
    values = np.array([xys[i] for i in known], dtype=np.float64)
    t = np.arange(len(xys))
    return np.stack([np.interp(t, known, values[:, 0]), np.interp(t, known, values[:, 1])], axis=1)


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average that stays unbiased at the ends (normalised convolution)."""
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    window = min(window, n if n % 2 else n - 1)
    if window <= 1:
        return values.copy()
    kernel = np.ones(window)
    weights = np.convolve(np.ones(n), kernel, mode="same")
    return np.stack([np.convolve(values[:, c], kernel, mode="same") / weights for c in range(values.shape[1])], axis=1)


def kinematics(xy: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Speed (km/h, central differences) and cumulative distance (m) along a trajectory."""
    if len(xy) < 2:
        return np.zeros(len(xy)), np.zeros(len(xy))
    steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    distance = np.concatenate([[0.0], np.cumsum(steps)])
    speed = np.linalg.norm(np.gradient(xy, axis=0), axis=1) * fps * 3.6
    return speed, distance


def build_players(detections: Detections, camera: PinholeCamera, identity: Identity, smooth_window: int = 7) -> Players:
    tracks: dict[str, PlayerTrack] = {}
    for role in ROLES:
        ident = identity.players[role]
        if ident.track_id is None:
            raise ValueError(f"No track identified for the {role} player; run `horizon identify` first")
        ids, chosen, xys = associate(detections, camera, role, ident.track_id)
        foot = smooth(fill_gaps(xys), smooth_window)
        speed, distance = kinematics(foot, detections.fps)
        statures = [camera.height_above_ground(xy, det.bbox[1]) for det, xy in zip(chosen, xys) if det is not None]
        stature = float(np.clip(np.median(statures), 1.3, 2.2)) if statures else 1.8
        tracks[role] = PlayerTrack(
            role=role,
            name=ident.name,
            description=ident.description,
            track_ids=ids,
            bboxes=[None if d is None else d.bbox for d in chosen],
            foot_xy=foot,
            stature_m=stature,
            speed_kmh=speed,
            distance_m=distance,
        )
    return Players(fps=detections.fps, frame_count=len(detections.frames), tracks=tracks)
