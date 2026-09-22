"""Person detection + tracking with Ultralytics YOLO segmentation and ByteTrack."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from horizon.video import VideoInfo, iter_frames

DEFAULT_MODEL = "yolo26s-seg.pt"


@dataclass(frozen=True)
class Detection:
    track_id: int
    conf: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    polygon: tuple[tuple[float, float], ...] = ()  # mask outline in pixels

    @property
    def foot(self) -> tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2.0, self.bbox[3])

    @property
    def head(self) -> tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2.0, self.bbox[1])

    def mask(self, width: int, height: int) -> np.ndarray:
        canvas = np.zeros((height, width), dtype=np.uint8)
        if len(self.polygon) >= 3:
            pts = np.round(np.array(self.polygon)).astype(np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(canvas, [pts], 1)
        else:
            x1, y1, x2, y2 = (int(round(v)) for v in self.bbox)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), 1, thickness=-1)
        return canvas.astype(bool)

    def to_json(self) -> dict:
        return {
            "id": self.track_id,
            "conf": round(self.conf, 4),
            "bbox": [round(v, 2) for v in self.bbox],
            "polygon": [[round(x, 1), round(y, 1)] for x, y in self.polygon],
        }

    @classmethod
    def from_json(cls, data: dict) -> "Detection":
        return cls(
            track_id=int(data["id"]),
            conf=float(data["conf"]),
            bbox=tuple(float(v) for v in data["bbox"]),
            polygon=tuple((float(x), float(y)) for x, y in data.get("polygon", [])),
        )


@dataclass
class Detections:
    width: int
    height: int
    fps: float
    frames: list[list[Detection]]

    def by_track(self, frame: int, track_id: int) -> Detection | None:
        for det in self.frames[frame]:
            if det.track_id == track_id:
                return det
        return None

    def save(self, path: Path) -> None:
        data = {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frames": [[d.to_json() for d in frame] for frame in self.frames],
        }
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path: Path) -> "Detections":
        data = json.loads(Path(path).read_text())
        frames = [[Detection.from_json(d) for d in frame] for frame in data["frames"]]
        return cls(int(data["width"]), int(data["height"]), float(data["fps"]), frames)


def _simplify(polygon: np.ndarray, epsilon: float = 1.0) -> np.ndarray:
    polygon = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    if len(polygon) < 3:
        return polygon
    return cv2.approxPolyDP(polygon.reshape(-1, 1, 2), epsilon, True).reshape(-1, 2)


def detections_from_arrays(xyxy, conf, ids, polygons) -> list[Detection]:
    """One frame of Ultralytics output (numpy) -> Detections; boxes without a track id are dropped."""
    if ids is None:
        return []
    out = []
    for i in range(len(xyxy)):
        poly = _simplify(polygons[i]) if polygons is not None and i < len(polygons) else np.zeros((0, 2))
        out.append(
            Detection(
                track_id=int(ids[i]),
                conf=float(conf[i]),
                bbox=tuple(float(v) for v in xyxy[i]),
                polygon=tuple((float(x), float(y)) for x, y in poly),
            )
        )
    return out


def run_tracking(
    video_path: Path, info: VideoInfo, model_name: str = DEFAULT_MODEL, imgsz: int = 1280, conf: float = 0.2
) -> Detections:
    from ultralytics import YOLO

    model = YOLO(model_name)
    frames: list[list[Detection]] = []
    for index, rgb in iter_frames(video_path):
        result = model.track(
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            imgsz=imgsz,
            conf=conf,
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None or boxes.id is None or len(boxes) == 0:
            frames.append([])
        else:
            polygons = result.masks.xy if result.masks is not None else None
            frames.append(
                detections_from_arrays(
                    boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.id.cpu().numpy(), polygons
                )
            )
        if index % 50 == 0:
            print(f"  tracked frame {index}/{info.frame_count}")
    if len(frames) != info.frame_count:
        print(f"WARNING: decoded {len(frames)} frames but meta.json says {info.frame_count}")
    return Detections(info.width, info.height, info.fps, frames)
