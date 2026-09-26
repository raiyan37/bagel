"""Video helpers: OpenCV decoding, browser-friendly H.264 encoding via imageio-ffmpeg."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps


def probe(path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        return VideoInfo(
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=fps if fps > 0 else 25.0,
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()


def iter_frames(path: Path, start: int = 0, stop: int | None = None, step: int = 1) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame index, RGB uint8 frame) for start <= index < stop, every `step` frames."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    try:
        if start:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        index = start
        while stop is None or index < stop:
            ok, bgr = cap.read()
            if not ok:
                break
            if (index - start) % step == 0:
                yield index, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            index += 1
    finally:
        cap.release()


def read_frame(path: Path, index: int) -> np.ndarray:
    for _, frame in iter_frames(path, start=index, stop=index + 1):
        return frame
    raise IndexError(f"Frame {index} not found in {path}")


class H264Writer:
    """Write RGB uint8 frames to an H.264 yuv420p MP4 that browsers can play."""

    def __init__(self, path: Path, width: int, height: int, fps: float, crf: int = 20):
        if width % 2 or height % 2:
            raise ValueError(f"H.264 needs even frame dimensions, got {width}x{height}")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.size = (width, height)
        self.frames_written = 0
        self._gen = imageio_ffmpeg.write_frames(
            str(self.path),
            self.size,
            fps=fps,
            codec="libx264",
            pix_fmt_in="rgb24",
            pix_fmt_out="yuv420p",
            quality=None,
            macro_block_size=2,
            output_params=["-crf", str(crf), "-preset", "medium", "-movflags", "+faststart"],
        )
        self._gen.send(None)

    def write(self, frame: np.ndarray) -> None:
        expected = (self.size[1], self.size[0], 3)
        if frame.shape != expected or frame.dtype != np.uint8:
            raise ValueError(f"Expected uint8 frame of shape {expected}, got {frame.dtype} {frame.shape}")
        self._gen.send(np.ascontiguousarray(frame))
        self.frames_written += 1

    def close(self) -> None:
        self._gen.close()

    def __enter__(self) -> "H264Writer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def transcode_clip(
    src: Path, dst: Path, start_s: float = 0.0, duration_s: float | None = None, max_height: int = 720
) -> VideoInfo:
    """Trim [start_s, start_s + duration_s), downscale to <= max_height and re-encode as H.264."""
    info = probe(src)
    start = int(round(start_s * info.fps))
    stop = None if duration_s is None else start + int(round(duration_s * info.fps))
    scale = min(1.0, max_height / info.height)
    width = int(round(info.width * scale / 2)) * 2
    height = int(round(info.height * scale / 2)) * 2
    with H264Writer(dst, width, height, info.fps) as writer:
        for _, frame in iter_frames(src, start=start, stop=stop):
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
        count = writer.frames_written
    if count == 0:
        raise ValueError(f"No frames decoded from {src} in the requested range")
    return VideoInfo(width=width, height=height, fps=info.fps, frame_count=count)


def save_meta(path: Path, info: VideoInfo, **extra) -> None:
    data = {"width": info.width, "height": info.height, "fps": info.fps, "frame_count": info.frame_count, **extra}
    Path(path).write_text(json.dumps(data, indent=2))


def load_meta(path: Path) -> VideoInfo:
    data = json.loads(Path(path).read_text())
    return VideoInfo(int(data["width"]), int(data["height"]), float(data["fps"]), int(data["frame_count"]))
