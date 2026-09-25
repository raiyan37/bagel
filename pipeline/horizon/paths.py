"""Where every per-match artifact lives."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PIPELINE_ROOT.parent
DATA_ROOT = PIPELINE_ROOT / "data"

_MATCH_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


def resolve_data_root() -> Path:
    """Data directory for match artifacts; HORIZON_DATA_ROOT overrides it (tests use tmp dirs)."""
    return Path(os.environ.get("HORIZON_DATA_ROOT") or DATA_ROOT)


def validate_match_id(match_id: str) -> str:
    if not _MATCH_ID.fullmatch(match_id or ""):
        raise ValueError(
            f"Invalid match id {match_id!r}: use 1-64 lowercase letters, digits or '-', not starting with '-'"
        )
    return match_id


@dataclass(frozen=True)
class MatchPaths:
    root: Path

    @classmethod
    def for_match(cls, match_id: str, data_root: Path | None = None) -> "MatchPaths":
        root = Path(data_root) if data_root is not None else resolve_data_root()
        return cls(root / validate_match_id(match_id))

    def ensure(self) -> "MatchPaths":
        self.root.mkdir(parents=True, exist_ok=True)
        return self

    @property
    def match_id(self) -> str:
        return self.root.name

    @property
    def source_video(self) -> Path:
        return self.root / "source.mp4"

    @property
    def meta(self) -> Path:
        return self.root / "meta.json"

    @property
    def calibration(self) -> Path:
        return self.root / "calibration.json"

    @property
    def calibration_preview(self) -> Path:
        return self.root / "calibration_preview.jpg"

    @property
    def detections(self) -> Path:
        return self.root / "detections.json"

    @property
    def identity(self) -> Path:
        return self.root / "identity.json"

    @property
    def players(self) -> Path:
        return self.root / "players.json"

    @property
    def ball(self) -> Path:
        return self.root / "ball.json"

    @property
    def scene(self) -> Path:
        return self.root / "scene.npz"

    @property
    def ground_png(self) -> Path:
        return self.root / "ground.png"

    @property
    def free_cam_video(self) -> Path:
        return self.root / "free_cam.mp4"

    def pov_video(self, role: str) -> Path:
        return self.root / f"pov_{role}.mp4"
