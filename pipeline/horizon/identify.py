"""Decide which tracked person is the near player and which is the far player.

The AI services provide names, clothing and a frame box per player. Court geometry validates their answer,
and a persistence heuristic takes over whenever the answer is missing or inconsistent.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from horizon.camera import PinholeCamera
from horizon.court import HALF_DOUBLES
from horizon.llm import gemini, pegasus
from horizon.llm.backboard import run_tool_loop
from horizon.llm.pegasus import ClipAnalysis, PlayerDescription
from horizon.tracking import Detection, Detections

ROLES = ("near", "far")
HALF_Y_RANGE = {"near": (-18.0, -0.5), "far": (0.5, 18.0)}
MAX_ABS_X = 8.0

SYSTEM_PROMPT = (
    "You are Horizon Tennis Analyst. You identify the two players in broadcast tennis clips by calling tools. "
    "Always call analyze_tennis_clip first, then call locate_players with the near and far appearance strings it "
    "returned. Remember players you have seen before (names and outfits) to help with future clips. "
    "When both tools have run, reply with the single word: done."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "analyze_tennis_clip",
            "description": "Run TwelveLabs Pegasus on the tennis clip. Returns both players' names and clothing, "
            "the scoreboard text and a short summary of the point.",
            "parameters": {
                "type": "object",
                "properties": {"match_id": {"type": "string", "description": "Match id of the clip"}},
                "required": ["match_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "locate_players",
            "description": "Run Gemini on a frame of the clip to find the bounding boxes of the near and far players, "
            "given their clothing descriptions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "match_id": {"type": "string"},
                    "near_appearance": {"type": "string", "description": "Clothing of the near player"},
                    "far_appearance": {"type": "string", "description": "Clothing of the far player"},
                },
                "required": ["match_id", "near_appearance", "far_appearance"],
            },
        },
    },
]


def court_xy(camera: PinholeCamera, pixel: tuple[float, float]) -> tuple[float, float] | None:
    points, depth = camera.ground_intersection(np.array([pixel[0]]), np.array([pixel[1]]))
    if not np.isfinite(depth[0]):
        return None
    return float(points[0, 0]), float(points[0, 1])


def on_half(role: str, xy: tuple[float, float] | None) -> bool:
    if xy is None:
        return False
    lo, hi = HALF_Y_RANGE[role]
    return abs(xy[0]) <= MAX_ABS_X and lo <= xy[1] <= hi


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_box(box, candidates: list[Detection], width: int, height: int) -> Detection | None:
    """Detection overlapping `box` best (IoU >= 0.1), else the nearest centre within 15 % of the image diagonal."""
    if not candidates:
        return None
    best = max(candidates, key=lambda d: iou(box, d.bbox))
    if iou(box, best.bbox) >= 0.1:
        return best
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2

    def distance(d: Detection) -> float:
        return float(np.hypot((d.bbox[0] + d.bbox[2]) / 2 - cx, (d.bbox[1] + d.bbox[3]) / 2 - cy))

    nearest = min(candidates, key=distance)
    return nearest if distance(nearest) <= 0.15 * float(np.hypot(width, height)) else None


def _track_court_stats(detections: Detections, camera: PinholeCamera) -> dict[int, dict]:
    """Per-track ground positions, statures and median court xy, shared by heuristic_tracks and resolve_identity."""
    positions: dict[int, list[tuple[float, float]]] = {}
    statures: dict[int, list[float]] = {}
    for frame in detections.frames:
        for det in frame:
            xy = court_xy(camera, det.foot)
            if xy is None:
                continue
            positions.setdefault(det.track_id, []).append(xy)
            statures.setdefault(det.track_id, []).append(camera.height_above_ground(xy, det.bbox[1]))
    stats: dict[int, dict] = {}
    for track_id, xys in positions.items():
        arr = np.array(xys)
        stats[track_id] = {
            "positions": xys,
            "statures": statures[track_id],
            "median": (float(np.median(arr[:, 0])), float(np.median(arr[:, 1]))),
        }
    return stats


def _heuristic_tracks_from_stats(stats: dict[int, dict]) -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    for role in ROLES:
        best_id, best_score = None, -1.0
        for track_id, track in stats.items():
            median = track["median"]
            if not on_half(role, median):
                continue
            count = sum(on_half(role, xy) for xy in track["positions"])
            weight = 1.0 if abs(median[0]) <= HALF_DOUBLES + 0.5 else 0.5
            score = count * weight + 0.01 * float(np.median(track["statures"]))
            if score > best_score:
                best_id, best_score = track_id, score
        result[role] = best_id
    return result


def heuristic_tracks(detections: Detections, camera: PinholeCamera) -> dict[str, int | None]:
    """Most persistent track per court half; tracks whose median position is outside the court width count half."""
    return _heuristic_tracks_from_stats(_track_court_stats(detections, camera))


@dataclass
class PlayerIdentity:
    role: str
    name: str
    description: str
    track_id: int | None
    source: str  # "gemini", "heuristic" or "none"
    box: list[float] | None = None


@dataclass
class Identity:
    players: dict[str, PlayerIdentity]
    score: str
    summary: str
    frame_index: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "players": {role: asdict(p) for role, p in self.players.items()},
            "score": self.score,
            "summary": self.summary,
            "frame_index": self.frame_index,
            "warnings": self.warnings,
        }

    @classmethod
    def from_json(cls, data: dict) -> "Identity":
        return cls(
            players={role: PlayerIdentity(**p) for role, p in data["players"].items()},
            score=data["score"],
            summary=data["summary"],
            frame_index=int(data.get("frame_index", 0)),
            warnings=list(data.get("warnings", [])),
        )

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "Identity":
        return cls.from_json(json.loads(Path(path).read_text()))


def resolve_identity(
    analysis: ClipAnalysis, boxes: dict, detections: Detections, camera: PinholeCamera, frame_index: int = 0
) -> Identity:
    warnings: list[str] = []
    stats = _track_court_stats(detections, camera)
    fallback = _heuristic_tracks_from_stats(stats)
    candidates = detections.frames[frame_index] if frame_index < len(detections.frames) else []
    used: set[int] = set()
    players: dict[str, PlayerIdentity] = {}
    for role in ROLES:
        description: PlayerDescription = getattr(analysis, role)
        box = boxes.get(role)
        track_id, source = None, "heuristic"
        if box is not None:
            free = [d for d in candidates if d.track_id not in used]
            det = match_box(box, free, detections.width, detections.height)
            median_x = stats[det.track_id]["median"][0] if det is not None and det.track_id in stats else None
            if det is None:
                warnings.append(f"{role}: Gemini box matched no tracked person on frame {frame_index}")
            elif not on_half(role, court_xy(camera, det.foot)):
                warnings.append(f"{role}: Gemini picked track {det.track_id}, which is not on the {role} half")
            elif median_x is not None and abs(median_x) > HALF_DOUBLES + 0.5:
                warnings.append(f"{role}: Gemini picked track {det.track_id}, which stays outside the court width")
            else:
                track_id, source = det.track_id, "gemini"
        if track_id is None and fallback[role] not in used:
            track_id = fallback[role]
        if track_id is None:
            warnings.append(f"{role}: no track found for this player")
            source = "none"
        else:
            used.add(track_id)
        players[role] = PlayerIdentity(
            role=role,
            name=description.name,
            description=description.appearance,
            track_id=track_id,
            source=source,
            box=[float(v) for v in box] if box is not None else None,
        )
    return Identity(players=players, score=analysis.score, summary=analysis.summary, frame_index=frame_index, warnings=warnings)


def _fingerprint() -> str:
    return hashlib.sha1(json.dumps([SYSTEM_PROMPT, TOOLS], sort_keys=True).encode()).hexdigest()


def get_or_create_assistant(client, cache_path: Path) -> str:
    """Reuse the Backboard assistant (and its memory) across runs while the prompt/tools are unchanged."""
    cache_path = Path(cache_path)
    fingerprint = _fingerprint()
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text())
        if cached.get("fingerprint") == fingerprint and cached.get("assistant_id"):
            return str(cached["assistant_id"])
    assistant_id = client.create_assistant("Horizon Tennis Analyst", SYSTEM_PROMPT, TOOLS)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"assistant_id": assistant_id, "fingerprint": fingerprint}))
    return assistant_id


def identify_direct(video_path: Path, frame_rgb: np.ndarray) -> tuple[ClipAnalysis, dict]:
    analysis = pegasus.analyze_clip(video_path)
    boxes = gemini.locate_players(frame_rgb, analysis.near.appearance, analysis.far.appearance)
    return analysis, boxes


def identify_via_backboard(
    client,
    match_id: str,
    video_path: Path,
    frame_rgb: np.ndarray,
    cache_path: Path,
    llm_provider: str | None = None,
    model_name: str | None = None,
) -> tuple[ClipAnalysis, dict, list[str]]:
    """Let the Backboard assistant drive both tools. Results are captured locally; missing steps run directly."""
    state: dict = {}

    def analyze_tool(_args: dict) -> dict:
        state["analysis"] = pegasus.analyze_clip(video_path)
        return state["analysis"].to_json()

    def locate_tool(args: dict) -> dict:
        analysis = state.get("analysis")
        near = args.get("near_appearance") or (analysis.near.appearance if analysis else "")
        far = args.get("far_appearance") or (analysis.far.appearance if analysis else "")
        state["boxes"] = gemini.locate_players(frame_rgb, near, far)
        return {role: list(box) for role, box in state["boxes"].items()}

    assistant_id = get_or_create_assistant(client, cache_path)
    thread_id = client.create_thread(assistant_id)
    result = run_tool_loop(
        client,
        thread_id,
        f"Identify the near and far players in tennis clip '{match_id}'.",
        {"analyze_tennis_clip": analyze_tool, "locate_players": locate_tool},
        llm_provider=llm_provider,
        model_name=model_name,
    )
    notes = [f"backboard: {e.name} failed: {e.error}" for e in result.executions if e.error]
    if "analysis" not in state:
        notes.append("backboard: assistant skipped analyze_tennis_clip; called Pegasus directly")
        state["analysis"] = pegasus.analyze_clip(video_path)
    if "boxes" not in state:
        notes.append("backboard: assistant skipped locate_players; called Gemini directly")
        analysis = state["analysis"]
        state["boxes"] = gemini.locate_players(frame_rgb, analysis.near.appearance, analysis.far.appearance)
    return state["analysis"], state["boxes"], notes
