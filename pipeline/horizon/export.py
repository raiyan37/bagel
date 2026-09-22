"""Publish a processed match to the React app: public/matches/<id>/ and public/matches/index.json (spec §6)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from horizon.calibration import load_calibration
from horizon.camera import PinholeCamera
from horizon.identify import Identity
from horizon.paths import REPO_ROOT, MatchPaths
from horizon.players import Players
from horizon.video import VideoInfo, load_meta

PUBLIC_MATCHES = REPO_ROOT / "public" / "matches"
PLAYER_COLORS = {"near": "#3B82F6", "far": "#F97316"}
VIEWER_URL = "http://localhost:8080"


def head_anchor_percent(camera: PinholeCamera, players: Players, role: str, frame: int) -> tuple[float, float]:
    """Top-centre of the player's box in % of the frame; projected from the trajectory when undetected."""
    track = players.tracks[role]
    bbox = track.bboxes[frame]
    if bbox is not None:
        x, y = (bbox[0] + bbox[2]) / 2.0, bbox[1]
    else:
        fx, fy = track.foot_xy[frame]
        uv, _ = camera.project(np.array([[fx, fy, track.stature_m]]))
        x, y = float(uv[0, 0]), float(uv[0, 1])
    return round(100.0 * x / camera.width, 2), round(100.0 * y / camera.height, 2)


def build_tracks_json(players: Players, camera: PinholeCamera) -> dict:
    result: dict = {"fps": players.fps, "frameCount": players.frame_count, "players": {}}
    for role, track in players.tracks.items():
        anchors = [head_anchor_percent(camera, players, role, frame) for frame in range(players.frame_count)]
        result["players"][role] = {
            "x": [a[0] for a in anchors],
            "y": [a[1] for a in anchors],
            "visible": [1 if v else 0 for v in track.visible],
            "speedKmh": np.round(track.speed_kmh, 1).tolist(),
            "distanceM": np.round(track.distance_m, 1).tolist(),
        }
    return result


def build_manifest(
    match_id: str,
    info: VideoInfo,
    identity: Identity,
    players: Players,
    has_free_cam: bool,
    competition: str = "Tennis",
    viewer_url: str = VIEWER_URL,
) -> dict:
    near, far = identity.players["near"], identity.players["far"]
    return {
        "id": match_id,
        "title": f"{near.name} vs {far.name}",
        "competition": competition,
        "summary": identity.summary,
        "score": identity.score,
        "fps": info.fps,
        "frameCount": info.frame_count,
        "width": info.width,
        "height": info.height,
        "video": "main.mp4",
        "tracks": "tracks.json",
        "players": [
            {
                "id": role,
                "name": identity.players[role].name,
                "description": identity.players[role].description,
                "color": PLAYER_COLORS[role],
                "pov": f"pov_{role}.mp4",
                "statureM": round(players.tracks[role].stature_m, 2),
            }
            for role in ("near", "far")
        ],
        "freeCam": "free_cam.mp4" if has_free_cam else None,
        "viewerUrl": viewer_url,
    }


def update_index(public_matches: Path) -> dict:
    matches = []
    for manifest_path in sorted(Path(public_matches).glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        matches.append(
            {
                "id": manifest["id"],
                "title": manifest["title"],
                "near": manifest["players"][0]["name"],
                "far": manifest["players"][1]["name"],
                "competition": manifest["competition"],
                "status": "replay",
            }
        )
    index = {"matches": matches}
    (Path(public_matches) / "index.json").write_text(json.dumps(index, indent=2))
    return index


def export_match(
    paths: MatchPaths, public_matches: Path = PUBLIC_MATCHES, competition: str = "Tennis", viewer_url: str = VIEWER_URL
) -> Path:
    """Publish one match. The size check keeps tracks.json percentages relative to the frame the manifest declares."""
    required = [paths.source_video, paths.pov_video("near"), paths.pov_video("far")]
    missing = [p.name for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {', '.join(missing)} in {paths.root}; run the earlier pipeline steps")
    info = load_meta(paths.meta)
    identity = Identity.load(paths.identity)
    players = Players.load(paths.players)
    camera = load_calibration(paths.calibration).camera
    if (camera.width, camera.height) != (info.width, info.height):
        raise ValueError(
            f"calibration.json is for a {camera.width}x{camera.height} frame but source.mp4 is "
            f"{info.width}x{info.height}; re-run `horizon calibrate --match-id {paths.match_id}` "
            "(or `horizon all ... --recalibrate`)"
        )
    target = Path(public_matches) / paths.match_id
    target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(paths.source_video, target / "main.mp4")
    for role in ("near", "far"):
        shutil.copyfile(paths.pov_video(role), target / f"pov_{role}.mp4")
    has_free_cam = paths.free_cam_video.is_file()
    if has_free_cam:
        shutil.copyfile(paths.free_cam_video, target / "free_cam.mp4")
    (target / "tracks.json").write_text(json.dumps(build_tracks_json(players, camera)))
    manifest = build_manifest(paths.match_id, info, identity, players, has_free_cam, competition, viewer_url)
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2))
    update_index(public_matches)
    return target
