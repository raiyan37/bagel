import json

import numpy as np
import pytest
from synthetic import H, W, broadcast_camera, make_identity

from horizon.calibration import Calibration, save_calibration
from horizon.export import build_manifest, build_tracks_json, export_match, head_anchor_percent
from horizon.paths import MatchPaths
from horizon.players import Players, PlayerTrack
from horizon.video import VideoInfo, save_meta


def two_frame_players() -> Players:
    def track(role, xy, bbox):
        foot = np.tile(np.array(xy, dtype=float), (2, 1))
        return PlayerTrack(role, role.title(), "", [1, None], [bbox, None], foot, 1.85, np.array([3.0, 4.0]), np.array([0.0, 0.1]))

    return Players(
        25.0,
        2,
        {
            "near": track("near", (1.0, -10.0), (600.0, 300.0, 640.0, 420.0)),
            "far": track("far", (-2.0, 10.0), (500.0, 100.0, 520.0, 150.0)),
        },
    )


def write_match(root, match_id="demo", with_pov=True) -> MatchPaths:
    paths = MatchPaths.for_match(match_id, data_root=root).ensure()
    save_meta(paths.meta, VideoInfo(W, H, 25.0, 2))
    make_identity().save(paths.identity)
    two_frame_players().save(paths.players)
    save_calibration(Calibration(camera=broadcast_camera(), keypoints={}, rms_px=0.5), paths.calibration)
    paths.source_video.write_bytes(b"main")
    if with_pov:
        paths.pov_video("near").write_bytes(b"near")
        paths.pov_video("far").write_bytes(b"far")
    return paths


def test_head_anchor_uses_box_top_and_projects_missing_frames():
    cam, players = broadcast_camera(), two_frame_players()
    assert head_anchor_percent(cam, players, "near", 0) == (pytest.approx(100 * 620 / W, abs=0.01), pytest.approx(100 * 300 / H, abs=0.01))
    uv, _ = cam.project(np.array([[1.0, -10.0, 1.85]]))
    x, y = head_anchor_percent(cam, players, "near", 1)
    assert x == pytest.approx(100 * uv[0, 0] / W, abs=0.01) and y == pytest.approx(100 * uv[0, 1] / H, abs=0.01)


def test_tracks_and_manifest_follow_the_web_contract():
    players = two_frame_players()
    tracks = build_tracks_json(players, broadcast_camera())
    assert tracks["fps"] == 25.0 and tracks["frameCount"] == 2
    assert set(tracks["players"]["near"]) == {"x", "y", "visible", "speedKmh", "distanceM"}
    assert tracks["players"]["far"]["visible"] == [1, 0]
    manifest = build_manifest("demo", VideoInfo(W, H, 25.0, 2), make_identity(), players, has_free_cam=False)
    assert set(manifest) == {
        "id", "title", "competition", "summary", "score", "fps", "frameCount", "width", "height",
        "video", "tracks", "players", "freeCam", "viewerUrl",
    }
    assert manifest["title"] == "Near Player vs Far Player"
    assert [p["id"] for p in manifest["players"]] == ["near", "far"]
    assert manifest["players"][0] == {
        "id": "near", "name": "Near Player", "description": "white shirt", "color": "#3B82F6",
        "pov": "pov_near.mp4", "statureM": 1.85,
    }
    assert manifest["freeCam"] is None


def test_export_match_copies_media_and_rebuilds_the_index(tmp_path):
    public = tmp_path / "public" / "matches"
    export_match(write_match(tmp_path / "data", "demo"), public_matches=public)
    second = write_match(tmp_path / "data", "another")
    second.free_cam_video.write_bytes(b"free")
    export_match(second, public_matches=public)
    demo = public / "demo"
    assert (demo / "main.mp4").read_bytes() == b"main"
    assert (demo / "pov_far.mp4").read_bytes() == b"far"
    assert json.loads((demo / "manifest.json").read_text())["freeCam"] is None
    assert json.loads((public / "another" / "manifest.json").read_text())["freeCam"] == "free_cam.mp4"
    assert json.loads((demo / "tracks.json").read_text())["frameCount"] == 2
    index = json.loads((public / "index.json").read_text())
    assert [m["id"] for m in index["matches"]] == ["another", "demo"]
    assert index["matches"][1] == {
        "id": "demo", "title": "Near Player vs Far Player", "near": "Near Player", "far": "Far Player",
        "competition": "Tennis", "status": "replay",
    }


def test_export_reports_missing_videos(tmp_path):
    with pytest.raises(FileNotFoundError, match="pov_near.mp4"):
        export_match(write_match(tmp_path / "data", with_pov=False), public_matches=tmp_path / "public")
