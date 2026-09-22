import json

import numpy as np
import pytest

from horizon.calibration import (
    MAX_ACCEPTABLE_RMS_PX,
    draw_court_overlay,
    draw_keypoint_legend,
    load_calibration,
    save_calibration,
    solve_calibration,
)
from horizon.camera import PinholeCamera
from horizon.cli import main
from horizon.court import KEYPOINTS
from horizon.video import H264Writer

W, H = 1280, 720
CORNERS = ["near_doubles_left", "near_doubles_right", "far_doubles_right", "far_doubles_left"]


def true_camera() -> PinholeCamera:
    return PinholeCamera.look_at(eye=(0.5, -24.0, 9.0), target=(0.0, -3.0, 0.0), vertical_fov_deg=40.0, width=W, height=H)


def clicks(cam, names=None, noise=0.0, seed=0) -> dict[str, tuple[float, float]]:
    names = names or list(KEYPOINTS)
    uv, z = cam.project(np.array([[*KEYPOINTS[n], 0.0] for n in names]))
    assert (z > 0).all() and (uv >= 0).all() and (uv[:, 0] < W).all() and (uv[:, 1] < H).all()
    if noise:
        uv = uv + np.random.default_rng(seed).normal(0.0, noise, uv.shape)
    return {n: (float(u), float(v)) for n, (u, v) in zip(names, uv)}


def test_recovers_camera_from_all_keypoints():
    cam = true_camera()
    cal = solve_calibration(clicks(cam), W, H)
    assert cal.camera.fy == pytest.approx(cam.fy, rel=5e-3)
    assert np.allclose(cal.camera.center, cam.center, atol=0.05)
    assert cal.rms_px < 0.05


def test_recovers_camera_from_four_corners():
    cam = true_camera()
    cal = solve_calibration(clicks(cam, CORNERS), W, H)
    assert cal.camera.fy == pytest.approx(cam.fy, rel=1e-2)
    assert np.allclose(cal.camera.center, cam.center, atol=0.3)


def test_tolerates_click_noise():
    cam = true_camera()
    cal = solve_calibration(clicks(cam, noise=0.7, seed=3), W, H)
    assert cal.camera.fy == pytest.approx(cam.fy, rel=0.05)
    assert np.allclose(cal.camera.center, cam.center, atol=1.5)
    assert cal.rms_px < MAX_ACCEPTABLE_RMS_PX


def test_rejects_fewer_than_four_points():
    with pytest.raises(ValueError):
        solve_calibration(clicks(true_camera(), CORNERS[:3]), W, H)


def test_save_load_and_overlay(tmp_path):
    cal = solve_calibration(clicks(true_camera()), W, H)
    save_calibration(cal, tmp_path / "calibration.json")
    again = load_calibration(tmp_path / "calibration.json")
    assert np.allclose(again.camera.R, cal.camera.R)
    assert again.rms_px == pytest.approx(cal.rms_px)
    assert set(again.keypoints) == set(KEYPOINTS)
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    overlay = draw_court_overlay(frame, cal.camera, cal.keypoints)
    assert overlay.shape == frame.shape and overlay.any()
    assert not frame.any()


def test_legend_marks_current_keypoint():
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    draw_keypoint_legend(canvas, "far_service_center")
    assert canvas[:, :1100].sum() == 0
    red = (canvas[..., 2] == 255) & (canvas[..., 1] == 0) & (canvas[..., 0] == 0)
    assert red.any()


def test_calibrate_command_from_keypoints_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HORIZON_DATA_ROOT", str(tmp_path))
    match = tmp_path / "unit"
    match.mkdir()
    with H264Writer(match / "source.mp4", W, H, 25.0) as writer:
        writer.write(np.zeros((H, W, 3), dtype=np.uint8))
    keypoints_file = tmp_path / "kp.json"
    keypoints_file.write_text(json.dumps({"keypoints": {n: list(v) for n, v in clicks(true_camera()).items()}}))
    assert main(["calibrate", "--match-id", "unit", "--keypoints", str(keypoints_file)]) == 0
    assert (match / "calibration.json").is_file()
    assert (match / "calibration_preview.jpg").is_file()
