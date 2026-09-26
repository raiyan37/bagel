import numpy as np
import pytest

from horizon.camera import (
    PinholeCamera,
    camera_from_dict,
    camera_from_pose,
    camera_pose,
    camera_to_dict,
    rotation_to_wxyz,
    vertical_fov_from_horizontal,
    wxyz_to_rotation,
)


def broadcast_camera() -> PinholeCamera:
    return PinholeCamera.look_at(
        eye=(0.5, -24.0, 9.0), target=(0.0, -3.0, 0.0), vertical_fov_deg=40.0, width=1280, height=720
    )


def test_look_at_orientation():
    cam = PinholeCamera.look_at(eye=(0, -10, 2), target=(0, 0, 2), vertical_fov_deg=60, width=640, height=480)
    uv, z = cam.project(np.array([[0.0, 5.0, 2.0], [1.0, 5.0, 2.0], [0.0, 5.0, 3.0]]))
    assert z[0] == pytest.approx(15.0)
    assert uv[0] == pytest.approx([cam.cx, cam.cy])
    assert uv[1, 0] > cam.cx  # +X world appears to the right
    assert uv[2, 1] < cam.cy  # +Z world appears higher (smaller v)
    assert np.allclose(cam.center, [0, -10, 2])
    assert np.allclose(cam.forward, [0, 1, 0])
    assert np.linalg.det(cam.R) == pytest.approx(1.0)


def test_project_backproject_roundtrip():
    cam = broadcast_camera()
    pts = np.random.default_rng(0).uniform([-6, -12, 0], [6, 12, 3], size=(200, 3))
    uv, z = cam.project(pts)
    assert np.allclose(cam.backproject(uv[:, 0], uv[:, 1], z), pts, atol=1e-9)


def test_ground_intersection_recovers_points_and_depth():
    cam = broadcast_camera()
    pts = np.array([[0.0, 0.0, 0.0], [5.485, 11.885, 0.0], [-4.1, -11.0, 0.0]])
    uv, z = cam.project(pts)
    hit, depth = cam.ground_intersection(uv[:, 0], uv[:, 1])
    assert np.allclose(hit, pts, atol=1e-9)
    assert np.allclose(depth, z)


def test_rays_above_the_horizon_miss_the_ground():
    cam = broadcast_camera()
    uv, _ = cam.project(np.array([[0.0, 60.0, 30.0]]))
    _, depth = cam.ground_intersection(uv[:, 0], uv[:, 1])
    assert np.isinf(depth[0])


def test_height_above_ground_recovers_stature():
    cam = broadcast_camera()
    uv_head, _ = cam.project(np.array([[1.2, 9.0, 1.88]]))
    assert cam.height_above_ground((1.2, 9.0), float(uv_head[0, 1])) == pytest.approx(1.88, abs=1e-6)


def test_quaternion_roundtrip_random_rotations():
    rng = np.random.default_rng(1)
    for _ in range(50):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        R = wxyz_to_rotation(q)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.allclose(wxyz_to_rotation(rotation_to_wxyz(R)), R, atol=1e-9)


def test_camera_pose_roundtrip():
    cam = broadcast_camera()
    wxyz, position = camera_pose(cam)
    again = camera_from_pose(wxyz, position, np.degrees(cam.vertical_fov), cam.width, cam.height)
    pts = np.array([[0.0, 0.0, 0.0], [3.0, 8.0, 1.0]])
    assert np.allclose(again.project(pts)[0], cam.project(pts)[0], atol=1e-6)


def test_dict_roundtrip_and_fov_helper():
    cam = broadcast_camera()
    again = camera_from_dict(camera_to_dict(cam))
    assert np.allclose(again.K, cam.K) and np.allclose(again.R, cam.R) and np.allclose(again.t, cam.t)
    assert (again.width, again.height) == (1280, 720)
    assert vertical_fov_from_horizontal(90.0, 1600, 900) == pytest.approx(58.7155, abs=1e-3)
