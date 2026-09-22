import numpy as np
import pytest
from synthetic import (
    COURT_BLUE,
    H,
    LINE_WHITE,
    SURROUND_GREEN,
    TEST_EXTENT,
    W,
    WALL_HEIGHT,
    WALL_RED,
    WALL_Y,
    FakeEstimator,
    billboard_detection,
    broadcast_camera,
    make_identity,
    render_synthetic_frame,
)

from horizon.court import HALF_DOUBLES
from horizon.depth import ground_depth_map
from horizon.ground import build_ground_texture
from horizon.players import build_players
from horizon.reconstruct import Scene, background_cloud, median_plate, reconstruct_scene
from horizon.tracking import Detections
from horizon.video import H264Writer


def test_median_plate_removes_masked_players():
    background = np.full((20, 30, 3), 100, np.uint8)
    frames, masks = [], []
    for i in range(5):
        frame = background.copy()
        mask = np.zeros((20, 30), bool)
        mask[5:10, 2 + 5 * i : 6 + 5 * i] = True
        frame[mask] = (255, 0, 255)
        frames.append(frame)
        masks.append(mask)
    frames[0][15:18, 25:28] = (255, 0, 255)  # unmasked blob in one frame only: removed by the median
    plate = median_plate(frames, masks, dilate_px=3)
    assert np.array_equal(plate, background)


def test_ground_texture_samples_court_colours():
    cam = broadcast_camera()
    plate, _ = render_synthetic_frame(cam)
    texture = build_ground_texture(plate, cam, np.ones((H, W), bool), extent=TEST_EXTENT, resolution=0.1)
    assert texture.shape == (550, 800)
    colors = texture.sample(np.array([0.0, 8.0, HALF_DOUBLES]), np.array([-6.0, 0.0, -6.0]))
    assert np.abs(colors[0].astype(int) - COURT_BLUE).max() < 10
    assert np.abs(colors[1].astype(int) - SURROUND_GREEN).max() < 10
    assert colors[2].min() > 200 and np.abs(colors[2].astype(int) - LINE_WHITE).max() < 60
    assert np.abs(np.array(texture.fill) - SURROUND_GREEN).max() < 10
    points, point_colors = texture.world_points(stride=10)
    assert points.shape[1] == 3 and len(points) == len(point_colors) and np.allclose(points[:, 2], 0.0)


def test_background_cloud_keeps_only_the_wall():
    cam = broadcast_camera()
    plate, depth = render_synthetic_frame(cam)
    disparity = (1.0 / depth + 0.004) / 0.37
    points, colors, radii, ground_mask, alignment = background_cloud(
        plate, disparity, cam, ground_depth_map(cam), TEST_EXTENT, stride=2
    )
    assert alignment.scale == pytest.approx(0.37, rel=1e-3)
    assert len(points) > 1000
    assert np.all(np.abs(points[:, 1] - WALL_Y) < 0.5)
    assert np.all((points[:, 2] > -0.1) & (points[:, 2] < WALL_HEIGHT + 0.1))
    assert np.all(colors == WALL_RED)
    assert radii.min() > 0
    uv, _ = cam.project(np.array([[0.0, -6.0, 0.0]]))
    assert ground_mask[int(round(uv[0, 1])), int(round(uv[0, 0]))]


def test_reconstruct_scene_end_to_end(tmp_path):
    cam = broadcast_camera()
    people = [(1.0, -10.0, 1.85), (-2.0, 10.0, 1.90)]
    _, background_depth = render_synthetic_frame(cam)
    frame, _ = render_synthetic_frame(cam, people)
    video = tmp_path / "source.mp4"
    with H264Writer(video, W, H, 25.0) as writer:
        for _ in range(6):
            writer.write(frame)
    detections = Detections(
        W, H, 25.0, [[billboard_detection(cam, 5, *people[0]), billboard_detection(cam, 9, *people[1])] for _ in range(6)]
    )
    players = build_players(detections, cam, make_identity(5, 9))
    scene = reconstruct_scene(
        video, cam, players, detections, FakeEstimator(background_depth), stride=4, samples=6, extent=TEST_EXTENT, resolution=0.1
    )
    assert scene.frame_count == 6
    near, _, near_radii = scene.players_at(2, exclude_role="far")
    far, _, _ = scene.players_at(2, exclude_role="near")
    both, _, _ = scene.players_at(2)
    assert len(near) > 50 and len(far) > 20 and len(both) == len(near) + len(far)
    assert np.all(np.abs(near[:, 0] - 1.0) < 0.8) and np.all(np.abs(near[:, 1] + 10.0) < 1.5)
    assert np.all((near[:, 2] > -0.3) & (near[:, 2] < 2.2))
    assert np.all(np.abs(far[:, 1] - 10.0) < 1.5)
    assert near_radii.min() > 0
    assert np.all(np.abs(scene.background_points[:, 1] - WALL_Y) < 0.5)
    scene.save(tmp_path / "scene.npz")
    again = Scene.load(tmp_path / "scene.npz")
    assert np.array_equal(again.frame_offsets, scene.frame_offsets)
    assert again.ground.image.shape == scene.ground.image.shape
    assert again.fps == 25.0 and np.allclose(again.camera.K, cam.K)
