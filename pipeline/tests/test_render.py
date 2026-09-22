import numpy as np
from synthetic import broadcast_camera

from horizon.camera import PinholeCamera
from horizon.ground import GroundTexture
from horizon.reconstruct import Scene
from horizon.render import render_frame, render_view, sky_gradient

RED, BLUE = (220, 30, 30), (30, 30, 220)


def split_texture() -> GroundTexture:
    image = np.zeros((400, 200, 3), np.uint8)
    image[:, :100] = RED  # x < 0
    image[:, 100:] = BLUE  # x >= 0
    return GroundTexture(image, -10.0, 10.0, -20.0, 20.0, (0, 0, 0))


def test_ground_pass_draws_texture_below_horizon_and_sky_above():
    cam = PinholeCamera.look_at((0.0, -15.0, 1.7), (0.0, 0.0, 1.7), 50.0, 320, 180)
    image = render_view(cam, split_texture(), np.zeros((0, 3)), np.zeros((0, 3), np.uint8), np.zeros(0))
    assert tuple(image[-1, 5]) == RED
    assert tuple(image[-1, -5]) == BLUE
    assert tuple(image[0, 160]) == tuple(sky_gradient(320, 180)[0, 160])


def test_nearest_point_wins_regardless_of_order():
    cam = PinholeCamera.look_at((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), 60.0, 64, 36)
    points = np.array([[0.0, 10.0, 0.0], [0.0, 5.0, 0.0]])  # far blue, near red
    radii = np.full(2, 0.01)
    image = render_view(cam, None, points, np.array([BLUE, RED], np.uint8), radii)
    assert tuple(image[int(round(cam.cy)), int(round(cam.cx))]) == RED
    image = render_view(cam, None, points[::-1], np.array([RED, BLUE], np.uint8), radii)
    assert tuple(image[int(round(cam.cy)), int(round(cam.cx))]) == RED


def test_ground_hides_points_below_it():
    cam = PinholeCamera.look_at((0.0, -10.0, 1.7), (0.0, 0.0, 0.5), 60.0, 160, 90)
    under, above = np.array([[0.0, 0.0, -1.0]]), np.array([[0.0, 0.0, 1.0]])
    for pts, visible in ((under, False), (above, True)):
        image = render_view(cam, split_texture(), pts, np.array([(0, 255, 0)], np.uint8), np.array([0.02]))
        uv, _ = cam.project(pts)
        pixel = tuple(image[int(round(uv[0, 1])), int(round(uv[0, 0]))])
        assert (pixel == (0, 255, 0)) == visible


def test_splats_grow_when_the_camera_is_close():
    cam = PinholeCamera.look_at((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), 60.0, 640, 360)
    counts = []
    for distance in (2.0, 20.0):
        image = render_view(cam, None, np.array([[0.0, distance, 0.0]]), np.array([(0, 255, 0)], np.uint8), np.array([0.05]))
        counts.append(int(np.all(image == (0, 255, 0), axis=2).sum()))
    assert counts == [49, 4]


def test_render_frame_can_hide_the_viewer():
    scene = Scene(
        background_points=np.zeros((0, 3), np.float32),
        background_colors=np.zeros((0, 3), np.uint8),
        background_radii=np.zeros(0, np.float32),
        player_points=np.array([[0.0, 5.0, 1.0], [0.0, 8.0, 1.0]], np.float32),
        player_colors=np.array([RED, BLUE], np.uint8),
        player_radii=np.full(2, 0.05, np.float32),
        player_roles=np.array([0, 1], np.uint8),
        frame_offsets=np.array([0, 2]),
        ground=split_texture(),
        camera=broadcast_camera(),
        fps=25.0,
    )
    cam = PinholeCamera.look_at((0.0, 0.0, 1.0), (0.0, 1.0, 1.0), 60.0, 64, 36)
    center = int(round(cam.cy)), int(round(cam.cx))
    assert tuple(render_frame(scene, 0, cam)[center]) == RED
    assert tuple(render_frame(scene, 0, cam, exclude_role="near")[center]) == BLUE
