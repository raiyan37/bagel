import numpy as np
import pytest
from synthetic import broadcast_camera

from horizon.ball import BALL_RADIUS_M
from horizon.camera import PinholeCamera
from horizon.ground import GroundTexture
from horizon.reconstruct import Scene
from horizon.render import ball_cloud, render_frame, render_view, sky_gradient

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


def two_player_scene() -> Scene:
    """Two player points on the centre line, 5 m and 8 m out: near is RED, far is BLUE."""
    return Scene(
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


def test_render_frame_can_hide_the_viewer():
    scene = two_player_scene()
    cam = PinholeCamera.look_at((0.0, 0.0, 1.0), (0.0, 1.0, 1.0), 60.0, 64, 36)
    center = int(round(cam.cy)), int(round(cam.cx))
    assert tuple(render_frame(scene, 0, cam)[center]) == RED
    assert tuple(render_frame(scene, 0, cam, exclude_role="near")[center]) == BLUE


def ball_camera(width=160, height=90):
    """Eye at the origin at 1 m, looking down +Y. A ball on the +Y axis lands on the principal point."""
    return PinholeCamera.look_at((0.0, 0.0, 1.0), (0.0, 1.0, 1.0), 60.0, width, height)


def ball_pixel_count(image):
    """Pixels carrying a shaded tennis-ball colour. The RED and BLUE ground and players both fail the
    green test, and the sky is far too dark, so only the ball is counted."""
    pixels = image.astype(int)
    return int(((pixels[..., 0] > 110) & (pixels[..., 1] > 120) & (pixels[..., 2] < 150)).sum())


def test_ball_cloud_sits_on_a_sphere_of_the_right_size():
    center = np.array([1.0, 2.0, 1.5])
    points, colors, radii = ball_cloud(center)
    assert points.shape == (256, 3) and colors.shape == (256, 3) and radii.shape == (256,)
    assert points.dtype == np.float32 and colors.dtype == np.uint8 and radii.dtype == np.float32
    distances = np.linalg.norm(points.astype(np.float64) - center, axis=1)
    assert distances == pytest.approx(np.full(256, BALL_RADIUS_M), abs=1e-5)
    assert colors[:, 1].mean() > colors[:, 2].mean()  # yellow-green: more green than blue


def test_ball_cloud_is_shaded_brighter_on_top():
    points, colors, _radii = ball_cloud(np.zeros(3))
    top = points[:, 2] > 0
    assert colors[top].astype(int).mean() > colors[~top].astype(int).mean()


def test_render_frame_without_a_ball_is_unchanged():
    scene, cam = two_player_scene(), ball_camera()
    np.testing.assert_array_equal(render_frame(scene, 0, cam), render_frame(scene, 0, cam, ball_xyz=None))


def test_render_frame_draws_the_ball_where_it_is_projected():
    scene, cam = two_player_scene(), ball_camera()
    center = np.array([0.0, 0.6, 1.0])
    pixel = int(round(cam.cy)), int(round(cam.cx))
    plain = render_frame(scene, 0, cam)
    withball = render_frame(scene, 0, cam, ball_xyz=center)
    assert tuple(plain[pixel]) == RED  # the near player, until the ball gets in front of them
    assert tuple(withball[pixel]) != RED
    assert ball_pixel_count(withball) > 10 and ball_pixel_count(plain) == 0


def test_the_ball_grows_as_it_approaches():
    scene = two_player_scene()

    def count_at(distance):
        return ball_pixel_count(render_frame(scene, 0, ball_camera(), ball_xyz=np.array([0.0, distance, 1.0])))

    assert count_at(0.6) > count_at(2.5) > 0


def test_a_ball_behind_the_camera_draws_nothing():
    """Review Focus 5: nothing is smeared across the frame when the ball is not in front of the lens."""
    scene, cam = two_player_scene(), ball_camera()
    behind = np.array([0.0, -3.0, 1.0])  # the eye is at y = 0, looking towards +y
    np.testing.assert_array_equal(render_frame(scene, 0, cam), render_frame(scene, 0, cam, ball_xyz=behind))
