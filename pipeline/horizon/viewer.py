"""Viser app: the reconstructed point cloud, player POV frustums and a draggable free camera.

The free camera is a transform-controls gizmo with a frustum attached. It can snap to or follow either player
("eyes" = first person, "chase" = behind and above), be looked through live in the browser, preview-rendered with
our own renderer, and exported as free_cam.mp4 (frame-aligned with the broadcast clip).
"""

from __future__ import annotations

import threading
import time

import numpy as np

from horizon.camera import PinholeCamera, camera_from_pose, camera_pose, vertical_fov_from_horizontal
from horizon.export import PLAYER_COLORS
from horizon.ground import GroundTexture
from horizon.paths import MatchPaths
from horizon.players import Players
from horizon.pov import POV_HEIGHT, POV_HFOV_DEG, POV_WIDTH, pov_camera, render_clip
from horizon.reconstruct import Scene
from horizon.render import render_frame

FOLLOW_OPTIONS = ("off", "near eyes", "near chase", "far eyes", "far chase")
CHASE_DISTANCE = 2.5
CHASE_HEIGHT = 0.8
PREVIEW_W, PREVIEW_H = 512, 288
_HIDDEN_POINT = np.array([[0.0, 0.0, -100.0]], np.float32)


def free_camera(wxyz, position, hfov_deg: float, width: int, height: int) -> PinholeCamera:
    return camera_from_pose(wxyz, position, vertical_fov_from_horizontal(hfov_deg, width, height), width, height)


def look_target(camera: PinholeCamera, distance: float = 5.0) -> np.ndarray:
    return camera.center + distance * camera.forward


def snap_pose(players: Players, role: str, frame: int, hfov_deg: float = POV_HFOV_DEG) -> tuple[np.ndarray, np.ndarray]:
    return camera_pose(pov_camera(players, role, frame, hfov_deg))


def parse_follow(option: str) -> tuple[str, str] | None:
    if option == "off":
        return None
    if option not in FOLLOW_OPTIONS:
        raise ValueError(f"Unknown follow option {option!r}; expected one of {FOLLOW_OPTIONS}")
    role, style = option.split()
    return role, style


def excluded_role(option: str) -> str | None:
    parsed = parse_follow(option)
    return parsed[0] if parsed and parsed[1] == "eyes" else None


def follow_camera(players: Players, role: str, style: str, frame: int, hfov_deg: float, width: int, height: int) -> PinholeCamera:
    pov = pov_camera(players, role, frame, hfov_deg, width, height)
    if style == "eyes":
        return pov
    flat = pov.forward.copy()
    flat[2] = 0.0
    flat /= np.linalg.norm(flat)
    eye = pov.center - CHASE_DISTANCE * flat + np.array([0.0, 0.0, CHASE_HEIGHT])
    return PinholeCamera.look_at(eye, look_target(pov, 10.0), np.degrees(pov.vertical_fov), width, height)


def export_camera(option: str, players: Players, frame: int, static_pose, hfov_deg: float, width: int, height: int) -> PinholeCamera:
    parsed = parse_follow(option)
    if parsed is None:
        wxyz, position = static_pose
        return free_camera(wxyz, position, hfov_deg, width, height)
    role, style = parsed
    return follow_camera(players, role, style, frame, hfov_deg, width, height)


def downsample(points: np.ndarray, colors: np.ndarray, max_points: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    if len(points) <= max_points:
        return points, colors
    idx = np.sort(np.random.default_rng(seed).choice(len(points), max_points, replace=False))
    return points[idx], colors[idx]


def ground_cloud(texture: GroundTexture, spacing: float = 0.08) -> tuple[np.ndarray, np.ndarray]:
    return texture.world_points(stride=max(1, int(round(spacing / texture.resolution))))


def _rgb(hex_color: str) -> tuple[int, int, int]:
    return tuple(int(hex_color[i : i + 2], 16) for i in (1, 3, 5))


def run_viewer(paths: MatchPaths, host: str = "0.0.0.0", port: int = 8080) -> None:
    import viser

    scene = Scene.load(paths.scene)
    players = Players.load(paths.players)
    total = min(scene.frame_count, players.frame_count)
    server = viser.ViserServer(host=host, port=port)
    server.scene.set_up_direction("+z")
    server.initial_camera.position = tuple(float(v) for v in scene.camera.center)
    server.initial_camera.look_at = (0.0, 0.0, 0.0)

    ground_points, ground_colors = ground_cloud(scene.ground)
    server.scene.add_point_cloud("/ground", points=ground_points, colors=ground_colors, point_size=0.09, point_shape="square")
    bg_points, bg_colors = downsample(scene.background_points, scene.background_colors, 400_000)
    server.scene.add_point_cloud("/background", points=bg_points, colors=bg_colors, point_size=0.06, point_shape="rounded")
    player_cloud = server.scene.add_point_cloud(
        "/players", points=_HIDDEN_POINT, colors=np.zeros((1, 3), np.uint8), point_size=0.03, point_shape="rounded"
    )
    b_wxyz, b_pos = camera_pose(scene.camera)
    server.scene.add_camera_frustum(
        "/broadcast", fov=scene.camera.vertical_fov, aspect=scene.camera.width / scene.camera.height,
        scale=1.0, color=(200, 200, 200), wxyz=tuple(b_wxyz), position=tuple(b_pos),
    )
    pov_frustums = {
        role: server.scene.add_camera_frustum(
            f"/pov/{role}", fov=np.radians(vertical_fov_from_horizontal(POV_HFOV_DEG, POV_WIDTH, POV_HEIGHT)),
            aspect=POV_WIDTH / POV_HEIGHT, scale=0.5, color=_rgb(PLAYER_COLORS[role]),
        )
        for role in ("near", "far")
    }
    start_wxyz, start_pos = snap_pose(players, "near", 0)
    gizmo = server.scene.add_transform_controls("/free_cam", scale=1.2, wxyz=tuple(start_wxyz), position=tuple(start_pos))
    free_frustum = server.scene.add_camera_frustum(
        "/free_cam/frustum", fov=np.radians(vertical_fov_from_horizontal(POV_HFOV_DEG, 16, 9)), aspect=16 / 9,
        scale=0.6, color=(255, 255, 255),
    )

    with server.gui.add_folder("Playback"):
        frame_slider = server.gui.add_slider("Frame", min=0, max=max(total - 1, 1), step=1, initial_value=0)
        playing = server.gui.add_checkbox("Playing", initial_value=True)
        fps_slider = server.gui.add_slider("FPS", min=1, max=60, step=1, initial_value=int(round(scene.fps)))
    with server.gui.add_folder("Views"):
        view_broadcast = server.gui.add_button("Broadcast camera")
        view_near = server.gui.add_button("Near player's eyes")
        view_far = server.gui.add_button("Far player's eyes")
    with server.gui.add_folder("Free camera"):
        follow = server.gui.add_dropdown("Follow", FOLLOW_OPTIONS, initial_value="off")
        fov = server.gui.add_slider("FOV (horizontal)", min=30, max=110, step=1, initial_value=int(POV_HFOV_DEG))
        snap_near = server.gui.add_button("Snap to near player")
        snap_far = server.gui.add_button("Snap to far player")
        look_through = server.gui.add_button("Look through free camera")
        preview_button = server.gui.add_button("Render preview")
        preview = server.gui.add_image(np.zeros((PREVIEW_H, PREVIEW_W, 3), np.uint8), label="Free camera view", format="jpeg")
        export_button = server.gui.add_button("Export free-cam clip")
        status = server.gui.add_text("Status", initial_value="ready", disabled=True)

    lock = threading.Lock()
    busy = {"export": False}

    def place_gizmo(wxyz, position) -> None:
        with server.atomic():
            gizmo.wxyz = tuple(float(v) for v in wxyz)
            gizmo.position = tuple(float(v) for v in position)

    def current_free_camera(width: int, height: int) -> PinholeCamera:
        return free_camera(np.array(gizmo.wxyz), np.array(gizmo.position), fov.value, width, height)

    def look_through_camera(camera: PinholeCamera) -> None:
        for client in server.get_clients().values():
            with client.atomic():
                client.camera.fov = camera.vertical_fov
                client.camera.position = tuple(float(v) for v in camera.center)
                client.camera.look_at = tuple(float(v) for v in look_target(camera))

    def show_frame(frame: int) -> None:
        points, colors, _ = scene.players_at(frame)
        points, colors = downsample(points, colors, 60_000)
        if len(points) == 0:
            points, colors = _HIDDEN_POINT, np.zeros((1, 3), np.uint8)
        with lock, server.atomic():
            player_cloud.points = points
            player_cloud.colors = colors
            for role, frustum in pov_frustums.items():
                wxyz, position = snap_pose(players, role, frame)
                frustum.wxyz, frustum.position = tuple(wxyz), tuple(position)
            parsed = parse_follow(follow.value)
            if parsed is not None:
                camera = follow_camera(players, parsed[0], parsed[1], frame, fov.value, POV_WIDTH, POV_HEIGHT)
                wxyz, position = camera_pose(camera)
                gizmo.wxyz, gizmo.position = tuple(wxyz), tuple(position)

    frame_slider.on_update(lambda _: show_frame(int(frame_slider.value)))
    fov.on_update(lambda _: setattr(free_frustum, "fov", np.radians(vertical_fov_from_horizontal(fov.value, 16, 9))))
    view_broadcast.on_click(lambda _: look_through_camera(scene.camera))
    view_near.on_click(lambda _: look_through_camera(pov_camera(players, "near", int(frame_slider.value))))
    view_far.on_click(lambda _: look_through_camera(pov_camera(players, "far", int(frame_slider.value))))
    snap_near.on_click(lambda _: place_gizmo(*snap_pose(players, "near", int(frame_slider.value), fov.value)))
    snap_far.on_click(lambda _: place_gizmo(*snap_pose(players, "far", int(frame_slider.value), fov.value)))
    look_through.on_click(lambda _: look_through_camera(current_free_camera(POV_WIDTH, POV_HEIGHT)))

    @preview_button.on_click
    def _(_) -> None:
        camera = current_free_camera(PREVIEW_W, PREVIEW_H)
        preview.image = render_frame(scene, int(frame_slider.value), camera, exclude_role=excluded_role(follow.value))

    @export_button.on_click
    def _(_) -> None:
        if busy["export"]:
            return
        busy["export"] = True
        export_button.disabled = True
        fov.disabled = True
        option = follow.value
        static_pose = (np.array(gizmo.wxyz), np.array(gizmo.position))
        hfov = fov.value
        try:
            count = render_clip(
                scene,
                lambda f: export_camera(option, players, f, static_pose, hfov, POV_WIDTH, POV_HEIGHT),
                paths.free_cam_video,
                exclude_role=excluded_role(option),
                frame_count=total,
                log=lambda message: setattr(status, "value", message.strip()),
            )
            status.value = f"saved {paths.free_cam_video.name} ({count} frames); run: horizon export --match-id {paths.match_id}"
        except Exception as exc:  # show the failure in the GUI instead of killing the server thread
            status.value = f"export failed: {exc}"
        finally:
            busy["export"] = False
            export_button.disabled = False
            fov.disabled = False

    show_frame(0)
    print(f"Viewer running at http://localhost:{port} (Ctrl+C to stop)")
    while True:
        if playing.value and not busy["export"] and total > 1:
            frame_slider.value = (int(frame_slider.value) + 1) % total
        time.sleep(1.0 / max(float(fps_slider.value), 1.0))
