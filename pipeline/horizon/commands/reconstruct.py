"""`horizon reconstruct`: Depth Anything V2 point clouds + ground texture -> scene.npz, ground.png."""

from __future__ import annotations

import argparse

import cv2

from horizon.calibration import load_calibration
from horizon.depth import DEFAULT_DEPTH_MODEL, DepthAnythingV2
from horizon.paths import MatchPaths
from horizon.players import Players
from horizon.reconstruct import reconstruct_scene
from horizon.tracking import Detections


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("reconstruct", help="Build the 3D point cloud (monocular depth) and ground texture")
    p.add_argument("--match-id", required=True)
    p.add_argument("--depth-model", default=DEFAULT_DEPTH_MODEL, help="Hugging Face id of a relative Depth Anything V2 model")
    p.add_argument("--stride", type=int, default=2, help="Background pixel stride")
    p.add_argument("--samples", type=int, default=24, help="Frames used for the clean plate")
    p.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    paths = MatchPaths.for_match(args.match_id)
    scene = reconstruct_scene(
        paths.source_video,
        load_calibration(paths.calibration).camera,
        Players.load(paths.players),
        Detections.load(paths.detections),
        DepthAnythingV2(args.depth_model),
        stride=args.stride,
        samples=args.samples,
        log=print,
    )
    scene.save(paths.scene)
    cv2.imwrite(str(paths.ground_png), cv2.cvtColor(scene.ground.image, cv2.COLOR_RGB2BGR))
    per_frame = len(scene.player_points) / max(scene.frame_count, 1)
    print(
        f"Wrote {paths.scene}: {len(scene.background_points)} background points, "
        f"{per_frame:.0f} player points/frame over {scene.frame_count} frames; ground texture {paths.ground_png}"
    )
    return 0
