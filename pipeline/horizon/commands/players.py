"""`horizon players`: metric trajectories, stature, speed and distance for both players."""

from __future__ import annotations

import argparse

from horizon.calibration import load_calibration
from horizon.identify import Identity
from horizon.paths import MatchPaths
from horizon.players import build_players
from horizon.tracking import Detections


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("players", help="Build per-frame court trajectories for both players")
    p.add_argument("--match-id", required=True)
    p.add_argument("--smooth", type=int, default=7, help="Moving-average window in frames (odd)")
    p.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    paths = MatchPaths.for_match(args.match_id)
    players = build_players(
        Detections.load(paths.detections),
        load_calibration(paths.calibration).camera,
        Identity.load(paths.identity),
        smooth_window=args.smooth,
    )
    players.save(paths.players)
    for role, track in players.tracks.items():
        print(
            f"{role}: {track.name}, stature {track.stature_m:.2f} m, detected in {sum(track.visible)}/"
            f"{players.frame_count} frames, ran {track.distance_m[-1]:.1f} m, top speed {track.speed_kmh.max():.1f} km/h"
        )
    return 0
