"""`horizon identify`: which tracked person is the near player and which is the far player."""

from __future__ import annotations

import argparse
import os

from horizon.calibration import load_calibration
from horizon.identify import identify_direct, identify_via_backboard, resolve_identity
from horizon.llm.backboard import BackboardClient
from horizon.llm.pegasus import ClipAnalysis, PlayerDescription
from horizon.paths import MatchPaths, resolve_data_root
from horizon.tracking import Detections
from horizon.video import read_frame


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("identify", help="Identify the near/far players (Pegasus + Gemini via Backboard)")
    p.add_argument("--match-id", required=True)
    p.add_argument(
        "--orchestrator",
        choices=("backboard", "direct", "none"),
        default="backboard",
        help="backboard: assistant tool calls; direct: call Pegasus + Gemini; none: court geometry only",
    )
    p.add_argument("--frame", type=int, default=0, help="Frame that Gemini inspects")
    p.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    paths = MatchPaths.for_match(args.match_id)
    detections = Detections.load(paths.detections)
    camera = load_calibration(paths.calibration).camera
    notes: list[str] = []
    if args.orchestrator == "none":
        analysis = ClipAnalysis(PlayerDescription("Near player", ""), PlayerDescription("Far player", ""), "", "")
        boxes: dict = {}
    else:
        frame = read_frame(paths.source_video, args.frame)
        if args.orchestrator == "direct":
            analysis, boxes = identify_direct(paths.source_video, frame)
        else:
            analysis, boxes, notes = identify_via_backboard(
                BackboardClient(),
                args.match_id,
                paths.source_video,
                frame,
                resolve_data_root() / ".backboard_assistant.json",
                llm_provider=os.getenv("BACKBOARD_LLM_PROVIDER") or None,
                model_name=os.getenv("BACKBOARD_MODEL_NAME") or None,
            )
    identity = resolve_identity(analysis, boxes, detections, camera, frame_index=args.frame)
    identity.warnings[:0] = notes
    identity.save(paths.identity)
    for role, player in identity.players.items():
        print(f"{role}: {player.name} -> track {player.track_id} ({player.source})")
    for warning in identity.warnings:
        print(f"WARNING: {warning}")
    return 1 if any(p.track_id is None for p in identity.players.values()) else 0
