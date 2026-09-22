"""`horizon track`: detect and track every person in source.mp4."""

from __future__ import annotations

import argparse

from horizon.paths import MatchPaths
from horizon.tracking import DEFAULT_MODEL, run_tracking
from horizon.video import load_meta


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("track", help="Detect and track people (YOLO-seg + ByteTrack)")
    p.add_argument("--match-id", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL, help="Ultralytics weights, e.g. yolo26s-seg.pt or yolo11s-seg.pt")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.2)
    p.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    paths = MatchPaths.for_match(args.match_id)
    detections = run_tracking(paths.source_video, load_meta(paths.meta), args.model, args.imgsz, args.conf)
    detections.save(paths.detections)
    ids = {d.track_id for frame in detections.frames for d in frame}
    print(f"Wrote {paths.detections}: {len(detections.frames)} frames, {len(ids)} distinct track ids")
    return 0
