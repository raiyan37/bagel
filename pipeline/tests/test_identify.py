import json

import numpy as np
import pytest
from synthetic import H, W, broadcast_camera, person_detection

from horizon import identify
from horizon.calibration import Calibration, save_calibration
from horizon.cli import main
from horizon.llm.backboard import run_tool_loop
from horizon.llm.pegasus import ClipAnalysis, PlayerDescription
from horizon.tracking import Detections
from horizon.video import H264Writer


def sample_analysis() -> ClipAnalysis:
    return ClipAnalysis(
        near=PlayerDescription("Carlos Alcaraz", "white shirt"),
        far=PlayerDescription("Jannik Sinner", "navy shirt"),
        score="6-4 2-1",
        summary="Rally.",
    )


def scene_detections(frames: int = 3):
    cam = broadcast_camera()
    people = [
        person_detection(cam, 5, 1.0, -10.0),  # near player
        person_detection(cam, 9, -2.0, 10.0, stature=1.9),  # far player
        person_detection(cam, 2, 6.6, -12.5, stature=0.9),  # crouching ball kid outside the court
        person_detection(cam, 4, 7.5, 0.0, stature=1.8),  # chair umpire at the net
    ]
    return cam, Detections(W, H, 25.0, [list(people) for _ in range(frames)])


class FakeBackboard:
    def __init__(self, first_response, final_response):
        self.first, self.final = first_response, final_response
        self.created = 0
        self.submitted = []

    def create_assistant(self, name, system_prompt, tools):
        self.created += 1
        return "asst-1"

    def create_thread(self, assistant_id):
        return "thread-1"

    def send_message(self, thread_id, content, **kwargs):
        return self.first

    def submit_tool_outputs(self, thread_id, run_id, tool_outputs):
        self.submitted.append((run_id, tool_outputs))
        return self.final


def patch_services(monkeypatch):
    monkeypatch.setattr(identify.pegasus, "analyze_clip", lambda path: sample_analysis())
    monkeypatch.setattr(
        identify.gemini,
        "locate_players",
        lambda frame, near, far: {"near": (1.0, 2.0, 3.0, 4.0), "far": (5.0, 6.0, 7.0, 8.0)},
    )


def test_iou_and_box_matching():
    assert identify.iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)
    _, dets = scene_detections(1)
    shifted = tuple(v + 3 for v in dets.frames[0][1].bbox)
    assert identify.match_box(shifted, dets.frames[0], W, H).track_id == 9
    assert identify.match_box((0, 0, 5, 5), dets.frames[0], W, H) is None


def test_heuristic_picks_persistent_track_in_each_half():
    cam, dets = scene_detections()
    assert identify.heuristic_tracks(dets, cam) == {"near": 5, "far": 9}


def test_resolve_uses_gemini_boxes_when_consistent():
    cam, dets = scene_detections()
    boxes = {"near": tuple(v + 2 for v in dets.frames[0][0].bbox), "far": dets.frames[0][1].bbox}
    identity = identify.resolve_identity(sample_analysis(), boxes, dets, cam)
    assert identity.players["near"].track_id == 5 and identity.players["near"].source == "gemini"
    assert identity.players["far"].track_id == 9 and identity.players["far"].name == "Jannik Sinner"
    assert identity.warnings == []


def test_resolve_falls_back_when_gemini_box_is_on_the_wrong_half():
    cam, dets = scene_detections()
    far_box = dets.frames[0][1].bbox
    identity = identify.resolve_identity(sample_analysis(), {"near": far_box, "far": far_box}, dets, cam)
    assert identity.players["near"].track_id == 5 and identity.players["near"].source == "heuristic"
    assert identity.players["far"].track_id == 9 and identity.players["far"].source == "gemini"
    assert len(identity.warnings) == 1


def test_run_tool_loop_executes_calls_and_submits_outputs():
    first = {
        "status": "REQUIRES_ACTION",
        "run_id": "run-1",
        "tool_calls": [
            {"id": "c1", "function": {"name": "analyze_tennis_clip", "arguments": json.dumps({"match_id": "demo"})}},
            {"id": "c2", "function": {"name": "locate_players", "arguments": {"near_appearance": "white"}}},
            {"id": "c3", "function": {"name": "delete_everything", "arguments": "{}"}},
        ],
    }
    fake = FakeBackboard(first, {"status": "COMPLETED", "content": "done"})
    seen = {}

    def analyze(args):
        seen["analyze"] = args
        return {"ok": True}

    def locate(args):
        seen["locate"] = args
        return {"near": [1, 2, 3, 4]}

    result = run_tool_loop(fake, "thread-1", "go", {"analyze_tennis_clip": analyze, "locate_players": locate})
    assert result.content == "done"
    assert seen == {"analyze": {"match_id": "demo"}, "locate": {"near_appearance": "white"}}
    run_id, outputs = fake.submitted[0]
    assert run_id == "run-1"
    assert [o["tool_call_id"] for o in outputs] == ["c1", "c2", "c3"]
    assert json.loads(outputs[2]["output"]) == {"error": "unknown tool 'delete_everything'"}
    assert [e.name for e in result.executions if e.error] == ["delete_everything"]


def test_identify_via_backboard_runs_tools_and_caches_assistant(tmp_path, monkeypatch):
    patch_services(monkeypatch)
    first = {
        "run_id": "r",
        "tool_calls": [
            {"id": "a", "function": {"name": "analyze_tennis_clip", "arguments": "{}"}},
            {"id": "b", "function": {"name": "locate_players", "arguments": "{}"}},
        ],
    }
    fake = FakeBackboard(first, {"content": "done"})
    cache = tmp_path / "assistant.json"
    frame = np.zeros((4, 4, 3), np.uint8)
    analysis, boxes, notes = identify.identify_via_backboard(fake, "demo", tmp_path / "source.mp4", frame, cache)
    assert analysis == sample_analysis()
    assert boxes["far"] == (5.0, 6.0, 7.0, 8.0)
    assert notes == []
    identify.identify_via_backboard(fake, "demo", tmp_path / "source.mp4", frame, cache)
    assert fake.created == 1


def test_identify_via_backboard_calls_services_directly_when_tools_are_skipped(tmp_path, monkeypatch):
    patch_services(monkeypatch)
    fake = FakeBackboard({"content": "I cannot help with that"}, {})
    frame = np.zeros((4, 4, 3), np.uint8)
    analysis, boxes, notes = identify.identify_via_backboard(fake, "demo", tmp_path / "s.mp4", frame, tmp_path / "c.json")
    assert analysis.near.name == "Carlos Alcaraz"
    assert boxes["near"] == (1.0, 2.0, 3.0, 4.0)
    assert len(notes) == 2


def test_identify_command_without_ai_services(tmp_path, monkeypatch):
    monkeypatch.setenv("HORIZON_DATA_ROOT", str(tmp_path))
    cam, dets = scene_detections()
    match = tmp_path / "unit"
    match.mkdir()
    with H264Writer(match / "source.mp4", W, H, 25.0) as writer:
        writer.write(np.zeros((H, W, 3), dtype=np.uint8))
    dets.save(match / "detections.json")
    save_calibration(Calibration(camera=cam, keypoints={}, rms_px=0.0), match / "calibration.json")
    assert main(["identify", "--match-id", "unit", "--orchestrator", "none"]) == 0
    identity = identify.Identity.load(match / "identity.json")
    assert identity.players["near"].track_id == 5
    assert identity.players["far"].track_id == 9
    assert identity.players["near"].name == "Near player"
