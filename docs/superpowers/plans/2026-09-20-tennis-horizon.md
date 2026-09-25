# Project Horizon Tennis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the Project Horizon soccer demo into a tennis spin-off. A broadcast tennis clip is reconstructed in 3D with a monocular depth model. The project renders first-person POV videos for both players, provides a free-placement 3D camera, and presents everything in the Horizon web UI.

**Architecture:** An offline Python pipeline (`pipeline/`, package `horizon`, CLI `horizon`) turns one clip into artifacts in stages:
- court calibration
- YOLO tracking
- player identification (Pegasus + Gemini via Backboard)
- metric trajectories
- Depth Anything V2 point cloud with a textured ground plane
- POV rendering
- export to `public/matches/<id>/`

A Viser app (`horizon view`) hosts the free camera. The React app reads the exported manifest and embeds the Viser viewer for FREE CAM.

**Tech Stack:**
- Python 3.11: numpy, OpenCV, imageio-ffmpeg, PyTorch (CUDA 12.8 wheels), transformers (Depth Anything V2), Ultralytics YOLO26-seg + ByteTrack, boto3 (Bedrock Pegasus), requests (OpenRouter Gemini, Backboard), viser, pytest.
- Web: React 19 + Vite 7 + TypeScript 5.9, Vitest 4 + Testing Library.

**Spec:** `docs/superpowers/specs/2026-09-20-tennis-horizon-design.md` (read it first; §5.1 conventions and §6 data contracts are normative)

## Global Constraints

- Work happens in the spin-off copy `C:\Users\rhaqu\Desktop\ProjectHorizon-Tennis`. The original `ProjectHorizon-main` folder is never modified.
- The shell is Windows PowerShell 5.1: no `&&`, and paths use `\`. Python always runs as `.venv\Scripts\python` from `pipeline\`.
- Python `>=3.11,<3.13`. The venv is created with `uv venv --python 3.11 .venv`.
- Automated tests never need a GPU, network access, API keys or model downloads. `torch`, `transformers`, `ultralytics`, `viser` and `boto3` are imported lazily inside the functions that use them.
- World frame: metres, origin at the court centre on the ground, +X across (right as seen from the broadcast camera), +Y towards the far baseline, +Z up.
- Camera frame: OpenCV (+X right, +Y down, +Z forward), `x_cam = R·x_world + t`, principal point `((w−1)/2, (h−1)/2)`.
- Player roles are exactly `near` and `far`. Colours: near `#3B82F6`, far `#F97316`.
- Every video the pipeline writes is H.264 `yuv420p` MP4 with `+faststart`, even width/height, and the same fps and frame count as `source.mp4`.
- Secrets come only from `pipeline\.env`: `AWS_REGION`, `TWELVELABS_MODEL_ID`, `OPENROUTER_API_KEY`, `GEMINI_MODEL_ID`, `BACKBOARD_API_KEY`, `BACKBOARD_LLM_PROVIDER`, `BACKBOARD_MODEL_NAME`. Never print their values.
- The JSON artifact shapes in spec §6 are a contract between the pipeline and the web app. Do not rename fields.
- Web code must pass `npm run build` (`tsc -b` strict with `noUnusedLocals`/`noUnusedParameters`) and `npm run lint`.
- Commit at the end of every task with a conventional-commit message.

## File Structure

```
ProjectHorizon-Tennis/
├── .gitignore                          (modify: Python + generated media ignores)
├── README.md                           (rewrite in Task 17)
├── docs/superpowers/{specs,plans}/…    (this plan + spec)
├── pipeline/
│   ├── pyproject.toml                  Task 1  – package, extras ml/viewer/dev, `horizon` script
│   ├── .env.example                    Task 1  – documented env vars
│   ├── horizon/
│   │   ├── __init__.py, __main__.py    Task 1
│   │   ├── env.py                      Task 1  – loads pipeline/.env
│   │   ├── paths.py                    Task 1  – MatchPaths: per-match artifact locations
│   │   ├── cli.py                      Task 1  – argparse root, auto-discovers horizon/commands/*
│   │   ├── court.py                    Task 2  – ITF court model (keypoints, lines, net)
│   │   ├── camera.py                   Task 2  – PinholeCamera, look_at, rays, stature, quaternions
│   │   ├── video.py                    Task 3  – probe, iter_frames, H264Writer, transcode_clip
│   │   ├── calibration.py              Task 4  – keypoints → camera (IPPE + focal search), overlay
│   │   ├── tracking.py                 Task 5  – Detection(s), YOLO-seg + ByteTrack wrapper
│   │   ├── llm/pegasus.py              Task 6  – TwelveLabs Pegasus via Bedrock
│   │   ├── llm/gemini.py               Task 6  – Gemini box_2d via OpenRouter
│   │   ├── llm/backboard.py            Task 7  – Backboard REST client + tool loop
│   │   ├── identify.py                 Task 7  – identity.json (LLM + geometric validation)
│   │   ├── players.py                  Task 8  – per-frame association, smoothing, stature, speed
│   │   ├── depth.py                    Task 9  – Depth Anything wrapper + plane alignment
│   │   ├── ground.py                   Task 10 – GroundTexture (orthophoto)
│   │   ├── reconstruct.py              Task 10 – clean plate, background + player point clouds
│   │   ├── render.py                   Task 11 – ground pass + z-buffered point splatting
│   │   ├── pov.py                      Task 12 – POV cameras + clip rendering
│   │   ├── export.py                   Task 13 – manifest/tracks/index for the web app
│   │   ├── viewer.py                   Task 14 – Viser free-camera app
│   │   └── commands/                   one module per CLI sub-command (doctor, init, calibrate,
│   │                                   track, identify, players, reconstruct, render, export, view, all)
│   └── tests/                          one test module per horizon module + synthetic fixtures
├── src/
│   ├── lib/match.ts                    Task 15 – types + pure helpers (frame lookup, cover transform, sync)
│   ├── hooks/useMatchData.ts           Task 15 – fetch manifest/tracks/index
│   ├── test/setup.ts, test/fixtures.ts Task 15
│   ├── components/tennis/*             Task 16 – ScoreOverlay, PovVideo, PlayerCard, PovOverlay, FreeCamOverlay
│   └── pages/{GamesPage,ProcessingPage,StreamPage}.tsx   Task 17 (modify/rewrite)
└── vite.config.ts, package.json        Task 15 (Vitest)
```

Removed in Tasks 17–18 (soccer-only): `backend/`, `preprocessing/`, `public/bruno_tracks.json`, `src/hooks/usePlayerTracking.ts`, `src/components/player/{PlayerCard,POVVideoOverlay}.tsx`, `src/components/stream/{PenaltyScoreOverlay,GoalConfetti}.tsx` + their CSS, and the soccer `.mov`/`.png` assets.

---

## Phase A — Reconstruction pipeline

### Task 1: Spin-off workspace and pipeline scaffold

**Files:**
- Create: `pipeline/pyproject.toml`, `pipeline/.env.example`, `pipeline/horizon/__init__.py`, `pipeline/horizon/__main__.py`, `pipeline/horizon/env.py`, `pipeline/horizon/paths.py`, `pipeline/horizon/cli.py`, `pipeline/horizon/commands/__init__.py`, `pipeline/horizon/commands/doctor.py`
- Modify: `.gitignore`
- Test: `pipeline/tests/test_cli.py`, `pipeline/tests/test_paths.py`

**Interfaces:**
- Produces:
  - `horizon.paths.validate_match_id(match_id: str) -> str` (raises `ValueError`)
  - `horizon.paths.MatchPaths.for_match(match_id: str, data_root: Path | None = None) -> MatchPaths`, with properties `root, match_id, source_video, meta, calibration, calibration_preview, detections, identity, players, scene, ground_png, free_cam_video`, method `pov_video(role: str) -> Path`, and `ensure() -> MatchPaths`
  - `horizon.paths.DATA_ROOT: Path` (= `pipeline/data`)
  - `horizon.paths.resolve_data_root() -> Path`: `$HORIZON_DATA_ROOT` if set, else `DATA_ROOT`. Tests point it at `tmp_path`.
  - `horizon.paths.PIPELINE_ROOT: Path` (= `pipeline/`)
  - `horizon.paths.REPO_ROOT: Path`
  - `horizon.cli.build_parser() -> argparse.ArgumentParser`
  - `horizon.cli.main(argv: list[str] | None = None) -> int`
  - Command-module contract: every `horizon/commands/<name>.py` defines `register(sub: argparse._SubParsersAction) -> None`, which calls `p.set_defaults(handler=run)`, and `run(args: argparse.Namespace) -> int`.

- [ ] **Step 1: Install Git (not currently installed) and confirm the toolchain**

Run (PowerShell):
```powershell
winget install --id Git.Git -e --source winget
```
Close and reopen PowerShell, then:
```powershell
git --version; node --version; uv --version; py -3.11 --version
```
Expected: four version lines (git 2.x, node v24.x, uv 0.x, Python 3.11.x).

- [ ] **Step 2: Copy the project into the spin-off folder**

```powershell
Copy-Item -Recurse -Path C:\Users\rhaqu\Desktop\ProjectHorizon-main\ProjectHorizon-main -Destination C:\Users\rhaqu\Desktop\ProjectHorizon-Tennis
Set-Location C:\Users\rhaqu\Desktop\ProjectHorizon-Tennis
Get-ChildItem
```
Expected: the same top-level entries as the original (`src`, `backend`, `preprocessing`, `public`, `docs`, `package.json`, …).

- [ ] **Step 3: Extend `.gitignore`**

Append these lines to the end of `.gitignore`:
```gitignore

# Python pipeline
__pycache__/
*.egg-info/
.pytest_cache/
pipeline/.venv/
pipeline/.env
pipeline/data/

# Generated match media (written by `horizon export`)
public/matches/
```

- [ ] **Step 4: Create the git repository with the untouched baseline**

```powershell
git init
git config user.name
```
If `git config user.name` prints nothing, run `git config --global user.name "Your Name"` and `git config --global user.email "h.raiyanul@gmail.com"` first. Then:
```powershell
git add -A
git commit -m "chore: import Project Horizon soccer baseline"
```
Expected: a single root commit. (The `*.mov` ignore rule keeps the soccer videos out of git. They still exist on disk.)

- [ ] **Step 5: Write `pipeline/pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=69"]
build-backend = "setuptools.build_meta"

[project]
name = "horizon-tennis"
version = "0.1.0"
description = "Project Horizon Tennis: monocular 3D reconstruction, player POV cameras and a free camera for broadcast tennis clips"
requires-python = ">=3.11,<3.13"
dependencies = [
  "numpy>=1.26",
  "opencv-python>=4.9",
  "imageio-ffmpeg>=0.5.1",
  "pillow>=10.0",
  "requests>=2.31",
  "python-dotenv>=1.0",
  "boto3>=1.34",
]

[project.optional-dependencies]
ml = ["torch>=2.4", "transformers>=4.45", "ultralytics>=8.3", "lap>=0.5.12"]
viewer = ["viser>=1.0"]
dev = ["pytest>=8.0"]

[project.scripts]
horizon = "horizon.cli:main"

[tool.setuptools.packages.find]
include = ["horizon*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```
(`lap` is listed explicitly because Ultralytics' ByteTrack otherwise tries to pip-install it at runtime, and `uv` venvs have no pip.)

- [ ] **Step 6: Write `pipeline/.env.example`**

```dotenv
# Copy to pipeline/.env and fill in. Never commit .env.
# TwelveLabs Pegasus on Amazon Bedrock (AWS credentials come from `aws configure` or AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY)
AWS_REGION=us-east-1
TWELVELABS_MODEL_ID=us.twelvelabs.pegasus-1-2-v1:0
# Gemini through OpenRouter (check https://openrouter.ai/google for current model ids)
OPENROUTER_API_KEY=
GEMINI_MODEL_ID=google/gemini-3.5-flash
# Backboard orchestration (leave provider/model empty to use the assistant default)
BACKBOARD_API_KEY=
BACKBOARD_LLM_PROVIDER=
BACKBOARD_MODEL_NAME=
```

- [ ] **Step 7: Create the virtual environment and install dependencies**

```powershell
Set-Location C:\Users\rhaqu\Desktop\ProjectHorizon-Tennis\pipeline
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv\Scripts\python.exe pytest
```
Expected: the torch wheel installs (≈2.5 GB download) along with pytest. The package files needed for `pip install -e` are created in the next steps, so the editable install happens in Step 11.

- [ ] **Step 8: Write the failing tests**

`pipeline/tests/test_paths.py`:
```python
import pytest

from horizon.paths import MatchPaths, validate_match_id


def test_match_paths_layout(tmp_path):
    paths = MatchPaths.for_match("demo-1", data_root=tmp_path).ensure()
    assert paths.root == tmp_path / "demo-1"
    assert paths.root.is_dir()
    assert paths.source_video.name == "source.mp4"
    assert paths.calibration.name == "calibration.json"
    assert paths.pov_video("near").name == "pov_near.mp4"
    assert paths.free_cam_video.name == "free_cam.mp4"


@pytest.mark.parametrize("bad", ["", "Demo", "../x", "a b", "x" * 65, "-lead"])
def test_invalid_match_ids_are_rejected(bad):
    with pytest.raises(ValueError):
        validate_match_id(bad)
```

`pipeline/tests/test_cli.py`:
```python
from horizon.cli import build_parser, main


def test_parser_exposes_doctor_command():
    args = build_parser().parse_args(["doctor"])
    assert args.command == "doctor"
    assert callable(args.handler)


def test_doctor_reports_tools_without_leaking_secrets(capsys, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "super-secret-value")
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "ffmpeg:" in out
    assert "OPENROUTER_API_KEY: set" in out
    assert "super-secret-value" not in out
```

- [ ] **Step 9: Run the tests and confirm they fail**

Run: `.venv\Scripts\python -m pytest -q`
Expected: collection errors, `ModuleNotFoundError: No module named 'horizon'`.

- [ ] **Step 10: Implement the package skeleton**

`pipeline/horizon/__init__.py`:
```python
"""Project Horizon Tennis reconstruction pipeline."""

__version__ = "0.1.0"
```

`pipeline/horizon/__main__.py`:
```python
from horizon.cli import main

raise SystemExit(main())
```

`pipeline/horizon/paths.py`:
```python
"""Where every per-match artifact lives."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PIPELINE_ROOT.parent
DATA_ROOT = PIPELINE_ROOT / "data"

_MATCH_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


def resolve_data_root() -> Path:
    """Data directory for match artifacts; HORIZON_DATA_ROOT overrides it (tests use tmp dirs)."""
    return Path(os.environ.get("HORIZON_DATA_ROOT") or DATA_ROOT)


def validate_match_id(match_id: str) -> str:
    if not _MATCH_ID.fullmatch(match_id or ""):
        raise ValueError(
            f"Invalid match id {match_id!r}: use 1-64 lowercase letters, digits or '-', not starting with '-'"
        )
    return match_id


@dataclass(frozen=True)
class MatchPaths:
    root: Path

    @classmethod
    def for_match(cls, match_id: str, data_root: Path | None = None) -> "MatchPaths":
        root = Path(data_root) if data_root is not None else resolve_data_root()
        return cls(root / validate_match_id(match_id))

    def ensure(self) -> "MatchPaths":
        self.root.mkdir(parents=True, exist_ok=True)
        return self

    @property
    def match_id(self) -> str:
        return self.root.name

    @property
    def source_video(self) -> Path:
        return self.root / "source.mp4"

    @property
    def meta(self) -> Path:
        return self.root / "meta.json"

    @property
    def calibration(self) -> Path:
        return self.root / "calibration.json"

    @property
    def calibration_preview(self) -> Path:
        return self.root / "calibration_preview.jpg"

    @property
    def detections(self) -> Path:
        return self.root / "detections.json"

    @property
    def identity(self) -> Path:
        return self.root / "identity.json"

    @property
    def players(self) -> Path:
        return self.root / "players.json"

    @property
    def scene(self) -> Path:
        return self.root / "scene.npz"

    @property
    def ground_png(self) -> Path:
        return self.root / "ground.png"

    @property
    def free_cam_video(self) -> Path:
        return self.root / "free_cam.mp4"

    def pov_video(self, role: str) -> Path:
        return self.root / f"pov_{role}.mp4"
```

`pipeline/horizon/env.py`:
```python
"""Load pipeline/.env without overriding variables already set in the shell."""

from __future__ import annotations

from dotenv import load_dotenv

from horizon.paths import PIPELINE_ROOT


def load_env() -> None:
    load_dotenv(PIPELINE_ROOT / ".env", override=False)
```

`pipeline/horizon/cli.py`:
```python
"""`horizon` command line. Sub-commands live in horizon/commands/*.py and are discovered automatically."""

from __future__ import annotations

import argparse
import importlib
import pkgutil

import horizon.commands as commands_pkg
from horizon.env import load_env


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="horizon", description="Project Horizon Tennis pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    for info in sorted(pkgutil.iter_modules(commands_pkg.__path__), key=lambda m: m.name):
        module = importlib.import_module(f"horizon.commands.{info.name}")
        module.register(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    load_env()
    args = build_parser().parse_args(argv)
    return int(args.handler(args) or 0)
```

`pipeline/horizon/commands/__init__.py`:
```python
"""CLI sub-commands. Each module exposes register(sub) and run(args)."""
```

`pipeline/horizon/commands/doctor.py`:
```python
"""`horizon doctor`: report tool availability and which secrets are configured (never their values)."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys

from horizon.paths import resolve_data_root

SECRETS = (
    "AWS_REGION",
    "TWELVELABS_MODEL_ID",
    "OPENROUTER_API_KEY",
    "GEMINI_MODEL_ID",
    "BACKBOARD_API_KEY",
)


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("doctor", help="Check the pipeline environment")
    p.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    import imageio_ffmpeg

    print(f"python: {sys.version.split()[0]}")
    print(f"ffmpeg: {imageio_ffmpeg.get_ffmpeg_exe()}")
    for module in ("torch", "transformers", "ultralytics", "viser", "boto3"):
        status = "installed" if importlib.util.find_spec(module) else "MISSING"
        print(f"{module}: {status}")
    if importlib.util.find_spec("torch"):
        import torch

        print(f"cuda: {torch.cuda.is_available()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu only'})")
    for name in SECRETS:
        print(f"{name}: {'set' if os.getenv(name) else 'missing'}")
    print(f"data root: {resolve_data_root()}")
    return 0
```

- [ ] **Step 11: Install the package and run the tests**

```powershell
uv pip install --python .venv\Scripts\python.exe -e ".[ml,viewer,dev]"
.venv\Scripts\python -m pytest -q
```
Expected: `9 passed`.

- [ ] **Step 12: Smoke-check the CLI**

Run: `.venv\Scripts\python -m horizon doctor`
Expected: `ffmpeg:` points into `imageio_ffmpeg\binaries`, `torch: installed`, and `cuda: True (NVIDIA GeForce RTX 3060 Ti)`.

- [ ] **Step 13: Commit**

```powershell
Set-Location ..
git add .gitignore pipeline
git commit -m "feat(pipeline): scaffold horizon package, CLI discovery and doctor command"
```

### Task 2: Court model and pinhole camera math

**Files:**
- Create: `pipeline/horizon/court.py`, `pipeline/horizon/camera.py`
- Test: `pipeline/tests/test_court.py`, `pipeline/tests/test_camera.py`

**Interfaces:**
- Produces (`horizon.court`):
  - constants `HALF_LENGTH=11.885`, `HALF_DOUBLES=5.485`, `HALF_SINGLES=4.115`, `SERVICE_LINE_FROM_NET=6.40`, `NET_HEIGHT_POSTS=1.07`, `NET_POST_X`
  - `KEYPOINTS: dict[str, tuple[float, float]]`, 14 names in click order
  - `keypoint_world(name) -> np.ndarray(3)`
  - `court_lines() -> list[((x0, y0), (x1, y1))]`
  - `net_quad() -> np.ndarray(4, 3)`
  - `court_rect(margin_x=0.0, margin_y=0.0) -> np.ndarray(4, 3)`
- Produces (`horizon.camera`):
  - `intrinsics(focal, width, height) -> K`
  - `vertical_fov_from_horizontal(hfov_deg, width, height) -> float` (degrees)
  - `PinholeCamera(K, R, t, width, height)`:
    - properties `fx fy cx cy center forward vertical_fov (radians) projection_matrix`
    - methods `world_to_camera(pts)`, `project(pts) -> (uv (N,2), z (N,))`, `backproject(u, v, z) -> (N,3)`, `ray_directions(u, v) -> (N,3)` (camera-z = 1), `ground_intersection(u, v) -> (points (N,3), depth (N,), inf = miss)`, `height_above_ground(foot_xy, v_top) -> float`
    - classmethod `look_at(eye, target, vertical_fov_deg, width, height, up=(0,0,1))`
  - `rotation_to_wxyz(R) -> (4,)`, `wxyz_to_rotation(q) -> (3,3)`
  - `camera_pose(cam) -> (wxyz, position)`: the pose of the camera node in world coordinates, for Viser
  - `camera_from_pose(wxyz, position, vertical_fov_deg, width, height) -> PinholeCamera`
  - `camera_to_dict(cam) -> dict`, `camera_from_dict(d) -> PinholeCamera`

- [ ] **Step 1: Write the failing tests**

`pipeline/tests/test_court.py`:
```python
import numpy as np
import pytest

from horizon.court import (
    HALF_DOUBLES,
    HALF_LENGTH,
    HALF_SINGLES,
    KEYPOINTS,
    court_lines,
    court_rect,
    keypoint_world,
    net_quad,
)


def test_itf_dimensions():
    assert HALF_LENGTH * 2 == pytest.approx(23.77)
    assert HALF_DOUBLES * 2 == pytest.approx(10.97)
    assert HALF_SINGLES * 2 == pytest.approx(8.23)


def test_keypoints_are_mirror_symmetric_and_on_the_ground():
    assert len(KEYPOINTS) == 14
    for name, (x, y) in KEYPOINTS.items():
        mirror = name.replace("left", "@").replace("right", "left").replace("@", "right")
        mx, my = KEYPOINTS[mirror]
        assert (mx, my) == pytest.approx((-x, y))
        assert keypoint_world(name)[2] == 0.0


def test_lines_net_and_rect():
    lines = court_lines()
    assert len(lines) == 9
    for (x0, y0), (x1, y1) in lines:
        for x, y in ((x0, y0), (x1, y1)):
            assert abs(x) <= HALF_DOUBLES + 1e-9 and abs(y) <= HALF_LENGTH + 1e-9
    net = net_quad()
    assert net.shape == (4, 3)
    assert np.allclose(net[:, 1], 0.0)
    assert net[:, 2].max() == pytest.approx(1.07)
    rect = court_rect(margin_x=1.0, margin_y=2.0)
    assert rect[2, 0] == pytest.approx(HALF_DOUBLES + 1.0)
    assert rect[2, 1] == pytest.approx(HALF_LENGTH + 2.0)
```

`pipeline/tests/test_camera.py`:
```python
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
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run (from `pipeline\`): `.venv\Scripts\python -m pytest tests/test_court.py tests/test_camera.py -q`
Expected: `ModuleNotFoundError: No module named 'horizon.court'`.

- [ ] **Step 3: Implement `pipeline/horizon/court.py`**

```python
"""ITF tennis court model in metres (see spec §5.1 for the world frame)."""

from __future__ import annotations

import numpy as np

COURT_LENGTH = 23.77
DOUBLES_WIDTH = 10.97
SINGLES_WIDTH = 8.23
SERVICE_LINE_FROM_NET = 6.40
NET_HEIGHT_POSTS = 1.07

HALF_LENGTH = COURT_LENGTH / 2
HALF_DOUBLES = DOUBLES_WIDTH / 2
HALF_SINGLES = SINGLES_WIDTH / 2
NET_POST_X = HALF_DOUBLES + 0.914

# Named ground keypoints, in the order the calibration tool asks for them.
# "left"/"right" are as seen from the broadcast camera behind the near baseline.
KEYPOINTS: dict[str, tuple[float, float]] = {
    "near_doubles_left": (-HALF_DOUBLES, -HALF_LENGTH),
    "near_doubles_right": (HALF_DOUBLES, -HALF_LENGTH),
    "far_doubles_right": (HALF_DOUBLES, HALF_LENGTH),
    "far_doubles_left": (-HALF_DOUBLES, HALF_LENGTH),
    "near_singles_left": (-HALF_SINGLES, -HALF_LENGTH),
    "near_singles_right": (HALF_SINGLES, -HALF_LENGTH),
    "far_singles_right": (HALF_SINGLES, HALF_LENGTH),
    "far_singles_left": (-HALF_SINGLES, HALF_LENGTH),
    "near_service_left": (-HALF_SINGLES, -SERVICE_LINE_FROM_NET),
    "near_service_center": (0.0, -SERVICE_LINE_FROM_NET),
    "near_service_right": (HALF_SINGLES, -SERVICE_LINE_FROM_NET),
    "far_service_right": (HALF_SINGLES, SERVICE_LINE_FROM_NET),
    "far_service_center": (0.0, SERVICE_LINE_FROM_NET),
    "far_service_left": (-HALF_SINGLES, SERVICE_LINE_FROM_NET),
}


def keypoint_world(name: str) -> np.ndarray:
    x, y = KEYPOINTS[name]
    return np.array([x, y, 0.0])


def court_lines() -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Painted line segments on the ground as ((x0, y0), (x1, y1))."""
    L, D, S, SL = HALF_LENGTH, HALF_DOUBLES, HALF_SINGLES, SERVICE_LINE_FROM_NET
    return [
        ((-D, -L), (D, -L)),  # near baseline
        ((-D, L), (D, L)),  # far baseline
        ((-D, -L), (-D, L)),  # left doubles sideline
        ((D, -L), (D, L)),  # right doubles sideline
        ((-S, -L), (-S, L)),  # left singles sideline
        ((S, -L), (S, L)),  # right singles sideline
        ((-S, -SL), (S, -SL)),  # near service line
        ((-S, SL), (S, SL)),  # far service line
        ((0.0, -SL), (0.0, SL)),  # centre service line
    ]


def net_quad() -> np.ndarray:
    """Net corners (4x3): bottom-left, bottom-right, top-right, top-left."""
    return np.array(
        [
            [-NET_POST_X, 0.0, 0.0],
            [NET_POST_X, 0.0, 0.0],
            [NET_POST_X, 0.0, NET_HEIGHT_POSTS],
            [-NET_POST_X, 0.0, NET_HEIGHT_POSTS],
        ]
    )


def court_rect(margin_x: float = 0.0, margin_y: float = 0.0) -> np.ndarray:
    """Ground rectangle around the doubles court (4x3), counter-clockwise from near-left."""
    x, y = HALF_DOUBLES + margin_x, HALF_LENGTH + margin_y
    return np.array([[-x, -y, 0.0], [x, -y, 0.0], [x, y, 0.0], [-x, y, 0.0]])
```

- [ ] **Step 4: Implement `pipeline/horizon/camera.py`**

```python
"""Pinhole camera in the OpenCV convention (+X right, +Y down, +Z forward) and pose helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def intrinsics(focal: float, width: int, height: int) -> np.ndarray:
    return np.array([[focal, 0.0, (width - 1) / 2.0], [0.0, focal, (height - 1) / 2.0], [0.0, 0.0, 1.0]])


def vertical_fov_from_horizontal(hfov_deg: float, width: int, height: int) -> float:
    """Vertical field of view in degrees that matches a horizontal FOV at this aspect ratio."""
    half = np.radians(hfov_deg) / 2.0
    return float(np.degrees(2.0 * np.arctan(np.tan(half) * height / width)))


@dataclass(frozen=True, eq=False)
class PinholeCamera:
    K: np.ndarray  # (3, 3) intrinsics
    R: np.ndarray  # (3, 3) world -> camera rotation
    t: np.ndarray  # (3,) world -> camera translation
    width: int
    height: int

    @property
    def fx(self) -> float:
        return float(self.K[0, 0])

    @property
    def fy(self) -> float:
        return float(self.K[1, 1])

    @property
    def cx(self) -> float:
        return float(self.K[0, 2])

    @property
    def cy(self) -> float:
        return float(self.K[1, 2])

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t

    @property
    def forward(self) -> np.ndarray:
        return self.R[2].copy()

    @property
    def vertical_fov(self) -> float:
        """Vertical field of view in radians."""
        return float(2.0 * np.arctan2(self.height / 2.0, self.fy))

    @property
    def projection_matrix(self) -> np.ndarray:
        return self.K @ np.hstack([self.R, self.t.reshape(3, 1)])

    def world_to_camera(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points, dtype=np.float64) @ self.R.T + self.t

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World points (N,3) -> pixel coordinates (N,2) and camera-space depth z (N,)."""
        pc = self.world_to_camera(points)
        z = pc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.fx * pc[:, 0] / z + self.cx
            v = self.fy * pc[:, 1] / z + self.cy
        return np.stack([u, v], axis=1), z

    def backproject(self, u: np.ndarray, v: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Pixels plus camera-space depth -> world points (N,3)."""
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        pc = np.stack([(u - self.cx) / self.fx * z, (v - self.cy) / self.fy * z, z], axis=1)
        return (pc - self.t) @ self.R

    def ray_directions(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """World-space ray directions scaled so that their camera-space z component is exactly 1."""
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        dc = np.stack([(u - self.cx) / self.fx, (v - self.cy) / self.fy, np.ones_like(u)], axis=1)
        return dc @ self.R

    def ground_intersection(self, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Intersect pixel rays with the plane z=0. Returns world points and camera depth (inf = miss)."""
        d = self.ray_directions(u, v)
        c = self.center
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = -c[2] / d[:, 2]
        valid = np.isfinite(depth) & (depth > 0)
        depth = np.where(valid, depth, np.inf)
        points = c[None, :] + d * np.where(valid, depth, 0.0)[:, None]
        return points, depth

    def height_above_ground(self, foot_xy: tuple[float, float], v_top: float) -> float:
        """Height of the point on the vertical line through foot_xy that projects onto image row v_top."""
        P = self.projection_matrix
        x, y = foot_xy
        a = P[1, 0] * x + P[1, 1] * y + P[1, 3]
        b = P[1, 2]
        c = P[2, 0] * x + P[2, 1] * y + P[2, 3]
        d = P[2, 2]
        return float((a - v_top * c) / (v_top * d - b))

    @classmethod
    def look_at(
        cls,
        eye,
        target,
        vertical_fov_deg: float,
        width: int,
        height: int,
        up=(0.0, 0.0, 1.0),
    ) -> "PinholeCamera":
        eye = np.asarray(eye, dtype=np.float64)
        forward = np.asarray(target, dtype=np.float64) - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.asarray(up, dtype=np.float64))
        if np.linalg.norm(right) < 1e-9:  # looking straight up or down
            right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        R = np.stack([right, down, forward])
        focal = (height / 2.0) / np.tan(np.radians(vertical_fov_deg) / 2.0)
        return cls(K=intrinsics(focal, width, height), R=R, t=-R @ eye, width=width, height=height)


def rotation_to_wxyz(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (w, x, y, z) with w >= 0."""
    m = np.asarray(R, dtype=np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def wxyz_to_rotation(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def camera_pose(cam: PinholeCamera) -> tuple[np.ndarray, np.ndarray]:
    """Scene-node pose (wxyz, position) of a camera, as used by Viser frustums and gizmos."""
    return rotation_to_wxyz(cam.R.T), cam.center


def camera_from_pose(wxyz, position, vertical_fov_deg: float, width: int, height: int) -> PinholeCamera:
    R = wxyz_to_rotation(wxyz).T
    position = np.asarray(position, dtype=np.float64)
    focal = (height / 2.0) / np.tan(np.radians(vertical_fov_deg) / 2.0)
    return PinholeCamera(K=intrinsics(focal, width, height), R=R, t=-R @ position, width=width, height=height)


def camera_to_dict(cam: PinholeCamera) -> dict:
    return {
        "K": cam.K.tolist(),
        "R": cam.R.tolist(),
        "t": cam.t.tolist(),
        "width": cam.width,
        "height": cam.height,
    }


def camera_from_dict(data: dict) -> PinholeCamera:
    return PinholeCamera(
        K=np.array(data["K"], dtype=np.float64),
        R=np.array(data["R"], dtype=np.float64),
        t=np.array(data["t"], dtype=np.float64),
        width=int(data["width"]),
        height=int(data["height"]),
    )
```

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `.venv\Scripts\python -m pytest -q`
Expected: `20 passed`.

- [ ] **Step 6: Commit**

```powershell
git add pipeline/horizon/court.py pipeline/horizon/camera.py pipeline/tests/test_court.py pipeline/tests/test_camera.py
git commit -m "feat(pipeline): add ITF court model and pinhole camera math"
```

### Task 3: Video I/O and `horizon init`

**Files:**
- Create: `pipeline/horizon/video.py`, `pipeline/horizon/commands/init.py`
- Test: `pipeline/tests/test_video.py`

**Interfaces:**
- Consumes: `MatchPaths`, `resolve_data_root` (Task 1)
- Produces (`horizon.video`):
  - `VideoInfo(width, height, fps, frame_count)` with `.duration`
  - `probe(path) -> VideoInfo`
  - `iter_frames(path, start=0, stop=None, step=1) -> Iterator[tuple[int, np.ndarray RGB uint8]]`
  - `read_frame(path, index) -> np.ndarray`
  - `H264Writer(path, width, height, fps, crf=20)`: context manager with `.write(rgb)`, `.frames_written` and `.close()`
  - `transcode_clip(src, dst, start_s=0.0, duration_s=None, max_height=720) -> VideoInfo`
  - `save_meta(path, info, **extra)`, `load_meta(path) -> VideoInfo`
- CLI: `horizon init <video> --match-id <id> [--start S] [--duration S] [--max-height 720]` writes `source.mp4` and `meta.json`

- [ ] **Step 1: Write the failing tests**

`pipeline/tests/test_video.py`:
```python
import numpy as np
import pytest

from horizon.cli import main
from horizon.video import H264Writer, iter_frames, load_meta, probe, read_frame, transcode_clip


def write_color_clip(path, colors, size=(64, 48), fps=10.0):
    w, h = size
    with H264Writer(path, w, h, fps) as writer:
        for color in colors:
            writer.write(np.full((h, w, 3), color, dtype=np.uint8))


def test_writer_roundtrip(tmp_path):
    path = tmp_path / "clip.mp4"
    write_color_clip(path, [(200, 30, 30), (30, 200, 30), (30, 30, 200)] * 4)
    info = probe(path)
    assert (info.width, info.height) == (64, 48)
    assert info.fps == pytest.approx(10.0)
    frames = list(iter_frames(path))
    assert [i for i, _ in frames] == list(range(12))
    assert np.abs(frames[1][1].reshape(-1, 3).mean(axis=0) - (30, 200, 30)).max() < 12
    assert np.abs(read_frame(path, 2).reshape(-1, 3).mean(axis=0) - (30, 30, 200)).max() < 12


def test_writer_rejects_odd_sizes_and_bad_frames(tmp_path):
    with pytest.raises(ValueError):
        H264Writer(tmp_path / "odd.mp4", 63, 48, 10.0)
    with H264Writer(tmp_path / "ok.mp4", 64, 48, 10.0) as writer:
        with pytest.raises(ValueError):
            writer.write(np.zeros((48, 64), dtype=np.uint8))
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))


def test_transcode_trims_and_downscales(tmp_path):
    src = tmp_path / "raw.mp4"
    write_color_clip(src, [(i * 10, 100, 100) for i in range(20)], size=(128, 96), fps=10.0)
    info = transcode_clip(src, tmp_path / "out.mp4", start_s=0.5, duration_s=1.0, max_height=48)
    assert (info.width, info.height, info.frame_count) == (64, 48, 10)
    first = read_frame(tmp_path / "out.mp4", 0)
    assert abs(float(first[..., 0].mean()) - 50) < 12  # source frame 5 had red = 50


def test_init_command_writes_source_and_meta(tmp_path, monkeypatch):
    monkeypatch.setenv("HORIZON_DATA_ROOT", str(tmp_path / "data"))
    src = tmp_path / "raw.mp4"
    write_color_clip(src, [(90, 90, 90)] * 6, size=(64, 48), fps=12.0)
    assert main(["init", str(src), "--match-id", "unit"]) == 0
    assert (tmp_path / "data" / "unit" / "source.mp4").is_file()
    meta = load_meta(tmp_path / "data" / "unit" / "meta.json")
    assert (meta.width, meta.height, meta.frame_count) == (64, 48, 6)
    assert meta.fps == pytest.approx(12.0)
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.venv\Scripts\python -m pytest tests/test_video.py -q`
Expected: `ModuleNotFoundError: No module named 'horizon.video'`.

- [ ] **Step 3: Implement `pipeline/horizon/video.py`**

```python
"""Video helpers: OpenCV decoding, browser-friendly H.264 encoding via imageio-ffmpeg."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps


def probe(path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        return VideoInfo(
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=fps if fps > 0 else 25.0,
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()


def iter_frames(path: Path, start: int = 0, stop: int | None = None, step: int = 1) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame index, RGB uint8 frame) for start <= index < stop, every `step` frames."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    try:
        if start:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        index = start
        while stop is None or index < stop:
            ok, bgr = cap.read()
            if not ok:
                break
            if (index - start) % step == 0:
                yield index, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            index += 1
    finally:
        cap.release()


def read_frame(path: Path, index: int) -> np.ndarray:
    for _, frame in iter_frames(path, start=index, stop=index + 1):
        return frame
    raise IndexError(f"Frame {index} not found in {path}")


class H264Writer:
    """Write RGB uint8 frames to an H.264 yuv420p MP4 that browsers can play."""

    def __init__(self, path: Path, width: int, height: int, fps: float, crf: int = 20):
        if width % 2 or height % 2:
            raise ValueError(f"H.264 needs even frame dimensions, got {width}x{height}")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.size = (width, height)
        self.frames_written = 0
        self._gen = imageio_ffmpeg.write_frames(
            str(self.path),
            self.size,
            fps=fps,
            codec="libx264",
            pix_fmt_in="rgb24",
            pix_fmt_out="yuv420p",
            quality=None,
            macro_block_size=2,
            output_params=["-crf", str(crf), "-preset", "medium", "-movflags", "+faststart"],
        )
        self._gen.send(None)

    def write(self, frame: np.ndarray) -> None:
        expected = (self.size[1], self.size[0], 3)
        if frame.shape != expected or frame.dtype != np.uint8:
            raise ValueError(f"Expected uint8 frame of shape {expected}, got {frame.dtype} {frame.shape}")
        self._gen.send(np.ascontiguousarray(frame))
        self.frames_written += 1

    def close(self) -> None:
        self._gen.close()

    def __enter__(self) -> "H264Writer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def transcode_clip(
    src: Path, dst: Path, start_s: float = 0.0, duration_s: float | None = None, max_height: int = 720
) -> VideoInfo:
    """Trim [start_s, start_s + duration_s), downscale to <= max_height and re-encode as H.264."""
    info = probe(src)
    start = int(round(start_s * info.fps))
    stop = None if duration_s is None else start + int(round(duration_s * info.fps))
    scale = min(1.0, max_height / info.height)
    width = int(round(info.width * scale / 2)) * 2
    height = int(round(info.height * scale / 2)) * 2
    with H264Writer(dst, width, height, info.fps) as writer:
        for _, frame in iter_frames(src, start=start, stop=stop):
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
        count = writer.frames_written
    if count == 0:
        raise ValueError(f"No frames decoded from {src} in the requested range")
    return VideoInfo(width=width, height=height, fps=info.fps, frame_count=count)


def save_meta(path: Path, info: VideoInfo, **extra) -> None:
    data = {"width": info.width, "height": info.height, "fps": info.fps, "frame_count": info.frame_count, **extra}
    Path(path).write_text(json.dumps(data, indent=2))


def load_meta(path: Path) -> VideoInfo:
    data = json.loads(Path(path).read_text())
    return VideoInfo(int(data["width"]), int(data["height"]), float(data["fps"]), int(data["frame_count"]))
```

- [ ] **Step 4: Implement `pipeline/horizon/commands/init.py`**

```python
"""`horizon init`: import a raw clip as data/<match-id>/source.mp4 + meta.json."""

from __future__ import annotations

import argparse
from pathlib import Path

from horizon.paths import MatchPaths
from horizon.video import save_meta, transcode_clip

PEGASUS_INLINE_LIMIT = 25 * 1024 * 1024


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("init", help="Import a raw broadcast tennis clip")
    p.add_argument("video", type=Path, help="Raw clip (single static broadcast shot)")
    p.add_argument("--match-id", required=True)
    p.add_argument("--start", type=float, default=0.0, help="Start time in seconds")
    p.add_argument("--duration", type=float, default=None, help="Length in seconds (default: to the end)")
    p.add_argument("--max-height", type=int, default=720)
    p.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    paths = MatchPaths.for_match(args.match_id).ensure()
    info = transcode_clip(
        args.video, paths.source_video, start_s=args.start, duration_s=args.duration, max_height=args.max_height
    )
    save_meta(
        paths.meta,
        info,
        source=str(Path(args.video).resolve()),
        start_s=args.start,
        duration_s=args.duration,
    )
    size = paths.source_video.stat().st_size
    print(
        f"Wrote {paths.source_video} ({info.width}x{info.height} @ {info.fps:.2f} fps, "
        f"{info.frame_count} frames, {size / 1e6:.1f} MB)"
    )
    if size > PEGASUS_INLINE_LIMIT:
        print("WARNING: source.mp4 is over 25 MB and Pegasus will reject it; use --duration or --max-height.")
    return 0
```

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `.venv\Scripts\python -m pytest -q`
Expected: `24 passed`.

- [ ] **Step 6: Commit**

```powershell
git add pipeline/horizon/video.py pipeline/horizon/commands/init.py pipeline/tests/test_video.py
git commit -m "feat(pipeline): add H.264 video I/O and horizon init"
```

### Task 4: Court calibration (`horizon calibrate`)

**Files:**
- Create: `pipeline/horizon/calibration.py`, `pipeline/horizon/commands/calibrate.py`
- Test: `pipeline/tests/test_calibration.py`

**Interfaces:**
- Consumes: `PinholeCamera`, `intrinsics`, `camera_to_dict`, `camera_from_dict` (Task 2), `KEYPOINTS`, `court_lines`, `net_quad` (Task 2), `read_frame`, `H264Writer` (Task 3), `MatchPaths` (Task 1)
- Produces (`horizon.calibration`):
  - `MAX_ACCEPTABLE_RMS_PX = 4.0`
  - `Calibration(camera: PinholeCamera, keypoints: dict[str, tuple[float, float]], rms_px: float)`
  - `solve_calibration(keypoints, width, height) -> Calibration` (raises `ValueError` if <4 points or no valid pose)
  - `save_calibration(cal, path)`, `load_calibration(path) -> Calibration`
  - `draw_court_overlay(frame_rgb, camera, keypoints=None) -> np.ndarray` (new image)
  - `draw_keypoint_legend(canvas_bgr, current_name) -> None` (in place)
- CLI: `horizon calibrate --match-id <id> [--frame 0] [--keypoints file.json]`. It is interactive unless `--keypoints` is given, writes `calibration.json` and `calibration_preview.jpg`, and returns exit code 2 when RMS > 4 px.

- [ ] **Step 1: Write the failing tests**

`pipeline/tests/test_calibration.py`:
```python
import json

import numpy as np
import pytest

from horizon.calibration import (
    MAX_ACCEPTABLE_RMS_PX,
    draw_court_overlay,
    draw_keypoint_legend,
    load_calibration,
    save_calibration,
    solve_calibration,
)
from horizon.camera import PinholeCamera
from horizon.cli import main
from horizon.court import KEYPOINTS
from horizon.video import H264Writer

W, H = 1280, 720
CORNERS = ["near_doubles_left", "near_doubles_right", "far_doubles_right", "far_doubles_left"]


def true_camera() -> PinholeCamera:
    return PinholeCamera.look_at(eye=(0.5, -24.0, 9.0), target=(0.0, -3.0, 0.0), vertical_fov_deg=40.0, width=W, height=H)


def clicks(cam, names=None, noise=0.0, seed=0) -> dict[str, tuple[float, float]]:
    names = names or list(KEYPOINTS)
    uv, z = cam.project(np.array([[*KEYPOINTS[n], 0.0] for n in names]))
    assert (z > 0).all() and (uv >= 0).all() and (uv[:, 0] < W).all() and (uv[:, 1] < H).all()
    if noise:
        uv = uv + np.random.default_rng(seed).normal(0.0, noise, uv.shape)
    return {n: (float(u), float(v)) for n, (u, v) in zip(names, uv)}


def test_recovers_camera_from_all_keypoints():
    cam = true_camera()
    cal = solve_calibration(clicks(cam), W, H)
    assert cal.camera.fy == pytest.approx(cam.fy, rel=5e-3)
    assert np.allclose(cal.camera.center, cam.center, atol=0.05)
    assert cal.rms_px < 0.05


def test_recovers_camera_from_four_corners():
    cam = true_camera()
    cal = solve_calibration(clicks(cam, CORNERS), W, H)
    assert cal.camera.fy == pytest.approx(cam.fy, rel=1e-2)
    assert np.allclose(cal.camera.center, cam.center, atol=0.3)


def test_tolerates_click_noise():
    cam = true_camera()
    cal = solve_calibration(clicks(cam, noise=0.7, seed=3), W, H)
    assert cal.camera.fy == pytest.approx(cam.fy, rel=0.05)
    assert np.allclose(cal.camera.center, cam.center, atol=1.5)
    assert cal.rms_px < MAX_ACCEPTABLE_RMS_PX


def test_rejects_fewer_than_four_points():
    with pytest.raises(ValueError):
        solve_calibration(clicks(true_camera(), CORNERS[:3]), W, H)


def test_save_load_and_overlay(tmp_path):
    cal = solve_calibration(clicks(true_camera()), W, H)
    save_calibration(cal, tmp_path / "calibration.json")
    again = load_calibration(tmp_path / "calibration.json")
    assert np.allclose(again.camera.R, cal.camera.R)
    assert again.rms_px == pytest.approx(cal.rms_px)
    assert set(again.keypoints) == set(KEYPOINTS)
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    overlay = draw_court_overlay(frame, cal.camera, cal.keypoints)
    assert overlay.shape == frame.shape and overlay.any()
    assert not frame.any()


def test_legend_marks_current_keypoint():
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    draw_keypoint_legend(canvas, "far_service_center")
    assert canvas[:, :1100].sum() == 0
    red = (canvas[..., 2] == 255) & (canvas[..., 1] == 0) & (canvas[..., 0] == 0)
    assert red.any()


def test_calibrate_command_from_keypoints_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HORIZON_DATA_ROOT", str(tmp_path))
    match = tmp_path / "unit"
    match.mkdir()
    with H264Writer(match / "source.mp4", W, H, 25.0) as writer:
        writer.write(np.zeros((H, W, 3), dtype=np.uint8))
    keypoints_file = tmp_path / "kp.json"
    keypoints_file.write_text(json.dumps({"keypoints": {n: list(v) for n, v in clicks(true_camera()).items()}}))
    assert main(["calibrate", "--match-id", "unit", "--keypoints", str(keypoints_file)]) == 0
    assert (match / "calibration.json").is_file()
    assert (match / "calibration_preview.jpg").is_file()
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.venv\Scripts\python -m pytest tests/test_calibration.py -q`
Expected: `ModuleNotFoundError: No module named 'horizon.calibration'`.

- [ ] **Step 3: Implement `pipeline/horizon/calibration.py`**

```python
"""Solve the static broadcast camera from clicked court keypoints.

For a candidate focal length, planar PnP (IPPE) gives up to two poses. We keep the one with the camera
above the ground and the lowest reprojection error, and search the focal length that minimises it.
The principal point is fixed at the image centre and lens distortion is ignored.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from horizon.camera import PinholeCamera, camera_from_dict, camera_to_dict, intrinsics
from horizon.court import KEYPOINTS, court_lines, net_quad

MAX_ACCEPTABLE_RMS_PX = 4.0


@dataclass(frozen=True)
class Calibration:
    camera: PinholeCamera
    keypoints: dict[str, tuple[float, float]]
    rms_px: float


def _best_pose(world: np.ndarray, image: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray, float] | None:
    try:
        count, rvecs, tvecs, _ = cv2.solvePnPGeneric(world, image, K, None, flags=cv2.SOLVEPNP_IPPE)
    except cv2.error:
        return None
    best = None
    for rvec, tvec in zip(rvecs[:count], tvecs[:count]):
        R, _ = cv2.Rodrigues(rvec)
        t = np.asarray(tvec, dtype=np.float64).reshape(3)
        if (-R.T @ t)[2] <= 0:  # camera below the court: mirror solution
            continue
        projected, _ = cv2.projectPoints(world, rvec, tvec, K, None)
        rms = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - image) ** 2, axis=1))))
        if best is None or rms < best[2]:
            best = (R, t, rms)
    return best


def _golden_section(fn: Callable[[float], float], lo: float, hi: float, iterations: int = 60) -> float:
    ratio = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - ratio * (b - a), a + ratio * (b - a)
    fc, fd = fn(c), fn(d)
    for _ in range(iterations):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - ratio * (b - a)
            fc = fn(c)
        else:
            a, c, fc = c, d, fd
            d = a + ratio * (b - a)
            fd = fn(d)
    return (a + b) / 2.0


def solve_calibration(keypoints: dict[str, tuple[float, float]], width: int, height: int) -> Calibration:
    names = [name for name in KEYPOINTS if name in keypoints]
    if len(names) < 4:
        raise ValueError(f"Need at least 4 court keypoints, got {len(names)}")
    world = np.array([[*KEYPOINTS[n], 0.0] for n in names], dtype=np.float64)
    image = np.array([keypoints[n] for n in names], dtype=np.float64)

    def rms_for(focal: float) -> float:
        best = _best_pose(world, image, intrinsics(focal, width, height))
        return np.inf if best is None else best[2]

    grid = np.geomspace(0.3 * width, 8.0 * width, 240)
    errors = np.array([rms_for(f) for f in grid])
    if not np.isfinite(errors).any():
        raise ValueError("No camera pose above the court explains these keypoints; check the click order")
    i = int(np.argmin(errors))
    focal = _golden_section(rms_for, grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)])
    best = _best_pose(world, image, intrinsics(focal, width, height))
    if best is None or best[2] > errors[i]:
        focal = float(grid[i])
        best = _best_pose(world, image, intrinsics(focal, width, height))
    R, t, rms = best
    camera = PinholeCamera(K=intrinsics(focal, width, height), R=R, t=t, width=width, height=height)
    clicked = {n: (float(keypoints[n][0]), float(keypoints[n][1])) for n in names}
    return Calibration(camera=camera, keypoints=clicked, rms_px=rms)


def save_calibration(calibration: Calibration, path: Path) -> None:
    data = {
        "camera": camera_to_dict(calibration.camera),
        "keypoints": {k: list(v) for k, v in calibration.keypoints.items()},
        "rms_px": calibration.rms_px,
    }
    Path(path).write_text(json.dumps(data, indent=2))


def load_calibration(path: Path) -> Calibration:
    data = json.loads(Path(path).read_text())
    return Calibration(
        camera=camera_from_dict(data["camera"]),
        keypoints={k: (float(v[0]), float(v[1])) for k, v in data["keypoints"].items()},
        rms_px=float(data["rms_px"]),
    )


def _pt(uv: np.ndarray) -> tuple[int, int]:
    return int(round(float(uv[0]))), int(round(float(uv[1])))


def draw_court_overlay(frame_rgb: np.ndarray, camera: PinholeCamera, keypoints: dict | None = None) -> np.ndarray:
    """Court lines (red), net (yellow) and clicked keypoints (green) drawn over a copy of the frame."""
    out = frame_rgb.copy()
    for (x0, y0), (x1, y1) in court_lines():
        uv, z = camera.project(np.array([[x0, y0, 0.0], [x1, y1, 0.0]]))
        if (z > 0).all():
            cv2.line(out, _pt(uv[0]), _pt(uv[1]), (255, 64, 64), 2, cv2.LINE_AA)
    uv, z = camera.project(net_quad())
    if (z > 0).all():
        cv2.polylines(out, [np.round(uv).astype(np.int32).reshape(-1, 1, 2)], True, (255, 220, 0), 2, cv2.LINE_AA)
    for u, v in (keypoints or {}).values():
        cv2.circle(out, (int(round(u)), int(round(v))), 5, (0, 255, 0), 2, cv2.LINE_AA)
    return out


def draw_keypoint_legend(canvas_bgr: np.ndarray, current: str | None, size: tuple[int, int] = (120, 220)) -> None:
    """Top-down court diagram in the top-right corner; the keypoint to click next is red."""
    lw, lh = size
    x0 = canvas_bgr.shape[1] - lw - 12
    y0 = 40
    cv2.rectangle(canvas_bgr, (x0 - 6, y0 - 6), (x0 + lw + 6, y0 + lh + 18), (0, 0, 0), -1)

    def to_px(x: float, y: float) -> tuple[int, int]:
        return int(round(x0 + (x + 6.5) / 13.0 * lw)), int(round(y0 + (13.0 - y) / 26.0 * lh))

    for (xa, ya), (xb, yb) in court_lines():
        cv2.line(canvas_bgr, to_px(xa, ya), to_px(xb, yb), (200, 200, 200), 1, cv2.LINE_AA)
    for name, (x, y) in KEYPOINTS.items():
        color, radius = ((0, 0, 255), 5) if name == current else ((160, 160, 160), 2)
        cv2.circle(canvas_bgr, to_px(x, y), radius, color, -1, cv2.LINE_AA)
    cv2.putText(canvas_bgr, "camera side", (x0 + 18, y0 + lh + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
```

- [ ] **Step 4: Implement `pipeline/horizon/commands/calibrate.py`**

```python
"""`horizon calibrate`: click court keypoints on a frame and solve the broadcast camera."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from horizon.calibration import (
    MAX_ACCEPTABLE_RMS_PX,
    draw_court_overlay,
    draw_keypoint_legend,
    save_calibration,
    solve_calibration,
)
from horizon.court import KEYPOINTS
from horizon.paths import MatchPaths
from horizon.video import read_frame

WINDOW = "horizon calibrate"


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("calibrate", help="Solve the broadcast camera from court keypoints")
    p.add_argument("--match-id", required=True)
    p.add_argument("--frame", type=int, default=0, help="Frame to click on")
    p.add_argument("--keypoints", type=Path, default=None, help="JSON {name: [u, v]} (or a calibration.json) instead of clicking")
    p.set_defaults(handler=run)


def _text(canvas: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.55) -> None:
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def collect_keypoints(frame_rgb: np.ndarray) -> dict[str, tuple[float, float]]:
    """Interactive OpenCV window: click each named keypoint (S skip, U undo, Enter solve, Esc quit)."""
    names = list(KEYPOINTS)
    clicks: dict[str, tuple[float, float]] = {}
    state = {"index": 0}

    def on_mouse(event, x, y, _flags, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and state["index"] < len(names):
            clicks[names[state["index"]]] = (float(x), float(y))
            state["index"] += 1

    base = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, on_mouse)
    try:
        while True:
            canvas = base.copy()
            for name, (x, y) in clicks.items():
                cv2.circle(canvas, (int(x), int(y)), 5, (0, 255, 0), 2, cv2.LINE_AA)
                _text(canvas, name, (int(x) + 6, int(y) - 6), 0.4)
            current = names[state["index"]] if state["index"] < len(names) else None
            draw_keypoint_legend(canvas, current)
            prompt = f"Click {current}" if current else "All points placed"
            _text(canvas, f"{prompt} | S skip  U undo  Enter solve ({len(clicks)}/4+)  Esc quit", (12, 24))
            cv2.imshow(WINDOW, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key in (10, 13) and len(clicks) >= 4:
                return clicks
            if key in (ord("s"), ord("S")) and state["index"] < len(names):
                state["index"] += 1
            elif key in (ord("u"), ord("U")) and state["index"] > 0:
                state["index"] -= 1
                clicks.pop(names[state["index"]], None)
            elif key == 27:
                raise SystemExit("Calibration cancelled")
    finally:
        cv2.destroyWindow(WINDOW)


def run(args: argparse.Namespace) -> int:
    paths = MatchPaths.for_match(args.match_id)
    frame = read_frame(paths.source_video, args.frame)
    height, width = frame.shape[:2]
    if args.keypoints:
        data = json.loads(Path(args.keypoints).read_text())
        keypoints = {k: (float(v[0]), float(v[1])) for k, v in data.get("keypoints", data).items()}
    else:
        keypoints = collect_keypoints(frame)
    calibration = solve_calibration(keypoints, width, height)
    save_calibration(calibration, paths.calibration)
    preview = draw_court_overlay(frame, calibration.camera, calibration.keypoints)
    cv2.imwrite(str(paths.calibration_preview), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
    center = np.round(calibration.camera.center, 2).tolist()
    print(
        f"Saved {paths.calibration}: {len(calibration.keypoints)} keypoints, RMS {calibration.rms_px:.2f} px, "
        f"focal {calibration.camera.fy:.0f} px, camera at {center} m"
    )
    print(f"Check the overlay image: {paths.calibration_preview}")
    if calibration.rms_px > MAX_ACCEPTABLE_RMS_PX:
        print("WARNING: reprojection error above 4 px; re-run and click more carefully.")
        return 2
    return 0
```

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `.venv\Scripts\python -m pytest -q`
Expected: `31 passed`.

- [ ] **Step 6: Commit**

```powershell
git add pipeline/horizon/calibration.py pipeline/horizon/commands/calibrate.py pipeline/tests/test_calibration.py
git commit -m "feat(pipeline): solve broadcast camera from clicked court keypoints"
```

### Task 5: Person tracking (`horizon track`)

**Files:**
- Create: `pipeline/horizon/tracking.py`, `pipeline/horizon/commands/track.py`
- Test: `pipeline/tests/test_tracking.py`

**Interfaces:**
- Consumes: `iter_frames`, `load_meta`, `VideoInfo` (Task 3), `MatchPaths` (Task 1)
- Produces (`horizon.tracking`):
  - `DEFAULT_MODEL = "yolo26s-seg.pt"`
  - `Detection(track_id: int, conf: float, bbox: (x1, y1, x2, y2), polygon: tuple[(x, y), ...] = ())` (frozen), with properties `foot` (bottom-centre) and `head` (top-centre), and methods `mask(width, height) -> bool[H, W]`, `to_json()`, `from_json(d)`
  - `Detections(width, height, fps, frames: list[list[Detection]])`, with methods `by_track(frame, track_id) -> Detection | None`, `save(path)`, and classmethod `load(path)`
  - `detections_from_arrays(xyxy, conf, ids, polygons) -> list[Detection]`
  - `run_tracking(video_path, info, model_name=DEFAULT_MODEL, imgsz=1280, conf=0.2) -> Detections`
- CLI: `horizon track --match-id <id> [--model yolo26s-seg.pt] [--imgsz 1280] [--conf 0.2]` writes `detections.json`

- [ ] **Step 1: Write the failing tests**

`pipeline/tests/test_tracking.py`:
```python
import numpy as np

from horizon.tracking import Detection, Detections, detections_from_arrays


def square(x0, y0, size):
    return np.array([[x0, y0], [x0 + size, y0], [x0 + size, y0 + size], [x0, y0 + size]], dtype=np.float32)


def test_detections_from_arrays_converts_and_drops_untracked():
    xyxy = np.array([[10, 20, 30, 60], [100, 50, 120, 90]], dtype=np.float32)
    conf = np.array([0.9, 0.4])
    ids = np.array([7.0, 3.0])
    dets = detections_from_arrays(xyxy, conf, ids, [square(10, 20, 20), square(100, 50, 20)])
    assert [d.track_id for d in dets] == [7, 3]
    assert dets[0].bbox == (10.0, 20.0, 30.0, 60.0)
    assert dets[0].foot == (20.0, 60.0)
    assert dets[0].head == (20.0, 20.0)
    assert len(dets[0].polygon) == 4
    assert detections_from_arrays(xyxy, conf, None, None) == []


def test_polygon_is_simplified_and_rasterised():
    t = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    circle = np.stack([50 + 20 * np.cos(t), 50 + 20 * np.sin(t)], axis=1)
    det = detections_from_arrays(np.array([[30, 30, 70, 70]]), np.array([0.8]), np.array([1]), [circle])[0]
    assert 8 <= len(det.polygon) < 100
    mask = det.mask(100, 100)
    assert mask.dtype == bool
    assert abs(int(mask.sum()) - int(np.pi * 20**2)) < 120


def test_mask_falls_back_to_bbox_without_polygon():
    mask = Detection(track_id=1, conf=0.9, bbox=(10.0, 10.0, 20.0, 30.0)).mask(40, 40)
    assert mask[10:31, 10:21].all()
    assert mask.sum() == 11 * 21


def test_detections_json_roundtrip(tmp_path):
    frames = [[Detection(1, 0.9, (1.0, 2.0, 3.0, 4.0), ((1.0, 2.0), (3.0, 2.0), (3.0, 4.0)))], []]
    Detections(width=64, height=48, fps=25.0, frames=frames).save(tmp_path / "d.json")
    again = Detections.load(tmp_path / "d.json")
    assert (again.width, again.height, again.fps, len(again.frames)) == (64, 48, 25.0, 2)
    assert again.frames[0][0] == frames[0][0]
    assert again.by_track(0, 1) == frames[0][0]
    assert again.by_track(1, 1) is None
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.venv\Scripts\python -m pytest tests/test_tracking.py -q`
Expected: `ModuleNotFoundError: No module named 'horizon.tracking'`.

- [ ] **Step 3: Implement `pipeline/horizon/tracking.py`**

```python
"""Person detection + tracking with Ultralytics YOLO segmentation and ByteTrack."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from horizon.video import VideoInfo, iter_frames

DEFAULT_MODEL = "yolo26s-seg.pt"


@dataclass(frozen=True)
class Detection:
    track_id: int
    conf: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    polygon: tuple[tuple[float, float], ...] = ()  # mask outline in pixels

    @property
    def foot(self) -> tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2.0, self.bbox[3])

    @property
    def head(self) -> tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2.0, self.bbox[1])

    def mask(self, width: int, height: int) -> np.ndarray:
        canvas = np.zeros((height, width), dtype=np.uint8)
        if len(self.polygon) >= 3:
            pts = np.round(np.array(self.polygon)).astype(np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(canvas, [pts], 1)
        else:
            x1, y1, x2, y2 = (int(round(v)) for v in self.bbox)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), 1, thickness=-1)
        return canvas.astype(bool)

    def to_json(self) -> dict:
        return {
            "id": self.track_id,
            "conf": round(self.conf, 4),
            "bbox": [round(v, 2) for v in self.bbox],
            "polygon": [[round(x, 1), round(y, 1)] for x, y in self.polygon],
        }

    @classmethod
    def from_json(cls, data: dict) -> "Detection":
        return cls(
            track_id=int(data["id"]),
            conf=float(data["conf"]),
            bbox=tuple(float(v) for v in data["bbox"]),
            polygon=tuple((float(x), float(y)) for x, y in data.get("polygon", [])),
        )


@dataclass
class Detections:
    width: int
    height: int
    fps: float
    frames: list[list[Detection]]

    def by_track(self, frame: int, track_id: int) -> Detection | None:
        for det in self.frames[frame]:
            if det.track_id == track_id:
                return det
        return None

    def save(self, path: Path) -> None:
        data = {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frames": [[d.to_json() for d in frame] for frame in self.frames],
        }
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path: Path) -> "Detections":
        data = json.loads(Path(path).read_text())
        frames = [[Detection.from_json(d) for d in frame] for frame in data["frames"]]
        return cls(int(data["width"]), int(data["height"]), float(data["fps"]), frames)


def _simplify(polygon: np.ndarray, epsilon: float = 1.0) -> np.ndarray:
    polygon = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    if len(polygon) < 3:
        return polygon
    return cv2.approxPolyDP(polygon.reshape(-1, 1, 2), epsilon, True).reshape(-1, 2)


def detections_from_arrays(xyxy, conf, ids, polygons) -> list[Detection]:
    """One frame of Ultralytics output (numpy) -> Detections; boxes without a track id are dropped."""
    if ids is None:
        return []
    out = []
    for i in range(len(xyxy)):
        poly = _simplify(polygons[i]) if polygons is not None and i < len(polygons) else np.zeros((0, 2))
        out.append(
            Detection(
                track_id=int(ids[i]),
                conf=float(conf[i]),
                bbox=tuple(float(v) for v in xyxy[i]),
                polygon=tuple((float(x), float(y)) for x, y in poly),
            )
        )
    return out


def run_tracking(
    video_path: Path, info: VideoInfo, model_name: str = DEFAULT_MODEL, imgsz: int = 1280, conf: float = 0.2
) -> Detections:
    from ultralytics import YOLO

    model = YOLO(model_name)
    frames: list[list[Detection]] = []
    for index, rgb in iter_frames(video_path):
        result = model.track(
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            imgsz=imgsz,
            conf=conf,
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None or boxes.id is None or len(boxes) == 0:
            frames.append([])
        else:
            polygons = result.masks.xy if result.masks is not None else None
            frames.append(
                detections_from_arrays(
                    boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.id.cpu().numpy(), polygons
                )
            )
        if index % 50 == 0:
            print(f"  tracked frame {index}/{info.frame_count}")
    if len(frames) != info.frame_count:
        print(f"WARNING: decoded {len(frames)} frames but meta.json says {info.frame_count}")
    return Detections(info.width, info.height, info.fps, frames)
```

- [ ] **Step 4: Implement `pipeline/horizon/commands/track.py`**

```python
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
```

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `.venv\Scripts\python -m pytest -q`
Expected: `35 passed`.

- [ ] **Step 6: Commit**

```powershell
git add pipeline/horizon/tracking.py pipeline/horizon/commands/track.py pipeline/tests/test_tracking.py
git commit -m "feat(pipeline): track people with YOLO-seg + ByteTrack"
```

### Task 6: TwelveLabs Pegasus and Gemini clients

**Files:**
- Create: `pipeline/horizon/llm/__init__.py`, `pipeline/horizon/llm/jsonutil.py`, `pipeline/horizon/llm/pegasus.py`, `pipeline/horizon/llm/gemini.py`
- Test: `pipeline/tests/test_llm_clients.py`

**Interfaces:**
- Produces (`horizon.llm.jsonutil`):
  - `loads_lenient(text) -> Any`: strips code fences; falls back to the outermost `{…}`
- Produces (`horizon.llm.pegasus`):
  - `MAX_INLINE_BYTES`, `DEFAULT_MODEL_ID = "us.twelvelabs.pegasus-1-2-v1:0"`, `CLIP_ANALYSIS_PROMPT`, `CLIP_ANALYSIS_SCHEMA`
  - `PlayerDescription(name, appearance)`
  - `ClipAnalysis(near, far, score, summary)`, with `to_json()` / `from_json()`
  - `build_request(video_bytes, prompt=CLIP_ANALYSIS_PROMPT) -> dict`
  - `parse_response(body: dict) -> ClipAnalysis`
  - `analyze_clip(video_path, region=None, model_id=None, client=None) -> ClipAnalysis`
- Produces (`horizon.llm.gemini`):
  - `OPENROUTER_URL`, `DEFAULT_MODEL = "google/gemini-3.5-flash"`, `BOXES_SCHEMA`
  - `build_prompt(near_appearance, far_appearance) -> str`
  - `build_request(image_jpeg, near_appearance, far_appearance, model) -> dict`
  - `box_2d_to_xyxy(box, width, height) -> (x1, y1, x2, y2)`
  - `parse_response(body, width, height) -> {"near": xyxy, "far": xyxy}`
  - `locate_players(frame_rgb, near_appearance, far_appearance, model=None, session=None, api_key=None) -> {"near": xyxy, "far": xyxy}`

- [ ] **Step 1: Write the failing tests**

`pipeline/tests/test_llm_clients.py`:
```python
import io
import json

import numpy as np
import pytest

from horizon.llm import gemini, pegasus


class FakeBedrock:
    def __init__(self, body):
        self.body = body
        self.calls = []

    def invoke_model(self, **kwargs):
        self.calls.append(kwargs)
        return {"body": io.BytesIO(json.dumps(self.body).encode())}


class FakeResponse:
    def __init__(self, status, body):
        self.status_code = status
        self._body = body
        self.text = json.dumps(body)

    def json(self):
        return self._body


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


def pegasus_message(near="Carlos Alcaraz", far="Unknown"):
    data = {
        "near_player": {"name": near, "appearance": "white shirt, black shorts"},
        "far_player": {"name": far, "appearance": "navy shirt, white cap"},
        "score": "6-4 2-1 30-15",
        "summary": "Long baseline rally won by the near player.",
    }
    return {"message": json.dumps(data), "finishReason": "stop"}


def openrouter_body(content):
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def test_pegasus_request_uses_structured_output():
    request = pegasus.build_request(b"fake-video")
    assert request["mediaSource"]["base64String"] == "ZmFrZS12aWRlbw=="
    assert request["responseFormat"]["jsonSchema"]["required"] == ["near_player", "far_player", "score", "summary"]
    assert request["temperature"] == 0


def test_pegasus_rejects_clips_over_the_inline_limit():
    with pytest.raises(ValueError, match="25 MB"):
        pegasus.build_request(b"x" * (pegasus.MAX_INLINE_BYTES + 1))


def test_pegasus_parse_and_defaults():
    analysis = pegasus.parse_response(pegasus_message(far=""))
    assert analysis.near.name == "Carlos Alcaraz"
    assert analysis.far.name == "Unknown"
    assert analysis.far.appearance == "navy shirt, white cap"
    assert analysis.score == "6-4 2-1 30-15"
    with pytest.raises(ValueError):
        pegasus.parse_response({"finishReason": "stop"})


def test_analyze_clip_calls_bedrock(tmp_path):
    clip = tmp_path / "source.mp4"
    clip.write_bytes(b"video-bytes")
    fake = FakeBedrock(pegasus_message())
    analysis = pegasus.analyze_clip(clip, model_id="us.twelvelabs.pegasus-1-2-v1:0", client=fake)
    assert fake.calls[0]["modelId"] == "us.twelvelabs.pegasus-1-2-v1:0"
    assert json.loads(fake.calls[0]["body"])["inputPrompt"] == pegasus.CLIP_ANALYSIS_PROMPT
    assert analysis.to_json()["near"]["name"] == "Carlos Alcaraz"
    assert pegasus.ClipAnalysis.from_json(analysis.to_json()) == analysis


def test_box_2d_conversion_and_validation():
    assert gemini.box_2d_to_xyxy([100, 200, 500, 400], 1000, 500) == (200.0, 50.0, 400.0, 250.0)
    assert gemini.box_2d_to_xyxy([-5, 0, 1200, 1000], 100, 100) == (0.0, 0.0, 100.0, 100.0)
    with pytest.raises(ValueError):
        gemini.box_2d_to_xyxy([1, 2, 3], 10, 10)
    with pytest.raises(ValueError):
        gemini.box_2d_to_xyxy([500, 500, 400, 600], 10, 10)


def test_gemini_parse_handles_code_fences():
    data = {"near_player": {"box_2d": [600, 100, 900, 200]}, "far_player": {"box_2d": [200, 500, 300, 550]}}
    boxes = gemini.parse_response(openrouter_body("```json\n" + json.dumps(data) + "\n```"), 1000, 1000)
    assert boxes == {"near": (100.0, 600.0, 200.0, 900.0), "far": (500.0, 200.0, 550.0, 300.0)}


def test_locate_players_sends_image_and_schema():
    data = {"near_player": {"box_2d": [500, 100, 900, 200]}, "far_player": {"box_2d": [100, 600, 250, 650]}}
    session = FakeSession(FakeResponse(200, openrouter_body(json.dumps(data))))
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    boxes = gemini.locate_players(frame, "white shirt", "navy shirt", model="google/gemini-3.5-flash", session=session, api_key="k")
    url, kwargs = session.calls[0]
    assert url == gemini.OPENROUTER_URL
    assert kwargs["headers"]["Authorization"] == "Bearer k"
    payload = kwargs["json"]
    assert payload["model"] == "google/gemini-3.5-flash"
    assert payload["response_format"]["type"] == "json_schema"
    text_part, image_part = payload["messages"][0]["content"]
    assert "white shirt" in text_part["text"] and "navy shirt" in text_part["text"]
    assert image_part["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert boxes["near"] == (20.0, 50.0, 40.0, 90.0)


def test_locate_players_raises_on_http_error():
    session = FakeSession(FakeResponse(429, {"error": {"message": "rate limited"}}))
    with pytest.raises(RuntimeError, match="429"):
        gemini.locate_players(np.zeros((10, 10, 3), np.uint8), "", "", session=session, api_key="k")
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.venv\Scripts\python -m pytest tests/test_llm_clients.py -q`
Expected: `ModuleNotFoundError: No module named 'horizon.llm'`.

- [ ] **Step 3: Implement the shared JSON helper**

`pipeline/horizon/llm/__init__.py`:
```python
"""Clients for the AI services: TwelveLabs Pegasus, Gemini (OpenRouter) and Backboard."""
```

`pipeline/horizon/llm/jsonutil.py`:
```python
"""Lenient JSON parsing for LLM output (code fences, stray prose around the object)."""

from __future__ import annotations

import json
from typing import Any


def loads_lenient(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])
```

- [ ] **Step 4: Implement `pipeline/horizon/llm/pegasus.py`**

```python
"""TwelveLabs Pegasus 1.2 video understanding via Amazon Bedrock, using structured JSON output."""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from horizon.llm.jsonutil import loads_lenient

MAX_INLINE_BYTES = 25 * 1024 * 1024  # Bedrock limit for mediaSource.base64String
DEFAULT_MODEL_ID = "us.twelvelabs.pegasus-1-2-v1:0"

_PLAYER_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "appearance": {"type": "string"}},
    "required": ["name", "appearance"],
}

CLIP_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "near_player": _PLAYER_SCHEMA,
        "far_player": _PLAYER_SCHEMA,
        "score": {"type": "string"},
        "summary": {"type": "string"},
    },
    "required": ["near_player", "far_player", "score", "summary"],
}

CLIP_ANALYSIS_PROMPT = (
    "This is a broadcast tennis clip filmed from a fixed camera behind one baseline. "
    "The NEAR player is on the half of the court closest to the camera (lower part of the frame); "
    "the FAR player is on the other side of the net (upper part of the frame). "
    "For each player give their name if it appears on screen or is spoken by the commentators, otherwise 'Unknown', "
    "and describe their clothing (shirt, shorts or skirt, headwear, shoes) and hair precisely enough to pick them out "
    "in a single still frame. Ignore ball kids, line judges, the chair umpire and spectators. "
    "Report the score exactly as shown on any on-screen scoreboard (empty string if none) and summarise the point "
    "in at most two sentences."
)


@dataclass(frozen=True)
class PlayerDescription:
    name: str
    appearance: str


@dataclass(frozen=True)
class ClipAnalysis:
    near: PlayerDescription
    far: PlayerDescription
    score: str
    summary: str

    def to_json(self) -> dict:
        return {
            "near": {"name": self.near.name, "appearance": self.near.appearance},
            "far": {"name": self.far.name, "appearance": self.far.appearance},
            "score": self.score,
            "summary": self.summary,
        }

    @classmethod
    def from_json(cls, data: dict) -> "ClipAnalysis":
        return cls(
            near=PlayerDescription(**data["near"]),
            far=PlayerDescription(**data["far"]),
            score=data["score"],
            summary=data["summary"],
        )


def build_request(video_bytes: bytes, prompt: str = CLIP_ANALYSIS_PROMPT) -> dict:
    if len(video_bytes) > MAX_INLINE_BYTES:
        raise ValueError(
            f"Clip is {len(video_bytes) / 1e6:.1f} MB; Pegasus accepts at most 25 MB inline. "
            "Re-run `horizon init` with a shorter --duration or smaller --max-height."
        )
    return {
        "inputPrompt": prompt,
        "mediaSource": {"base64String": base64.b64encode(video_bytes).decode("ascii")},
        "temperature": 0,
        "maxOutputTokens": 1024,
        "responseFormat": {"jsonSchema": CLIP_ANALYSIS_SCHEMA},
    }


def parse_response(body: dict) -> ClipAnalysis:
    message = body.get("message")
    if not isinstance(message, str):
        raise ValueError(f"Unexpected Pegasus response: {json.dumps(body)[:300]}")
    data = loads_lenient(message)

    def player(key: str) -> PlayerDescription:
        raw = data.get(key) or {}
        name = str(raw.get("name") or "").strip() or "Unknown"
        return PlayerDescription(name=name, appearance=str(raw.get("appearance") or "").strip())

    return ClipAnalysis(
        near=player("near_player"),
        far=player("far_player"),
        score=str(data.get("score") or "").strip(),
        summary=str(data.get("summary") or "").strip(),
    )


def analyze_clip(video_path: Path, region: str | None = None, model_id: str | None = None, client: Any = None) -> ClipAnalysis:
    model_id = model_id or os.getenv("TWELVELABS_MODEL_ID") or DEFAULT_MODEL_ID
    if client is None:
        import boto3

        client = boto3.client("bedrock-runtime", region_name=region or os.getenv("AWS_REGION") or "us-east-1")
    request = build_request(Path(video_path).read_bytes())
    response = client.invoke_model(
        modelId=model_id, body=json.dumps(request), contentType="application/json", accept="application/json"
    )
    return parse_response(json.loads(response["body"].read()))
```

- [ ] **Step 5: Implement `pipeline/horizon/llm/gemini.py`**

```python
"""Gemini (through OpenRouter) finds both players in one frame as normalised box_2d boxes.

Gemini's native box format is [ymin, xmin, ymax, xmax] on a 0-1000 grid. The soccer prototype asked for pixels
and got 0-1000 values back (y=721 in a 480-px-high frame), so we ask for the native format and convert.
"""

from __future__ import annotations

import base64
import os

import cv2
import numpy as np
import requests

from horizon.llm.jsonutil import loads_lenient

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-3.5-flash"

_BOX = {
    "type": "object",
    "properties": {"box_2d": {"type": "array", "items": {"type": "number"}}},
    "required": ["box_2d"],
    "additionalProperties": False,
}
BOXES_SCHEMA = {
    "type": "object",
    "properties": {"near_player": _BOX, "far_player": _BOX},
    "required": ["near_player", "far_player"],
    "additionalProperties": False,
}


def build_prompt(near_appearance: str, far_appearance: str) -> str:
    return (
        "Detect the two tennis players in this broadcast frame.\n"
        f"NEAR player (on the half of the court closest to the camera): {near_appearance or 'no description'}.\n"
        f"FAR player (on the other side of the net): {far_appearance or 'no description'}.\n"
        "Ignore ball kids, line judges, the chair umpire and spectators. "
        "Return one tight bounding box per player as box_2d = [ymin, xmin, ymax, xmax] normalised to 0-1000."
    )


def build_request(image_jpeg: bytes, near_appearance: str, far_appearance: str, model: str) -> dict:
    data_url = "data:image/jpeg;base64," + base64.b64encode(image_jpeg).decode("ascii")
    return {
        "model": model,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_prompt(near_appearance, far_appearance)},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "player_boxes", "strict": True, "schema": BOXES_SCHEMA},
        },
    }


def box_2d_to_xyxy(box, width: int, height: int) -> tuple[float, float, float, float]:
    if len(box) != 4:
        raise ValueError(f"box_2d must have 4 numbers, got {box!r}")
    ymin, xmin, ymax, xmax = (min(max(float(v), 0.0), 1000.0) for v in box)
    if xmax <= xmin or ymax <= ymin:
        raise ValueError(f"Degenerate box_2d {box!r}")
    return (xmin / 1000.0 * width, ymin / 1000.0 * height, xmax / 1000.0 * width, ymax / 1000.0 * height)


def parse_response(body: dict, width: int, height: int) -> dict[str, tuple[float, float, float, float]]:
    content = body["choices"][0]["message"]["content"]
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    data = loads_lenient(content)
    return {
        "near": box_2d_to_xyxy(data["near_player"]["box_2d"], width, height),
        "far": box_2d_to_xyxy(data["far_player"]["box_2d"], width, height),
    }


def locate_players(
    frame_rgb: np.ndarray,
    near_appearance: str,
    far_appearance: str,
    model: str | None = None,
    session=None,
    api_key: str | None = None,
) -> dict[str, tuple[float, float, float, float]]:
    api_key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set (add it to pipeline/.env)")
    model = model or os.getenv("GEMINI_MODEL_ID") or DEFAULT_MODEL
    ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    session = session or requests.Session()
    response = session.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=build_request(jpeg.tobytes(), near_appearance, far_appearance, model),
        timeout=120,
    )
    if response.status_code != 200:
        raise RuntimeError(f"OpenRouter request failed ({response.status_code}): {response.text[:500]}")
    height, width = frame_rgb.shape[:2]
    return parse_response(response.json(), width, height)
```

- [ ] **Step 6: Run the tests and confirm they pass**

Run: `.venv\Scripts\python -m pytest -q`
Expected: `43 passed`.

- [ ] **Step 7: Commit**

```powershell
git add pipeline/horizon/llm pipeline/tests/test_llm_clients.py
git commit -m "feat(pipeline): add Pegasus (Bedrock) and Gemini (OpenRouter) clients"
```

### Task 7: Backboard orchestration and player identification (`horizon identify`)

**Files:**
- Create: `pipeline/horizon/llm/backboard.py`, `pipeline/horizon/identify.py`, `pipeline/horizon/commands/identify.py`, `pipeline/tests/synthetic.py`
- Test: `pipeline/tests/test_identify.py`

**Interfaces:**
- Consumes:
  - `pegasus.analyze_clip`, `ClipAnalysis`, `PlayerDescription`, `gemini.locate_players` (Task 6)
  - `Detections`, `Detection` (Task 5)
  - `load_calibration`, `save_calibration`, `Calibration` (Task 4)
  - `read_frame` (Task 3)
  - `PinholeCamera.ground_intersection`, `.height_above_ground` (Task 2)
- Produces (`horizon.llm.backboard`):
  - `BackboardClient(api_key=None, session=None, base_url=BASE_URL, timeout=180)`, with methods `create_assistant(name, system_prompt, tools) -> str`, `create_thread(assistant_id) -> str`, `send_message(thread_id, content, *, llm_provider=None, model_name=None, memory="Auto") -> dict`, and `submit_tool_outputs(thread_id, run_id, tool_outputs) -> dict`
  - `ToolExecution(name, arguments, output=None, error=None)`, `ToolLoopResult(content, executions)`
  - `run_tool_loop(client, thread_id, content, handlers, *, llm_provider=None, model_name=None, max_rounds=6) -> ToolLoopResult`
- Produces (`horizon.identify`):
  - `ROLES = ("near", "far")`, `SYSTEM_PROMPT`, `TOOLS`
  - `court_xy(camera, pixel) -> (x, y) | None`
  - `on_half(role, xy) -> bool`
  - `iou(a, b) -> float`
  - `match_box(box, candidates, width, height) -> Detection | None`
  - `heuristic_tracks(detections, camera) -> {"near": id|None, "far": id|None}`
  - `PlayerIdentity(role, name, description, track_id, source, box)`
  - `Identity(players, score, summary, frame_index=0, warnings=[])`, with `save(path)`, `load(path)`, `to_json()`, `from_json()`
  - `resolve_identity(analysis, boxes, detections, camera, frame_index=0) -> Identity`
  - `get_or_create_assistant(client, cache_path) -> str`
  - `identify_direct(video_path, frame_rgb) -> (ClipAnalysis, boxes)`
  - `identify_via_backboard(client, match_id, video_path, frame_rgb, cache_path, llm_provider=None, model_name=None) -> (ClipAnalysis, boxes, notes)`
- Produces (`tests/synthetic.py`, shared by later test modules): `W=1280`, `H=720`, `broadcast_camera()`, `person_detection(camera, track_id, x, y, stature=1.85, conf=0.9) -> Detection`
- CLI: `horizon identify --match-id <id> [--orchestrator backboard|direct|none] [--frame 0]` writes `identity.json`. It exits 1 if a role has no track.

- [ ] **Step 1: Write the shared synthetic fixtures**

`pipeline/tests/synthetic.py`:
```python
"""Shared synthetic fixtures: a broadcast camera behind the near baseline and people standing on the court."""

import numpy as np

from horizon.camera import PinholeCamera
from horizon.tracking import Detection

W, H = 1280, 720


def broadcast_camera() -> PinholeCamera:
    return PinholeCamera.look_at(eye=(0.5, -24.0, 9.0), target=(0.0, -3.0, 0.0), vertical_fov_deg=40.0, width=W, height=H)


def person_detection(camera: PinholeCamera, track_id: int, x: float, y: float, stature: float = 1.85, conf: float = 0.9) -> Detection:
    """Box whose bottom-centre is the exact projection of the feet and whose top row is the projected head."""
    uv, _ = camera.project(np.array([[x, y, 0.0], [x, y, stature]]))
    height_px = uv[0, 1] - uv[1, 1]
    half_w = 0.2 * height_px
    cx = uv[0, 0]
    return Detection(track_id=track_id, conf=conf, bbox=(cx - half_w, uv[1, 1], cx + half_w, uv[0, 1]))
```

- [ ] **Step 2: Write the failing tests**

`pipeline/tests/test_identify.py`:
```python
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
```

- [ ] **Step 3: Run the tests and confirm they fail**

Run: `.venv\Scripts\python -m pytest tests/test_identify.py -q`
Expected: `ImportError: cannot import name 'identify' from 'horizon'`.

- [ ] **Step 4: Implement `pipeline/horizon/llm/backboard.py`**

```python
"""Backboard.io REST client and the tool-calling loop that orchestrates Pegasus + Gemini."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field

import requests

BASE_URL = "https://app.backboard.io/api"


class BackboardClient:
    def __init__(self, api_key: str | None = None, session=None, base_url: str = BASE_URL, timeout: float = 180):
        self.api_key = api_key or os.getenv("BACKBOARD_API_KEY")
        if not self.api_key:
            raise RuntimeError("BACKBOARD_API_KEY is not set (add it to pipeline/.env or use --orchestrator direct)")
        self.session = session or requests.Session()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, payload: dict) -> dict:
        response = self.session.post(
            f"{self.base_url}{path}",
            headers={"X-API-Key": self.api_key, "Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout,
        )
        if response.status_code not in (200, 201):
            raise RuntimeError(f"Backboard {path} failed ({response.status_code}): {response.text[:500]}")
        return response.json()

    def create_assistant(self, name: str, system_prompt: str, tools: list[dict]) -> str:
        data = self._post("/assistants", {"name": name, "system_prompt": system_prompt, "tools": tools})
        return str(data.get("assistant_id") or data.get("id"))

    def create_thread(self, assistant_id: str) -> str:
        return str(self._post(f"/assistants/{assistant_id}/threads", {})["thread_id"])

    def send_message(
        self, thread_id: str, content: str, *, llm_provider: str | None = None, model_name: str | None = None, memory: str = "Auto"
    ) -> dict:
        payload: dict = {"content": content, "stream": False, "memory": memory}
        if llm_provider:
            payload["llm_provider"] = llm_provider
        if model_name:
            payload["model_name"] = model_name
        return self._post(f"/threads/{thread_id}/messages", payload)

    def submit_tool_outputs(self, thread_id: str, run_id: str, tool_outputs: list[dict]) -> dict:
        return self._post(f"/threads/{thread_id}/runs/{run_id}/submit-tool-outputs", {"tool_outputs": tool_outputs})


@dataclass
class ToolExecution:
    name: str
    arguments: dict
    output: dict | None = None
    error: str | None = None


@dataclass
class ToolLoopResult:
    content: str
    executions: list[ToolExecution] = field(default_factory=list)


def _parse_arguments(function: dict) -> dict:
    raw = function.get("parsed_arguments", function.get("arguments"))
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def run_tool_loop(
    client,
    thread_id: str,
    content: str,
    handlers: dict[str, Callable[[dict], dict]],
    *,
    llm_provider: str | None = None,
    model_name: str | None = None,
    max_rounds: int = 6,
) -> ToolLoopResult:
    """Send `content`, execute every requested tool locally, submit outputs, repeat until no tool calls remain."""
    response = client.send_message(thread_id, content, llm_provider=llm_provider, model_name=model_name)
    executions: list[ToolExecution] = []
    for _ in range(max_rounds):
        calls = response.get("tool_calls") or []
        if not calls:
            break
        run_id = response.get("run_id")
        if not run_id:
            raise RuntimeError("Backboard requested tool calls without a run_id")
        outputs = []
        for call in calls:
            function = call.get("function") or {}
            execution = ToolExecution(name=function.get("name", ""), arguments=_parse_arguments(function))
            handler = handlers.get(execution.name)
            if handler is None:
                execution.error = f"unknown tool {execution.name!r}"
            else:
                try:
                    execution.output = handler(execution.arguments)
                except Exception as exc:  # the failure is reported back to the assistant
                    execution.error = f"{type(exc).__name__}: {exc}"
            executions.append(execution)
            payload = execution.output if execution.error is None else {"error": execution.error}
            outputs.append({"tool_call_id": call.get("id"), "output": json.dumps(payload)})
        response = client.submit_tool_outputs(thread_id, run_id, outputs)
    return ToolLoopResult(content=str(response.get("content") or ""), executions=executions)
```

- [ ] **Step 5: Implement `pipeline/horizon/identify.py`**

```python
"""Decide which tracked person is the near player and which is the far player.

The AI services provide names, clothing and a frame box per player. Court geometry validates their answer,
and a persistence heuristic takes over whenever the answer is missing or inconsistent.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from horizon.camera import PinholeCamera
from horizon.court import HALF_DOUBLES
from horizon.llm import gemini, pegasus
from horizon.llm.backboard import run_tool_loop
from horizon.llm.pegasus import ClipAnalysis, PlayerDescription
from horizon.tracking import Detection, Detections

ROLES = ("near", "far")
HALF_Y_RANGE = {"near": (-18.0, -0.5), "far": (0.5, 18.0)}
MAX_ABS_X = 8.0

SYSTEM_PROMPT = (
    "You are Horizon Tennis Analyst. You identify the two players in broadcast tennis clips by calling tools. "
    "Always call analyze_tennis_clip first, then call locate_players with the near and far appearance strings it "
    "returned. Remember players you have seen before (names and outfits) to help with future clips. "
    "When both tools have run, reply with the single word: done."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "analyze_tennis_clip",
            "description": "Run TwelveLabs Pegasus on the tennis clip. Returns both players' names and clothing, "
            "the scoreboard text and a short summary of the point.",
            "parameters": {
                "type": "object",
                "properties": {"match_id": {"type": "string", "description": "Match id of the clip"}},
                "required": ["match_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "locate_players",
            "description": "Run Gemini on a frame of the clip to find the bounding boxes of the near and far players, "
            "given their clothing descriptions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "match_id": {"type": "string"},
                    "near_appearance": {"type": "string", "description": "Clothing of the near player"},
                    "far_appearance": {"type": "string", "description": "Clothing of the far player"},
                },
                "required": ["match_id", "near_appearance", "far_appearance"],
            },
        },
    },
]


def court_xy(camera: PinholeCamera, pixel: tuple[float, float]) -> tuple[float, float] | None:
    points, depth = camera.ground_intersection(np.array([pixel[0]]), np.array([pixel[1]]))
    if not np.isfinite(depth[0]):
        return None
    return float(points[0, 0]), float(points[0, 1])


def on_half(role: str, xy: tuple[float, float] | None) -> bool:
    if xy is None:
        return False
    lo, hi = HALF_Y_RANGE[role]
    return abs(xy[0]) <= MAX_ABS_X and lo <= xy[1] <= hi


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_box(box, candidates: list[Detection], width: int, height: int) -> Detection | None:
    """Detection overlapping `box` best (IoU >= 0.1), else the nearest centre within 15 % of the image diagonal."""
    if not candidates:
        return None
    best = max(candidates, key=lambda d: iou(box, d.bbox))
    if iou(box, best.bbox) >= 0.1:
        return best
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2

    def distance(d: Detection) -> float:
        return float(np.hypot((d.bbox[0] + d.bbox[2]) / 2 - cx, (d.bbox[1] + d.bbox[3]) / 2 - cy))

    nearest = min(candidates, key=distance)
    return nearest if distance(nearest) <= 0.15 * float(np.hypot(width, height)) else None


def heuristic_tracks(detections: Detections, camera: PinholeCamera) -> dict[str, int | None]:
    """Most persistent track per court half; tracks whose median position is outside the court width count half."""
    positions: dict[int, list[tuple[float, float]]] = {}
    statures: dict[int, list[float]] = {}
    for frame in detections.frames:
        for det in frame:
            xy = court_xy(camera, det.foot)
            if xy is None:
                continue
            positions.setdefault(det.track_id, []).append(xy)
            statures.setdefault(det.track_id, []).append(camera.height_above_ground(xy, det.bbox[1]))
    result: dict[str, int | None] = {}
    for role in ROLES:
        best_id, best_score = None, -1.0
        for track_id, xys in positions.items():
            arr = np.array(xys)
            median = (float(np.median(arr[:, 0])), float(np.median(arr[:, 1])))
            if not on_half(role, median):
                continue
            count = sum(on_half(role, xy) for xy in xys)
            weight = 1.0 if abs(median[0]) <= HALF_DOUBLES + 0.5 else 0.5
            score = count * weight + 0.01 * float(np.median(statures[track_id]))
            if score > best_score:
                best_id, best_score = track_id, score
        result[role] = best_id
    return result


@dataclass
class PlayerIdentity:
    role: str
    name: str
    description: str
    track_id: int | None
    source: str  # "gemini", "heuristic" or "none"
    box: list[float] | None = None


@dataclass
class Identity:
    players: dict[str, PlayerIdentity]
    score: str
    summary: str
    frame_index: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "players": {role: asdict(p) for role, p in self.players.items()},
            "score": self.score,
            "summary": self.summary,
            "frame_index": self.frame_index,
            "warnings": self.warnings,
        }

    @classmethod
    def from_json(cls, data: dict) -> "Identity":
        return cls(
            players={role: PlayerIdentity(**p) for role, p in data["players"].items()},
            score=data["score"],
            summary=data["summary"],
            frame_index=int(data.get("frame_index", 0)),
            warnings=list(data.get("warnings", [])),
        )

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "Identity":
        return cls.from_json(json.loads(Path(path).read_text()))


def resolve_identity(
    analysis: ClipAnalysis, boxes: dict, detections: Detections, camera: PinholeCamera, frame_index: int = 0
) -> Identity:
    warnings: list[str] = []
    fallback = heuristic_tracks(detections, camera)
    candidates = detections.frames[frame_index] if frame_index < len(detections.frames) else []
    used: set[int] = set()
    players: dict[str, PlayerIdentity] = {}
    for role in ROLES:
        description: PlayerDescription = getattr(analysis, role)
        box = boxes.get(role)
        track_id, source = None, "heuristic"
        if box is not None:
            free = [d for d in candidates if d.track_id not in used]
            det = match_box(box, free, detections.width, detections.height)
            if det is None:
                warnings.append(f"{role}: Gemini box matched no tracked person on frame {frame_index}")
            elif not on_half(role, court_xy(camera, det.foot)):
                warnings.append(f"{role}: Gemini picked track {det.track_id}, which is not on the {role} half")
            else:
                track_id, source = det.track_id, "gemini"
        if track_id is None and fallback[role] not in used:
            track_id = fallback[role]
        if track_id is None:
            warnings.append(f"{role}: no track found for this player")
            source = "none"
        else:
            used.add(track_id)
        players[role] = PlayerIdentity(
            role=role,
            name=description.name,
            description=description.appearance,
            track_id=track_id,
            source=source,
            box=[float(v) for v in box] if box is not None else None,
        )
    return Identity(players=players, score=analysis.score, summary=analysis.summary, frame_index=frame_index, warnings=warnings)


def _fingerprint() -> str:
    return hashlib.sha1(json.dumps([SYSTEM_PROMPT, TOOLS], sort_keys=True).encode()).hexdigest()


def get_or_create_assistant(client, cache_path: Path) -> str:
    """Reuse the Backboard assistant (and its memory) across runs while the prompt/tools are unchanged."""
    cache_path = Path(cache_path)
    fingerprint = _fingerprint()
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text())
        if cached.get("fingerprint") == fingerprint and cached.get("assistant_id"):
            return str(cached["assistant_id"])
    assistant_id = client.create_assistant("Horizon Tennis Analyst", SYSTEM_PROMPT, TOOLS)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"assistant_id": assistant_id, "fingerprint": fingerprint}))
    return assistant_id


def identify_direct(video_path: Path, frame_rgb: np.ndarray) -> tuple[ClipAnalysis, dict]:
    analysis = pegasus.analyze_clip(video_path)
    boxes = gemini.locate_players(frame_rgb, analysis.near.appearance, analysis.far.appearance)
    return analysis, boxes


def identify_via_backboard(
    client,
    match_id: str,
    video_path: Path,
    frame_rgb: np.ndarray,
    cache_path: Path,
    llm_provider: str | None = None,
    model_name: str | None = None,
) -> tuple[ClipAnalysis, dict, list[str]]:
    """Let the Backboard assistant drive both tools. Results are captured locally; missing steps run directly."""
    state: dict = {}

    def analyze_tool(_args: dict) -> dict:
        state["analysis"] = pegasus.analyze_clip(video_path)
        return state["analysis"].to_json()

    def locate_tool(args: dict) -> dict:
        analysis = state.get("analysis")
        near = args.get("near_appearance") or (analysis.near.appearance if analysis else "")
        far = args.get("far_appearance") or (analysis.far.appearance if analysis else "")
        state["boxes"] = gemini.locate_players(frame_rgb, near, far)
        return {role: list(box) for role, box in state["boxes"].items()}

    assistant_id = get_or_create_assistant(client, cache_path)
    thread_id = client.create_thread(assistant_id)
    result = run_tool_loop(
        client,
        thread_id,
        f"Identify the near and far players in tennis clip '{match_id}'.",
        {"analyze_tennis_clip": analyze_tool, "locate_players": locate_tool},
        llm_provider=llm_provider,
        model_name=model_name,
    )
    notes = [f"backboard: {e.name} failed: {e.error}" for e in result.executions if e.error]
    if "analysis" not in state:
        notes.append("backboard: assistant skipped analyze_tennis_clip; called Pegasus directly")
        state["analysis"] = pegasus.analyze_clip(video_path)
    if "boxes" not in state:
        notes.append("backboard: assistant skipped locate_players; called Gemini directly")
        analysis = state["analysis"]
        state["boxes"] = gemini.locate_players(frame_rgb, analysis.near.appearance, analysis.far.appearance)
    return state["analysis"], state["boxes"], notes
```

- [ ] **Step 6: Implement `pipeline/horizon/commands/identify.py`**

```python
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
```

- [ ] **Step 7: Run the tests and confirm they pass**

Run: `.venv\Scripts\python -m pytest -q`
Expected: `51 passed`.

- [ ] **Step 8: Commit**

```powershell
git add pipeline/horizon/llm/backboard.py pipeline/horizon/identify.py pipeline/horizon/commands/identify.py pipeline/tests/synthetic.py pipeline/tests/test_identify.py
git commit -m "feat(pipeline): identify players with Pegasus + Gemini orchestrated by Backboard"
```

### Task 8: Metric player trajectories (`horizon players`)

**Files:**
- Create: `pipeline/horizon/players.py`, `pipeline/horizon/commands/players.py`
- Test: `pipeline/tests/test_players.py`

**Interfaces:**
- Consumes: `court_xy`, `on_half`, `ROLES`, `Identity`, `PlayerIdentity` (Task 7), `Detections`, `Detection` (Task 5), `load_calibration` (Task 4), `PinholeCamera.height_above_ground` (Task 2), `tests/synthetic.py` (Task 7)
- Produces (`horizon.players`):
  - `EYE_HEIGHT_RATIO = 0.94`
  - `PlayerTrack(role, name, description, track_ids, bboxes, foot_xy (T,2), stature_m, speed_kmh (T,), distance_m (T,))`, with properties `eye_height_m` and `visible: list[bool]`, and method `eye(frame) -> np.ndarray(3)`
  - `Players(fps, frame_count, tracks: dict[str, PlayerTrack])`, with methods `opponent(role) -> PlayerTrack`, `save(path)`, and classmethod `load(path)`
  - `associate(detections, camera, role, start_track_id, max_base_jump=1.5, max_speed=10.0) -> (track_ids, dets, xys)`
  - `fill_gaps(xys) -> np.ndarray(T,2)`
  - `smooth(values (T,C), window) -> np.ndarray`
  - `kinematics(xy, fps) -> (speed_kmh, distance_m)`
  - `build_players(detections, camera, identity, smooth_window=7) -> Players` (raises `ValueError` if a role has no track)
- CLI: `horizon players --match-id <id> [--smooth 7]` writes `players.json`

- [ ] **Step 1: Write the failing tests**

`pipeline/tests/test_players.py`:
```python
import numpy as np
import pytest
from synthetic import H, W, broadcast_camera, person_detection

from horizon.identify import Identity, PlayerIdentity
from horizon.players import EYE_HEIGHT_RATIO, Players, associate, build_players, fill_gaps, kinematics, smooth
from horizon.tracking import Detections


def identity(near_id=5, far_id=9) -> Identity:
    return Identity(
        players={
            "near": PlayerIdentity("near", "A", "white shirt", near_id, "gemini"),
            "far": PlayerIdentity("far", "B", "navy shirt", far_id, "gemini"),
        },
        score="",
        summary="",
    )


def test_associate_follows_id_switch_skips_gaps_and_ignores_ball_kid():
    cam = broadcast_camera()
    frames = []
    for i in range(10):
        frame = [person_detection(cam, 2, 6.6, -12.5, stature=0.9)]
        if i != 3:
            frame.append(person_detection(cam, 5 if i < 5 else 11, 0.2 * i, -10.0))
        frames.append(frame)
    dets = Detections(W, H, 25.0, frames)
    ids, chosen, xys = associate(dets, cam, "near", start_track_id=5)
    assert ids == [5, 5, 5, None, 5, 11, 11, 11, 11, 11]
    assert chosen[3] is None and xys[3] is None
    assert xys[6] == pytest.approx((1.2, -10.0), abs=1e-6)


def test_fill_gaps_and_smoothing():
    filled = fill_gaps([None, (0.0, 0.0), None, (2.0, 2.0), None])
    assert filled.tolist() == [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [2.0, 2.0]]
    ramp = np.stack([np.arange(20.0), np.full(20, 3.0)], axis=1)
    smoothed = smooth(ramp, 7)
    assert np.allclose(smoothed[3:-3], ramp[3:-3])
    assert np.allclose(smoothed[:, 1], 3.0)
    assert np.allclose(smooth(ramp[:2], 7), ramp[:2])
    with pytest.raises(ValueError):
        fill_gaps([None, None])


def test_kinematics_constant_speed():
    xy = np.stack([np.arange(26) * 0.2, np.zeros(26)], axis=1)  # 0.2 m per frame at 25 fps = 5 m/s
    speed, distance = kinematics(xy, 25.0)
    assert np.allclose(speed, 18.0)
    assert distance[0] == 0.0 and distance[-1] == pytest.approx(5.0)
    single_speed, single_distance = kinematics(xy[:1], 25.0)
    assert single_speed.tolist() == [0.0] and single_distance.tolist() == [0.0]


def test_build_players_end_to_end(tmp_path):
    cam = broadcast_camera()
    frames = [[person_detection(cam, 5, 1.0, -10.0, 1.85), person_detection(cam, 9, -2.0, 10.0, 1.90)] for _ in range(12)]
    players = build_players(Detections(W, H, 25.0, frames), cam, identity())
    near, far = players.tracks["near"], players.tracks["far"]
    assert near.stature_m == pytest.approx(1.85, abs=1e-3)
    assert far.eye_height_m == pytest.approx(EYE_HEIGHT_RATIO * 1.90, abs=1e-3)
    assert np.allclose(near.foot_xy, [1.0, -10.0], atol=1e-6)
    assert np.allclose(near.speed_kmh, 0.0) and near.distance_m[-1] == pytest.approx(0.0)
    assert near.eye(4) == pytest.approx([1.0, -10.0, EYE_HEIGHT_RATIO * 1.85], abs=1e-3)
    assert players.opponent("near") is far
    assert all(near.visible) and near.name == "A"
    players.save(tmp_path / "players.json")
    again = Players.load(tmp_path / "players.json")
    assert again.frame_count == 12 and again.fps == 25.0
    assert np.allclose(again.tracks["far"].foot_xy, far.foot_xy, atol=1e-4)
    assert again.tracks["near"].track_ids == [5] * 12


def test_build_players_requires_identified_tracks():
    cam = broadcast_camera()
    with pytest.raises(ValueError, match="near"):
        build_players(Detections(W, H, 25.0, [[]]), cam, identity(near_id=None))
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.venv\Scripts\python -m pytest tests/test_players.py -q`
Expected: `ModuleNotFoundError: No module named 'horizon.players'`.

- [ ] **Step 3: Implement `pipeline/horizon/players.py`**

```python
"""Per-frame metric trajectories (court metres) for the near and far players."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from horizon.camera import PinholeCamera
from horizon.identify import ROLES, Identity, court_xy, on_half
from horizon.tracking import Detection, Detections

EYE_HEIGHT_RATIO = 0.94  # eye height / stature for adults


@dataclass
class PlayerTrack:
    role: str
    name: str
    description: str
    track_ids: list[int | None]
    bboxes: list[tuple[float, float, float, float] | None]
    foot_xy: np.ndarray  # (T, 2) smoothed court position in metres
    stature_m: float
    speed_kmh: np.ndarray  # (T,)
    distance_m: np.ndarray  # (T,) cumulative

    @property
    def eye_height_m(self) -> float:
        return EYE_HEIGHT_RATIO * self.stature_m

    @property
    def visible(self) -> list[bool]:
        return [b is not None for b in self.bboxes]

    def eye(self, frame: int) -> np.ndarray:
        x, y = self.foot_xy[frame]
        return np.array([x, y, self.eye_height_m])

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "stature_m": round(self.stature_m, 4),
            "track_ids": self.track_ids,
            "bboxes": [None if b is None else [round(v, 2) for v in b] for b in self.bboxes],
            "foot_xy": np.round(self.foot_xy, 4).tolist(),
            "speed_kmh": np.round(self.speed_kmh, 3).tolist(),
            "distance_m": np.round(self.distance_m, 3).tolist(),
        }

    @classmethod
    def from_json(cls, role: str, data: dict) -> "PlayerTrack":
        return cls(
            role=role,
            name=data["name"],
            description=data["description"],
            track_ids=list(data["track_ids"]),
            bboxes=[None if b is None else tuple(b) for b in data["bboxes"]],
            foot_xy=np.array(data["foot_xy"], dtype=np.float64).reshape(-1, 2),
            stature_m=float(data["stature_m"]),
            speed_kmh=np.array(data["speed_kmh"], dtype=np.float64),
            distance_m=np.array(data["distance_m"], dtype=np.float64),
        )


@dataclass
class Players:
    fps: float
    frame_count: int
    tracks: dict[str, PlayerTrack]

    def opponent(self, role: str) -> PlayerTrack:
        return self.tracks["far" if role == "near" else "near"]

    def save(self, path: Path) -> None:
        data = {
            "fps": self.fps,
            "frame_count": self.frame_count,
            "players": {role: track.to_json() for role, track in self.tracks.items()},
        }
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path: Path) -> "Players":
        data = json.loads(Path(path).read_text())
        tracks = {role: PlayerTrack.from_json(role, t) for role, t in data["players"].items()}
        return cls(fps=float(data["fps"]), frame_count=int(data["frame_count"]), tracks=tracks)


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def associate(
    detections: Detections,
    camera: PinholeCamera,
    role: str,
    start_track_id: int,
    max_base_jump: float = 1.5,
    max_speed: float = 10.0,
):
    """Follow one player through the clip on the court plane, surviving ByteTrack ID switches.

    Keeps the current track id while it is present on the player's half; otherwise takes the closest
    person on that half if they are within reach (max_base_jump + max_speed * elapsed seconds).
    """
    current = start_track_id
    last_xy: tuple[float, float] | None = None
    last_frame: int | None = None
    for i, frame in enumerate(detections.frames):
        det = next((d for d in frame if d.track_id == current), None)
        if det is not None:
            last_xy, last_frame = court_xy(camera, det.foot), i
            break
    ids: list[int | None] = []
    chosen: list[Detection | None] = []
    xys: list[tuple[float, float] | None] = []
    for i, frame in enumerate(detections.frames):
        options = []
        for det in frame:
            xy = court_xy(camera, det.foot)
            if on_half(role, xy):
                options.append((det, xy))
        pick = next(((d, xy) for d, xy in options if d.track_id == current), None)
        if pick is None and options and last_xy is not None:
            elapsed = abs(i - last_frame) / detections.fps if last_frame is not None else 0.0
            det, xy = min(options, key=lambda o: _distance(o[1], last_xy))
            if _distance(xy, last_xy) <= max_base_jump + max_speed * elapsed:
                pick = (det, xy)
                current = det.track_id
        if pick is None:
            ids.append(None)
            chosen.append(None)
            xys.append(None)
        else:
            det, xy = pick
            ids.append(det.track_id)
            chosen.append(det)
            xys.append(xy)
            last_xy, last_frame = xy, i
    return ids, chosen, xys


def fill_gaps(xys: list[tuple[float, float] | None]) -> np.ndarray:
    known = [i for i, v in enumerate(xys) if v is not None]
    if not known:
        raise ValueError("player was never detected on their half of the court")
    values = np.array([xys[i] for i in known], dtype=np.float64)
    t = np.arange(len(xys))
    return np.stack([np.interp(t, known, values[:, 0]), np.interp(t, known, values[:, 1])], axis=1)


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average that stays unbiased at the ends (normalised convolution)."""
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    window = min(window, n if n % 2 else n - 1)
    if window <= 1:
        return values.copy()
    kernel = np.ones(window)
    weights = np.convolve(np.ones(n), kernel, mode="same")
    return np.stack([np.convolve(values[:, c], kernel, mode="same") / weights for c in range(values.shape[1])], axis=1)


def kinematics(xy: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Speed (km/h, central differences) and cumulative distance (m) along a trajectory."""
    if len(xy) < 2:
        return np.zeros(len(xy)), np.zeros(len(xy))
    steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    distance = np.concatenate([[0.0], np.cumsum(steps)])
    speed = np.linalg.norm(np.gradient(xy, axis=0), axis=1) * fps * 3.6
    return speed, distance


def build_players(detections: Detections, camera: PinholeCamera, identity: Identity, smooth_window: int = 7) -> Players:
    tracks: dict[str, PlayerTrack] = {}
    for role in ROLES:
        ident = identity.players[role]
        if ident.track_id is None:
            raise ValueError(f"No track identified for the {role} player; run `horizon identify` first")
        ids, chosen, xys = associate(detections, camera, role, ident.track_id)
        foot = smooth(fill_gaps(xys), smooth_window)
        speed, distance = kinematics(foot, detections.fps)
        statures = [camera.height_above_ground(xy, det.bbox[1]) for det, xy in zip(chosen, xys) if det is not None]
        stature = float(np.clip(np.median(statures), 1.3, 2.2)) if statures else 1.8
        tracks[role] = PlayerTrack(
            role=role,
            name=ident.name,
            description=ident.description,
            track_ids=ids,
            bboxes=[None if d is None else d.bbox for d in chosen],
            foot_xy=foot,
            stature_m=stature,
            speed_kmh=speed,
            distance_m=distance,
        )
    return Players(fps=detections.fps, frame_count=len(detections.frames), tracks=tracks)
```

- [ ] **Step 4: Implement `pipeline/horizon/commands/players.py`**

```python
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
```

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `.venv\Scripts\python -m pytest -q`
Expected: `56 passed`.

- [ ] **Step 6: Commit**

```powershell
git add pipeline/horizon/players.py pipeline/horizon/commands/players.py pipeline/tests/test_players.py
git commit -m "feat(pipeline): build metric player trajectories with stature, speed and distance"
```

### Task 9: Monocular depth and metric alignment

**Files:**
- Create: `pipeline/horizon/depth.py`
- Test: `pipeline/tests/test_depth.py`

**Interfaces:**
- Consumes: `PinholeCamera` (Task 2), `court_rect`, `net_quad` (Task 2), `tests/synthetic.py` (Task 7)
- Produces (`horizon.depth`):
  - `DEFAULT_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"`
  - `DepthEstimator` Protocol: `__call__(rgb uint8 HxWx3) -> float32 HxW` relative disparity (larger = closer)
  - `DepthAnythingV2(model_id=DEFAULT_DEPTH_MODEL, device=None)` (implements `DepthEstimator`, lazy torch/transformers)
  - `pixel_grid(width, height, stride=1) -> (us, vs)` float grids
  - `ground_depth_map(camera) -> HxW` camera-z depth of the plane z=0 (`inf` where rays miss)
  - `court_fit_mask(camera, exclude=None, margin_x=2.0, margin_y=3.0) -> bool HxW`
  - `DisparityAlignment(scale, shift, residual)`, with `.depth(disparity, max_depth=120.0) -> depth`
  - `fit_disparity_alignment(disparity, target_depth, mask, max_samples=20000, trim=0.2, seed=0) -> DisparityAlignment` (raises `ValueError` for <50 pixels or scale ≤ 0)
  - `anchor_player_depth(depth, mask, foot_depth, band=0.6) -> 1-D depths for mask pixels (row-major order)`

- [ ] **Step 1: Write the failing tests**

`pipeline/tests/test_depth.py`:
```python
import numpy as np
import pytest
from synthetic import H, W, broadcast_camera

from horizon.depth import (
    DisparityAlignment,
    anchor_player_depth,
    court_fit_mask,
    fit_disparity_alignment,
    ground_depth_map,
)


def test_ground_depth_map_grows_towards_the_top_of_the_image():
    depth = ground_depth_map(broadcast_camera())
    assert depth.shape == (H, W)
    assert np.isfinite(depth).all()
    assert depth[-1, W // 2] < depth[H // 2, W // 2] < depth[0, W // 2]


def test_fit_recovers_affine_disparity_despite_outliers():
    cam = broadcast_camera()
    z = ground_depth_map(cam)
    scale, shift = 0.37, -0.004
    disparity = (1.0 / z - shift) / scale
    rng = np.random.default_rng(0)
    outliers = rng.random(disparity.shape) < 0.05
    disparity[outliers] += rng.uniform(0.5, 2.0, outliers.sum())
    mask = court_fit_mask(cam)
    fit = fit_disparity_alignment(disparity, z, mask)
    assert fit.scale == pytest.approx(scale, rel=1e-4)
    assert fit.shift == pytest.approx(shift, abs=1e-6)
    clean = mask & ~outliers
    assert np.allclose(fit.depth(disparity)[clean], z[clean], rtol=1e-4)


def test_fit_rejects_depth_like_output_and_tiny_masks():
    cam = broadcast_camera()
    z = ground_depth_map(cam)
    with pytest.raises(ValueError, match="disparity"):
        fit_disparity_alignment(z.copy(), z, court_fit_mask(cam))
    tiny = np.zeros((H, W), bool)
    tiny[400, 600:610] = True
    with pytest.raises(ValueError, match="ground pixels"):
        fit_disparity_alignment(1.0 / z, z, tiny)


def test_alignment_clamps_far_values():
    depth = DisparityAlignment(scale=1.0, shift=0.0, residual=0.0).depth(np.array([0.0, 0.5]), max_depth=120.0)
    assert depth.tolist() == [120.0, 2.0]


def test_court_fit_mask_excludes_net_and_players():
    cam = broadcast_camera()
    (u_court, v_court), (u_net, v_net) = cam.project(np.array([[0.0, -6.0, 0.0], [0.0, 0.0, 0.8]]))[0]
    players = np.zeros((H, W), bool)
    players[int(v_court) - 5 : int(v_court) + 5, int(u_court) - 5 : int(u_court) + 5] = True
    assert court_fit_mask(cam)[int(round(v_court)), int(round(u_court))]
    assert not court_fit_mask(cam)[int(round(v_net)), int(round(u_net))]
    assert not court_fit_mask(cam, exclude=players)[int(v_court), int(u_court)]


def test_anchor_player_depth_moves_feet_to_the_calibrated_depth():
    depth = np.zeros((40, 20))
    rows = np.arange(40, dtype=float)[:, None].repeat(20, axis=1)
    depth[:] = 5.0 + 0.01 * rows
    mask = np.zeros((40, 20), bool)
    mask[10:30, 5:15] = True
    depth[12, 8] = 15.0  # outlier inside the person
    anchored = anchor_player_depth(depth, mask, foot_depth=20.0, band=0.6)
    assert anchored.shape == (int(mask.sum()),)
    feet = anchored.reshape(20, 10)[-2:]
    assert np.allclose(feet, 20.0, atol=0.006)
    assert anchored.max() == pytest.approx(20.6)
    assert anchored.min() >= 19.4
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.venv\Scripts\python -m pytest tests/test_depth.py -q`
Expected: `ModuleNotFoundError: No module named 'horizon.depth'`.

- [ ] **Step 3: Implement `pipeline/horizon/depth.py`**

```python
"""Monocular depth (Depth Anything V2) and its metric alignment to the calibrated court plane.

Depth Anything V2 relative checkpoints predict affine-invariant disparity (larger = closer). On pixels that
certainly show the court, the calibrated camera tells us the true depth, so we fit
scale * disparity + shift = 1 / depth by trimmed least squares and apply it to the whole frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from horizon.camera import PinholeCamera
from horizon.court import court_rect, net_quad

DEFAULT_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


class DepthEstimator(Protocol):
    def __call__(self, rgb: np.ndarray) -> np.ndarray: ...


class DepthAnythingV2:
    """Hugging Face `depth-estimation` pipeline wrapper returning full-resolution disparity."""

    def __init__(self, model_id: str = DEFAULT_DEPTH_MODEL, device: str | None = None):
        import torch
        from transformers import pipeline

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._pipe = pipeline(task="depth-estimation", model=model_id, device=device)

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        from PIL import Image

        predicted = self._pipe(Image.fromarray(rgb))["predicted_depth"]
        if hasattr(predicted, "detach"):
            predicted = predicted.detach().float().cpu().numpy()
        disparity = np.squeeze(np.asarray(predicted, dtype=np.float32))
        h, w = rgb.shape[:2]
        if disparity.shape != (h, w):
            disparity = cv2.resize(disparity, (w, h), interpolation=cv2.INTER_LINEAR)
        return disparity


def pixel_grid(width: int, height: int, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    vs, us = np.mgrid[0:height:stride, 0:width:stride]
    return us.astype(np.float64), vs.astype(np.float64)


def ground_depth_map(camera: PinholeCamera) -> np.ndarray:
    us, vs = pixel_grid(camera.width, camera.height)
    _, depth = camera.ground_intersection(us.ravel(), vs.ravel())
    return depth.reshape(camera.height, camera.width)


def _fill(mask: np.ndarray, camera: PinholeCamera, polygon: np.ndarray, value: int) -> None:
    uv, z = camera.project(polygon)
    if (z > 0).all():
        cv2.fillPoly(mask, [np.round(uv).astype(np.int32).reshape(-1, 1, 2)], value)


def court_fit_mask(
    camera: PinholeCamera, exclude: np.ndarray | None = None, margin_x: float = 2.0, margin_y: float = 3.0
) -> np.ndarray:
    """Pixels that certainly show the court surface: court + run-off, minus the net and `exclude` (e.g. players)."""
    mask = np.zeros((camera.height, camera.width), np.uint8)
    _fill(mask, camera, court_rect(margin_x, margin_y), 1)
    _fill(mask, camera, net_quad(), 0)
    result = mask.astype(bool)
    if exclude is not None:
        result &= ~exclude
    return result


@dataclass(frozen=True)
class DisparityAlignment:
    scale: float
    shift: float
    residual: float  # RMS residual (1/m) on inliers

    def depth(self, disparity: np.ndarray, max_depth: float = 120.0) -> np.ndarray:
        inverse = self.scale * np.asarray(disparity, dtype=np.float64) + self.shift
        return 1.0 / np.maximum(inverse, 1.0 / max_depth)


def fit_disparity_alignment(
    disparity: np.ndarray,
    target_depth: np.ndarray,
    mask: np.ndarray,
    max_samples: int = 20000,
    trim: float = 0.2,
    seed: int = 0,
) -> DisparityAlignment:
    sel = mask & np.isfinite(target_depth) & (target_depth > 0)
    d = disparity[sel].astype(np.float64)
    y = 1.0 / target_depth[sel].astype(np.float64)
    if d.size < 50:
        raise ValueError(f"Only {d.size} ground pixels available for depth alignment (need >= 50)")
    if d.size > max_samples:
        idx = np.random.default_rng(seed).choice(d.size, max_samples, replace=False)
        d, y = d[idx], y[idx]
    A = np.stack([d, np.ones_like(d)], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    residual = np.abs(A @ coef - y)
    keep = residual <= np.quantile(residual, 1.0 - trim)
    coef, *_ = np.linalg.lstsq(A[keep], y[keep], rcond=None)
    scale, shift = float(coef[0]), float(coef[1])
    if scale <= 0:
        raise ValueError(
            "Depth model output does not behave like disparity (fitted scale <= 0); "
            "use a relative Depth Anything V2 checkpoint"
        )
    rms = float(np.sqrt(np.mean((A[keep] @ coef - y[keep]) ** 2)))
    return DisparityAlignment(scale=scale, shift=shift, residual=rms)


def anchor_player_depth(depth: np.ndarray, mask: np.ndarray, foot_depth: float, band: float = 0.6) -> np.ndarray:
    """Depths of the mask pixels, shifted so the lowest 10 % of rows sit at foot_depth, clamped to +-band m."""
    values = depth[mask].astype(np.float64)
    rows = np.nonzero(mask.any(axis=1))[0]
    if rows.size == 0:
        return values
    top, bottom = int(rows[0]), int(rows[-1])
    cutoff = bottom - max(1, int(round(0.1 * (bottom - top + 1)))) + 1
    feet = mask.copy()
    feet[:cutoff] = False
    offset = foot_depth - float(np.median(depth[feet]))
    return np.clip(values + offset, foot_depth - band, foot_depth + band)
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `.venv\Scripts\python -m pytest -q`
Expected: `62 passed`.

- [ ] **Step 5: Commit**

```powershell
git add pipeline/horizon/depth.py pipeline/tests/test_depth.py
git commit -m "feat(pipeline): align Depth Anything V2 disparity to the calibrated court plane"
```

### Task 10: Ground texture and point-cloud reconstruction (`horizon reconstruct`)

**Files:**
- Create: `pipeline/horizon/ground.py`, `pipeline/horizon/reconstruct.py`, `pipeline/horizon/commands/reconstruct.py`
- Modify: `pipeline/tests/synthetic.py` (append scene-rendering helpers)
- Test: `pipeline/tests/test_reconstruct.py`

**Interfaces:**
- Consumes:
  - `ground_depth_map`, `court_fit_mask`, `fit_disparity_alignment`, `anchor_player_depth`, `pixel_grid`, `DepthEstimator`, `DepthAnythingV2`, `DEFAULT_DEPTH_MODEL` (Task 9)
  - `Players`, `build_players` (Task 8)
  - `Identity`, `PlayerIdentity` (Task 7)
  - `Detections`, `Detection` (Task 5)
  - `iter_frames`, `H264Writer` (Task 3)
  - `load_calibration` (Task 4)
  - `camera_from_dict`, `camera_to_dict` (Task 2)
- Produces (`horizon.ground`):
  - `DEFAULT_EXTENT = (-10.0, 10.0, -20.0, 20.0)`
  - `GroundTexture(image, x_min, x_max, y_min, y_max, fill)`, with properties `resolution` and `shape` and methods `sample(x, y) -> (N,3) uint8` and `world_points(stride=1) -> (points (N,3), colors (N,3))`
  - `build_ground_texture(plate_rgb, camera, ground_mask, extent=DEFAULT_EXTENT, resolution=0.025) -> GroundTexture`
- Produces (`horizon.reconstruct`):
  - `ROLE_CODES = {"near": 0, "far": 1}`
  - `Scene(background_points, background_colors, background_radii, player_points, player_colors, player_radii, player_roles, frame_offsets, ground, camera, fps)`, with property `frame_count`, method `players_at(frame, exclude_role=None) -> (points, colors, radii)`, `save(path)`, and classmethod `load(path)`
  - `median_plate(frames, masks, dilate_px=9) -> uint8 HxWx3`
  - `player_detection(detections, track, frame) -> Detection | None`
  - `background_cloud(plate, disparity, camera, ground_depth, extent, stride=2, ground_tolerance=0.12, max_depth=120.0) -> (points, colors, radii, ground_mask, alignment)`
  - `reconstruct_scene(video_path, camera, players, detections, estimator, stride=2, samples=24, extent=DEFAULT_EXTENT, resolution=0.025, log=None) -> Scene`
- Produces (`tests/synthetic.py` additions):
  - constants `WALL_Y`, `WALL_HEIGHT`, `COURT_BLUE`, `SURROUND_GREEN`, `LINE_WHITE`, `WALL_RED`, `PLAYER_MAGENTA`, `TEST_EXTENT`
  - `ground_colors(x, y)`
  - `render_synthetic_frame(camera, people=()) -> (rgb, true_depth)`
  - `billboard_detection(camera, track_id, x, y, stature)`
  - `make_identity(near_id, far_id)`
  - `FakeEstimator(background_depth, scale=0.37, shift=-0.004, player_depth=30.0)`
- CLI: `horizon reconstruct --match-id <id> [--depth-model ...] [--stride 2] [--samples 24]` writes `scene.npz` and `ground.png`

- [ ] **Step 1: Append the synthetic scene helpers to `pipeline/tests/synthetic.py`**

Add these imports at the top of the file (below `import numpy as np`):
```python
import cv2

from horizon.court import HALF_DOUBLES, HALF_LENGTH, HALF_SINGLES
from horizon.depth import pixel_grid
from horizon.identify import Identity, PlayerIdentity
```
and append:
```python
WALL_Y, WALL_HEIGHT = 22.0, 12.0
COURT_BLUE, SURROUND_GREEN, LINE_WHITE = (40, 80, 170), (40, 120, 70), (240, 240, 240)
WALL_RED, PLAYER_MAGENTA = (170, 40, 40), (255, 0, 255)
TEST_EXTENT = (-40.0, 40.0, -30.0, 25.0)  # covers every ground pixel the synthetic camera sees


def ground_colors(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    colors = np.empty((len(x), 3), np.uint8)
    colors[:] = SURROUND_GREEN
    court = (np.abs(x) <= HALF_DOUBLES) & (np.abs(y) <= HALF_LENGTH)
    colors[court] = COURT_BLUE
    line = court & (
        (np.abs(np.abs(y) - HALF_LENGTH) < 0.05)
        | (np.abs(np.abs(x) - HALF_DOUBLES) < 0.05)
        | (np.abs(np.abs(x) - HALF_SINGLES) < 0.05)
    )
    colors[line] = LINE_WHITE
    return colors


def render_synthetic_frame(camera: PinholeCamera, people=()) -> tuple[np.ndarray, np.ndarray]:
    """Court ground + a red wall behind the far baseline + magenta billboards (x, y, stature). Returns (rgb, depth)."""
    us, vs = pixel_grid(camera.width, camera.height)
    u, v = us.ravel(), vs.ravel()
    ground_pts, ground_depth = camera.ground_intersection(u, v)
    rays = camera.ray_directions(u, v)
    c = camera.center
    with np.errstate(divide="ignore", invalid="ignore"):
        wall_t = (WALL_Y - c[1]) / rays[:, 1]
    wall_z = c[2] + rays[:, 2] * wall_t
    wall_hit = (wall_t > 0) & (wall_z >= 0) & (wall_z <= WALL_HEIGHT) & (wall_t < ground_depth)
    depth = np.where(wall_hit, wall_t, ground_depth).reshape(camera.height, camera.width)
    rgb = ground_colors(ground_pts[:, 0], ground_pts[:, 1])
    rgb[wall_hit] = WALL_RED
    rgb = rgb.reshape(camera.height, camera.width, 3)
    for x, y, stature in people:
        quad = np.array([[x - 0.3, y, 0.0], [x + 0.3, y, 0.0], [x + 0.3, y, stature], [x - 0.3, y, stature]])
        uv, z = camera.project(quad)
        mask = np.zeros((camera.height, camera.width), np.uint8)
        cv2.fillPoly(mask, [np.round(uv).astype(np.int32).reshape(-1, 1, 2)], 1)
        mask = mask.astype(bool)
        rgb[mask] = PLAYER_MAGENTA
        depth[mask] = float(np.mean(z))
    return rgb, depth


def billboard_detection(camera: PinholeCamera, track_id: int, x: float, y: float, stature: float) -> Detection:
    quad = np.array([[x - 0.3, y, 0.0], [x + 0.3, y, 0.0], [x + 0.3, y, stature], [x - 0.3, y, stature]])
    uv, _ = camera.project(quad)
    bbox = (float(uv[:, 0].min()), float(uv[:, 1].min()), float(uv[:, 0].max()), float(uv[:, 1].max()))
    return Detection(track_id=track_id, conf=0.9, bbox=bbox, polygon=tuple((float(a), float(b)) for a, b in uv))


def make_identity(near_id: int | None = 5, far_id: int | None = 9) -> Identity:
    return Identity(
        players={
            "near": PlayerIdentity("near", "Near Player", "white shirt", near_id, "gemini"),
            "far": PlayerIdentity("far", "Far Player", "navy shirt", far_id, "gemini"),
        },
        score="",
        summary="",
    )


class FakeEstimator:
    """Returns exact disparity for the synthetic background; magenta pixels get one constant 'player' disparity."""

    def __init__(self, background_depth: np.ndarray, scale: float = 0.37, shift: float = -0.004, player_depth: float = 30.0):
        self.base = (1.0 / background_depth - shift) / scale
        self.player = (1.0 / player_depth - shift) / scale

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        disparity = self.base.copy()
        magenta = (rgb[..., 0] > 180) & (rgb[..., 1] < 90) & (rgb[..., 2] > 180)
        disparity[magenta] = self.player
        return disparity.astype(np.float32)
```

- [ ] **Step 2: Write the failing tests**

`pipeline/tests/test_reconstruct.py`:
```python
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

from horizon.court import HALF_LENGTH
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
    colors = texture.sample(np.array([0.0, 8.0, 0.0]), np.array([-6.0, 0.0, -HALF_LENGTH]))
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
```

- [ ] **Step 3: Run the tests and confirm they fail**

Run: `.venv\Scripts\python -m pytest tests/test_reconstruct.py -q`
Expected: `ModuleNotFoundError: No module named 'horizon.ground'`.

- [ ] **Step 4: Implement `pipeline/horizon/ground.py`**

```python
"""Top-down texture (orthophoto) of the ground plane, sampled from the clean plate through the broadcast camera."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from horizon.camera import PinholeCamera
from horizon.court import HALF_DOUBLES, HALF_LENGTH

DEFAULT_EXTENT = (-10.0, 10.0, -20.0, 20.0)  # x_min, x_max, y_min, y_max in metres
DEFAULT_FILL = (40, 90, 60)


@dataclass(frozen=True, eq=False)
class GroundTexture:
    image: np.ndarray  # (rows, cols, 3) uint8; row 0 is y = y_max (far end), column 0 is x = x_min
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    fill: tuple[int, int, int]

    @property
    def shape(self) -> tuple[int, int]:
        return self.image.shape[0], self.image.shape[1]

    @property
    def resolution(self) -> float:
        return (self.x_max - self.x_min) / self.image.shape[1]

    def sample(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Nearest-texel colours for ground points; the fill colour outside the texture."""
        rows, cols = self.shape
        ci = np.rint((np.asarray(x) - self.x_min) / self.resolution - 0.5).astype(np.int64)
        ri = np.rint((self.y_max - np.asarray(y)) / self.resolution - 0.5).astype(np.int64)
        inside = (ci >= 0) & (ci < cols) & (ri >= 0) & (ri < rows)
        out = np.empty((len(ci), 3), np.uint8)
        out[:] = self.fill
        out[inside] = self.image[ri[inside], ci[inside]]
        return out

    def world_points(self, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
        rows, cols = np.mgrid[0 : self.shape[0] : stride, 0 : self.shape[1] : stride]
        x = self.x_min + (cols + 0.5) * self.resolution
        y = self.y_max - (rows + 0.5) * self.resolution
        points = np.stack([x.ravel(), y.ravel(), np.zeros(x.size)], axis=1)
        return points.astype(np.float32), self.image[rows.ravel(), cols.ravel()]


def build_ground_texture(
    plate_rgb: np.ndarray,
    camera: PinholeCamera,
    ground_mask: np.ndarray,
    extent: tuple[float, float, float, float] = DEFAULT_EXTENT,
    resolution: float = 0.025,
) -> GroundTexture:
    x_min, x_max, y_min, y_max = extent
    cols = int(round((x_max - x_min) / resolution))
    rows = int(round((y_max - y_min) / resolution))
    r, c = np.mgrid[0:rows, 0:cols]
    x = x_min + (c + 0.5) * resolution
    y = y_max - (r + 0.5) * resolution
    uv, z = camera.project(np.stack([x.ravel(), y.ravel(), np.zeros(x.size)], axis=1))
    unusable = (z <= 0) | ~np.isfinite(uv).all(axis=1)
    uv[unusable] = -1.0
    map_x = uv[:, 0].reshape(rows, cols).astype(np.float32)
    map_y = uv[:, 1].reshape(rows, cols).astype(np.float32)
    texture = cv2.remap(plate_rgb, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    valid = cv2.remap(
        ground_mask.astype(np.uint8), map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    ).astype(bool)
    valid &= ~unusable.reshape(rows, cols)
    ring = valid & ((np.abs(x) > HALF_DOUBLES + 0.5) | (np.abs(y) > HALF_LENGTH + 0.5))
    if ring.sum() >= 100:
        fill = tuple(int(v) for v in np.median(texture[ring], axis=0))
    elif valid.any():
        fill = tuple(int(v) for v in np.median(texture[valid], axis=0))
    else:
        fill = DEFAULT_FILL
    texture[~valid] = fill
    return GroundTexture(texture, x_min, x_max, y_min, y_max, fill)
```

- [ ] **Step 5: Implement `pipeline/horizon/reconstruct.py`**

```python
"""Point clouds from the clip: one static background cloud plus per-frame player clouds.

Pipeline: masked-median clean plate -> Depth Anything disparity aligned to the calibrated court plane ->
pixels near the plane become the ground texture, the rest become background points. Every frame, each player's
mask pixels are lifted with that frame's aligned depth, anchored so their feet sit at the calibrated foot depth.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from horizon.camera import PinholeCamera, camera_from_dict, camera_to_dict
from horizon.depth import (
    DepthEstimator,
    DisparityAlignment,
    anchor_player_depth,
    court_fit_mask,
    fit_disparity_alignment,
    ground_depth_map,
    pixel_grid,
)
from horizon.ground import DEFAULT_EXTENT, GroundTexture, build_ground_texture
from horizon.players import PlayerTrack, Players
from horizon.tracking import Detection, Detections
from horizon.video import iter_frames

ROLE_CODES = {"near": 0, "far": 1}


@dataclass(frozen=True, eq=False)
class Scene:
    background_points: np.ndarray  # (N, 3) float32 world
    background_colors: np.ndarray  # (N, 3) uint8
    background_radii: np.ndarray  # (N,) float32 world footprint radius in metres
    player_points: np.ndarray  # (M, 3) float32, all frames concatenated
    player_colors: np.ndarray  # (M, 3) uint8
    player_radii: np.ndarray  # (M,) float32
    player_roles: np.ndarray  # (M,) uint8, see ROLE_CODES
    frame_offsets: np.ndarray  # (T + 1,) int64: frame i owns [offsets[i], offsets[i + 1])
    ground: GroundTexture
    camera: PinholeCamera  # the calibrated broadcast camera
    fps: float

    @property
    def frame_count(self) -> int:
        return len(self.frame_offsets) - 1

    def players_at(self, frame: int, exclude_role: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        start, end = int(self.frame_offsets[frame]), int(self.frame_offsets[frame + 1])
        points = self.player_points[start:end]
        colors = self.player_colors[start:end]
        radii = self.player_radii[start:end]
        if exclude_role is not None:
            keep = self.player_roles[start:end] != ROLE_CODES[exclude_role]
            points, colors, radii = points[keep], colors[keep], radii[keep]
        return points, colors, radii

    def save(self, path: Path) -> None:
        cam = camera_to_dict(self.camera)
        np.savez_compressed(
            path,
            background_points=self.background_points,
            background_colors=self.background_colors,
            background_radii=self.background_radii,
            player_points=self.player_points,
            player_colors=self.player_colors,
            player_radii=self.player_radii,
            player_roles=self.player_roles,
            frame_offsets=self.frame_offsets,
            ground_image=self.ground.image,
            ground_extent=np.array([self.ground.x_min, self.ground.x_max, self.ground.y_min, self.ground.y_max]),
            ground_fill=np.array(self.ground.fill, dtype=np.uint8),
            camera_K=np.array(cam["K"]),
            camera_R=np.array(cam["R"]),
            camera_t=np.array(cam["t"]),
            camera_size=np.array([cam["width"], cam["height"]]),
            fps=np.array(self.fps),
        )

    @classmethod
    def load(cls, path: Path) -> "Scene":
        with np.load(path) as d:
            x_min, x_max, y_min, y_max = (float(v) for v in d["ground_extent"])
            ground = GroundTexture(d["ground_image"], x_min, x_max, y_min, y_max, tuple(int(v) for v in d["ground_fill"]))
            width, height = (int(v) for v in d["camera_size"])
            camera = camera_from_dict(
                {"K": d["camera_K"], "R": d["camera_R"], "t": d["camera_t"], "width": width, "height": height}
            )
            return cls(
                background_points=d["background_points"],
                background_colors=d["background_colors"],
                background_radii=d["background_radii"],
                player_points=d["player_points"],
                player_colors=d["player_colors"],
                player_radii=d["player_radii"],
                player_roles=d["player_roles"],
                frame_offsets=d["frame_offsets"],
                ground=ground,
                camera=camera,
                fps=float(d["fps"]),
            )


def median_plate(frames: list[np.ndarray], masks: list[np.ndarray], dilate_px: int = 9) -> np.ndarray:
    """Per-pixel median over frames, ignoring (dilated) player pixels; pixels never visible are inpainted."""
    kernel = np.ones((dilate_px, dilate_px), np.uint8)
    stack = []
    for frame, mask in zip(frames, masks):
        grown = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
        values = frame.astype(np.float32)
        values[grown] = np.nan
        stack.append(values)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN pixels are handled below
        plate = np.nanmedian(np.stack(stack), axis=0)
    holes = np.isnan(plate[..., 0])
    plate = np.nan_to_num(plate, nan=0.0).clip(0, 255).astype(np.uint8)
    if holes.any():
        plate = cv2.inpaint(plate, holes.astype(np.uint8), 5, cv2.INPAINT_TELEA)
    return plate


def player_detection(detections: Detections, track: PlayerTrack, frame: int) -> Detection | None:
    track_id = track.track_ids[frame] if frame < len(track.track_ids) else None
    return None if track_id is None else detections.by_track(frame, track_id)


def _player_masks(detections: Detections, players: Players, frame: int) -> dict[str, tuple[Detection, np.ndarray]]:
    out = {}
    for role, track in players.tracks.items():
        det = player_detection(detections, track, frame)
        if det is not None:
            out[role] = (det, det.mask(detections.width, detections.height))
    return out


def background_cloud(
    plate: np.ndarray,
    disparity: np.ndarray,
    camera: PinholeCamera,
    ground_depth: np.ndarray,
    extent: tuple[float, float, float, float],
    stride: int = 2,
    ground_tolerance: float = 0.12,
    max_depth: float = 120.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, DisparityAlignment]:
    fit_mask = court_fit_mask(camera)
    alignment = fit_disparity_alignment(disparity, ground_depth, fit_mask)
    depth = alignment.depth(disparity, max_depth)
    x_min, x_max, y_min, y_max = extent
    us, vs = pixel_grid(camera.width, camera.height)
    hits, _ = camera.ground_intersection(us.ravel(), vs.ravel())
    gx = hits[:, 0].reshape(depth.shape)
    gy = hits[:, 1].reshape(depth.shape)
    in_extent = np.isfinite(ground_depth) & (gx >= x_min) & (gx <= x_max) & (gy >= y_min) & (gy <= y_max)
    with np.errstate(invalid="ignore"):
        near_plane = np.abs(depth - ground_depth) <= ground_tolerance * ground_depth
    ground = (in_extent & near_plane) | fit_mask
    keep = np.zeros_like(ground)
    keep[::stride, ::stride] = True
    keep &= ~ground & (depth < max_depth * 0.999)
    v_idx, u_idx = np.nonzero(keep)
    z = depth[v_idx, u_idx]
    points = camera.backproject(u_idx, v_idx, z).astype(np.float32)
    radii = (0.5 * stride * z / camera.fy).astype(np.float32)
    return points, plate[v_idx, u_idx], radii, ground, alignment


def reconstruct_scene(
    video_path: Path,
    camera: PinholeCamera,
    players: Players,
    detections: Detections,
    estimator: DepthEstimator,
    stride: int = 2,
    samples: int = 24,
    extent: tuple[float, float, float, float] = DEFAULT_EXTENT,
    resolution: float = 0.025,
    log: Callable[[str], None] | None = None,
) -> Scene:
    log = log or (lambda _msg: None)
    total = players.frame_count
    sample_ids = set(np.linspace(0, total - 1, min(samples, total)).round().astype(int).tolist())
    frames, masks = [], []
    for index, rgb in iter_frames(video_path, stop=total):
        if index in sample_ids:
            frames.append(rgb)
            player_masks = _player_masks(detections, players, index)
            union = np.zeros(rgb.shape[:2], bool)
            for _, mask in player_masks.values():
                union |= mask
            masks.append(union)
    plate = median_plate(frames, masks)
    log(f"clean plate from {len(frames)} frames")

    ground_depth = ground_depth_map(camera)
    bg_points, bg_colors, bg_radii, ground_mask, alignment = background_cloud(
        plate, estimator(plate), camera, ground_depth, extent, stride=stride
    )
    ground = build_ground_texture(plate, camera, ground_mask, extent, resolution)
    log(f"background: {len(bg_points)} points, alignment residual {alignment.residual:.2e} 1/m")

    erode = np.ones((3, 3), np.uint8)
    grow = np.ones((9, 9), np.uint8)
    pts_list, col_list, rad_list, role_list = [], [], [], []
    offsets = [0]
    for index, rgb in iter_frames(video_path, stop=total):
        player_masks = _player_masks(detections, players, index)
        count = 0
        if player_masks:
            disparity = estimator(rgb)
            union = np.zeros(rgb.shape[:2], np.uint8)
            for _, mask in player_masks.values():
                union |= mask.astype(np.uint8)
            exclude = cv2.dilate(union, grow).astype(bool)
            depth = fit_disparity_alignment(disparity, ground_depth, court_fit_mask(camera, exclude=exclude)).depth(disparity)
            for role, (det, mask) in player_masks.items():
                inner = cv2.erode(mask.astype(np.uint8), erode).astype(bool)
                if inner.sum() < 10:
                    continue
                _, foot_depth = camera.ground_intersection(np.array([det.foot[0]]), np.array([det.foot[1]]))
                if not np.isfinite(foot_depth[0]):
                    continue
                values = anchor_player_depth(depth, inner, float(foot_depth[0]))
                v_idx, u_idx = np.nonzero(inner)
                pts_list.append(camera.backproject(u_idx, v_idx, values).astype(np.float32))
                col_list.append(rgb[v_idx, u_idx])
                rad_list.append((0.5 * values / camera.fy).astype(np.float32))
                role_list.append(np.full(len(values), ROLE_CODES[role], np.uint8))
                count += len(values)
        offsets.append(offsets[-1] + count)
        if index % 50 == 0:
            log(f"  players: frame {index}/{total}")
    while len(offsets) < total + 1:  # fewer decoded frames than expected: empty tail
        offsets.append(offsets[-1])

    def cat(parts: list[np.ndarray], shape: tuple[int, ...], dtype) -> np.ndarray:
        return np.concatenate(parts) if parts else np.zeros(shape, dtype)

    return Scene(
        background_points=bg_points,
        background_colors=bg_colors,
        background_radii=bg_radii,
        player_points=cat(pts_list, (0, 3), np.float32),
        player_colors=cat(col_list, (0, 3), np.uint8),
        player_radii=cat(rad_list, (0,), np.float32),
        player_roles=cat(role_list, (0,), np.uint8),
        frame_offsets=np.array(offsets, dtype=np.int64),
        ground=ground,
        camera=camera,
        fps=players.fps,
    )
```

- [ ] **Step 6: Implement `pipeline/horizon/commands/reconstruct.py`**

```python
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
```

- [ ] **Step 7: Run the tests and confirm they pass**

Run: `.venv\Scripts\python -m pytest -q`
Expected: `66 passed`.

- [ ] **Step 8: Commit**

```powershell
git add pipeline/horizon/ground.py pipeline/horizon/reconstruct.py pipeline/horizon/commands/reconstruct.py pipeline/tests/synthetic.py pipeline/tests/test_reconstruct.py
git commit -m "feat(pipeline): reconstruct ground texture, background and player point clouds"
```

### Task 11: Point-cloud renderer

**Files:**
- Create: `pipeline/horizon/render.py`
- Test: `pipeline/tests/test_render.py`

**Interfaces:**
- Consumes: `PinholeCamera` (Task 2), `pixel_grid` (Task 9), `GroundTexture` (Task 10), `Scene` (Task 10)
- Produces (`horizon.render`):
  - `SKY_TOP`, `SKY_HORIZON`
  - `sky_gradient(width, height) -> uint8 HxWx3`
  - `render_view(camera, ground: GroundTexture | None, points, colors, radii, max_splat=7, near=0.3) -> uint8 HxWx3`
  - `render_frame(scene, frame, camera, exclude_role=None, max_splat=7) -> uint8 HxWx3` (background + that frame's players, optionally without one player)

- [ ] **Step 1: Write the failing tests**

`pipeline/tests/test_render.py`:
```python
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
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.venv\Scripts\python -m pytest tests/test_render.py -q`
Expected: `ModuleNotFoundError: No module named 'horizon.render'`.

- [ ] **Step 3: Implement `pipeline/horizon/render.py`**

```python
"""Render the reconstructed scene from any pinhole camera.

Pass 1 intersects every pixel ray with the ground plane and samples the ground texture (sky gradient above the
horizon). Pass 2 projects points, splats each over ceil(2 * radius * f / z) pixels (clamped to 1..max_splat) and
keeps the nearest point per pixel that is in front of the ground.
"""

from __future__ import annotations

import numpy as np

from horizon.camera import PinholeCamera
from horizon.depth import pixel_grid
from horizon.ground import GroundTexture

SKY_TOP = (10, 12, 18)
SKY_HORIZON = (38, 44, 58)
GROUND_EPSILON = 0.05  # metres; points must be this much in front of the ground to win


def sky_gradient(width: int, height: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, height)[:, None, None]
    column = (1 - t) * np.array(SKY_TOP, float) + t * np.array(SKY_HORIZON, float)
    return np.broadcast_to(np.round(column).astype(np.uint8), (height, width, 3)).copy()


def render_view(
    camera: PinholeCamera,
    ground: GroundTexture | None,
    points: np.ndarray,
    colors: np.ndarray,
    radii: np.ndarray,
    max_splat: int = 7,
    near: float = 0.3,
) -> np.ndarray:
    width, height = camera.width, camera.height
    image = sky_gradient(width, height)
    flat = image.reshape(-1, 3)
    zbuf = np.full(width * height, np.inf)

    if ground is not None:
        us, vs = pixel_grid(width, height)
        hits, depth = camera.ground_intersection(us.ravel(), vs.ravel())
        valid = np.isfinite(depth) & (depth > near)
        flat[valid] = ground.sample(hits[valid, 0], hits[valid, 1])
        zbuf[valid] = depth[valid]

    if len(points):
        uv, z = camera.project(np.asarray(points, dtype=np.float64))
        ok = (z > near) & np.isfinite(uv).all(axis=1)
        uv, z = uv[ok], z[ok]
        colors, radii = np.asarray(colors)[ok], np.asarray(radii, dtype=np.float64)[ok]
        ui = np.rint(uv[:, 0]).astype(np.int64)
        vi = np.rint(uv[:, 1]).astype(np.int64)
        sizes = np.clip(np.ceil(2.0 * radii * camera.fy / z), 1, max_splat).astype(np.int64)
        lin_parts, src_parts = [], []
        for k in np.unique(sizes):
            sel = np.nonzero(sizes == k)[0]
            offsets = np.arange(k) - (k - 1) // 2
            dx, dy = np.meshgrid(offsets, offsets)
            uu = (ui[sel][:, None] + dx.ravel()[None, :]).ravel()
            vv = (vi[sel][:, None] + dy.ravel()[None, :]).ravel()
            src = np.repeat(sel, k * k)
            inside = (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height)
            lin_parts.append(vv[inside] * width + uu[inside])
            src_parts.append(src[inside])
        if lin_parts:
            lin = np.concatenate(lin_parts)
            src = np.concatenate(src_parts)
            zz = z[src]
            front = zz < zbuf[lin] - GROUND_EPSILON
            lin, src, zz = lin[front], src[front], zz[front]
            order = np.argsort(zz, kind="stable")
            lin, src = lin[order], src[order]
            unique, first = np.unique(lin, return_index=True)
            flat[unique] = colors[src[first]]
    return image


def render_frame(scene, frame: int, camera: PinholeCamera, exclude_role: str | None = None, max_splat: int = 7) -> np.ndarray:
    player_points, player_colors, player_radii = scene.players_at(frame, exclude_role=exclude_role)
    points = np.concatenate([scene.background_points, player_points])
    colors = np.concatenate([scene.background_colors, player_colors])
    radii = np.concatenate([scene.background_radii, player_radii])
    return render_view(camera, scene.ground, points, colors, radii, max_splat=max_splat)
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `.venv\Scripts\python -m pytest -q`
Expected: `71 passed`.

- [ ] **Step 5: Commit**

```powershell
git add pipeline/horizon/render.py pipeline/tests/test_render.py
git commit -m "feat(pipeline): render scenes with a textured ground plane and z-buffered point splats"
```

### Task 12: Player POV cameras and clips (`horizon render`)

**Files:**
- Create: `pipeline/horizon/pov.py`, `pipeline/horizon/commands/render.py`
- Test: `pipeline/tests/test_pov.py`

**Interfaces:**
- Consumes: `Players`, `PlayerTrack` (Task 8), `Scene` (Task 10), `render_frame`, `sky_gradient` (Task 11), `H264Writer`, `iter_frames`, `probe` (Task 3), `PinholeCamera.look_at`, `vertical_fov_from_horizontal` (Task 2)
- Produces (`horizon.pov`):
  - `POV_WIDTH = 1024`, `POV_HEIGHT = 576`, `POV_HFOV_DEG = 75.0`, `TARGET_HEIGHT = 1.0`
  - `pov_camera(players, role, frame, hfov_deg=POV_HFOV_DEG, width=POV_WIDTH, height=POV_HEIGHT) -> PinholeCamera`
  - `render_clip(scene, camera_for_frame: Callable[[int], PinholeCamera], out_path, exclude_role=None, frame_count=None, log=None) -> int` (frames written)
  - `render_pov_clip(scene, players, role, out_path, hfov_deg=POV_HFOV_DEG, width=POV_WIDTH, height=POV_HEIGHT, log=None) -> int`
- CLI: `horizon render --match-id <id> [--roles near far] [--width 1024] [--height 576] [--fov 75]` writes `pov_near.mp4` and `pov_far.mp4`

- [ ] **Step 1: Write the failing tests**

`pipeline/tests/test_pov.py`:
```python
import numpy as np
import pytest
from synthetic import broadcast_camera

from horizon.ground import GroundTexture
from horizon.players import EYE_HEIGHT_RATIO, Players, PlayerTrack
from horizon.pov import pov_camera, render_pov_clip
from horizon.reconstruct import Scene
from horizon.render import sky_gradient
from horizon.video import iter_frames, probe


def static_players(near_xy, far_xy, frames=3, stature=1.85) -> Players:
    def track(role, xy):
        foot = np.tile(np.array(xy, dtype=float), (frames, 1))
        return PlayerTrack(role, role.title(), "", [1] * frames, [(0.0, 0.0, 1.0, 1.0)] * frames, foot, stature, np.zeros(frames), np.zeros(frames))

    return Players(25.0, frames, {"near": track("near", near_xy), "far": track("far", far_xy)})


def empty_scene(frames: int) -> Scene:
    ground = np.zeros((400, 200, 3), np.uint8)  # 0.1 m texels over 20 m x 40 m
    ground[:] = (40, 120, 70)
    return Scene(
        background_points=np.zeros((0, 3), np.float32),
        background_colors=np.zeros((0, 3), np.uint8),
        background_radii=np.zeros(0, np.float32),
        player_points=np.zeros((0, 3), np.float32),
        player_colors=np.zeros((0, 3), np.uint8),
        player_radii=np.zeros(0, np.float32),
        player_roles=np.zeros(0, np.uint8),
        frame_offsets=np.zeros(frames + 1, np.int64),
        ground=GroundTexture(ground, -10.0, 10.0, -20.0, 20.0, (40, 120, 70)),
        camera=broadcast_camera(),
        fps=25.0,
    )


def test_pov_camera_sits_at_the_eye_and_faces_the_opponent():
    players = static_players((1.0, -10.0), (-2.0, 10.0))
    cam = pov_camera(players, "near", 1)
    eye = np.array([1.0, -10.0, EYE_HEIGHT_RATIO * 1.85])
    target = np.array([-2.0, 10.0, 1.0])
    assert np.allclose(cam.center, eye)
    assert np.allclose(cam.forward, (target - eye) / np.linalg.norm(target - eye))
    uv, _ = cam.project(target[None, :])
    assert uv[0] == pytest.approx([cam.cx, cam.cy])
    assert (cam.width, cam.height) == (1024, 576)
    assert cam.fx == pytest.approx(512 / np.tan(np.radians(37.5)), rel=1e-6)


def test_pov_camera_faces_the_net_when_players_overlap():
    players = static_players((0.0, 3.0), (0.0, 3.2))
    assert pov_camera(players, "near", 0).forward[1] > 0.99
    assert pov_camera(players, "far", 0).forward[1] < -0.99


def test_render_pov_clip_writes_a_frame_aligned_video(tmp_path):
    players = static_players((1.0, -10.0), (-2.0, 10.0), frames=4)
    out = tmp_path / "pov_near.mp4"
    written = render_pov_clip(empty_scene(4), players, "near", out, width=160, height=96)
    assert written == 4
    info = probe(out)
    assert (info.width, info.height) == (160, 96) and info.fps == pytest.approx(25.0)
    frames = [f for _, f in iter_frames(out)]
    assert len(frames) == 4
    sky = sky_gradient(160, 96)
    assert np.abs(frames[0][-10:].astype(int) - sky[-10:].astype(int)).mean() > 20  # the ground fills the bottom
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.venv\Scripts\python -m pytest tests/test_pov.py -q`
Expected: `ModuleNotFoundError: No module named 'horizon.pov'`.

- [ ] **Step 3: Implement `pipeline/horizon/pov.py`**

```python
"""First-person cameras for both players and clip rendering."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from horizon.camera import PinholeCamera, vertical_fov_from_horizontal
from horizon.players import Players
from horizon.render import render_frame
from horizon.video import H264Writer

POV_WIDTH, POV_HEIGHT = 1024, 576
POV_HFOV_DEG = 75.0
TARGET_HEIGHT = 1.0  # look at the opponent's chest


def pov_camera(
    players: Players, role: str, frame: int, hfov_deg: float = POV_HFOV_DEG, width: int = POV_WIDTH, height: int = POV_HEIGHT
) -> PinholeCamera:
    eye = players.tracks[role].eye(frame)
    ox, oy = players.opponent(role).foot_xy[frame]
    target = np.array([ox, oy, TARGET_HEIGHT])
    if np.hypot(target[0] - eye[0], target[1] - eye[1]) < 0.5:
        target = eye + np.array([0.0, 1.0 if role == "near" else -1.0, 0.0])  # face the net
    return PinholeCamera.look_at(eye, target, vertical_fov_from_horizontal(hfov_deg, width, height), width, height)


def render_clip(
    scene,
    camera_for_frame: Callable[[int], PinholeCamera],
    out_path: Path,
    exclude_role: str | None = None,
    frame_count: int | None = None,
    log: Callable[[str], None] | None = None,
) -> int:
    total = scene.frame_count if frame_count is None else min(frame_count, scene.frame_count)
    first = camera_for_frame(0)
    with H264Writer(out_path, first.width, first.height, scene.fps) as writer:
        for frame in range(total):
            camera = first if frame == 0 else camera_for_frame(frame)
            writer.write(render_frame(scene, frame, camera, exclude_role=exclude_role))
            if log and frame % 50 == 0:
                log(f"  {Path(out_path).name}: frame {frame}/{total}")
        return writer.frames_written


def render_pov_clip(
    scene,
    players: Players,
    role: str,
    out_path: Path,
    hfov_deg: float = POV_HFOV_DEG,
    width: int = POV_WIDTH,
    height: int = POV_HEIGHT,
    log: Callable[[str], None] | None = None,
) -> int:
    return render_clip(
        scene,
        lambda frame: pov_camera(players, role, frame, hfov_deg, width, height),
        out_path,
        exclude_role=role,
        frame_count=players.frame_count,
        log=log,
    )
```

- [ ] **Step 4: Implement `pipeline/horizon/commands/render.py`**

```python
"""`horizon render`: first-person POV videos for the players."""

from __future__ import annotations

import argparse

from horizon.paths import MatchPaths
from horizon.players import Players
from horizon.pov import POV_HEIGHT, POV_HFOV_DEG, POV_WIDTH, render_pov_clip
from horizon.reconstruct import Scene


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("render", help="Render each player's first-person POV video")
    p.add_argument("--match-id", required=True)
    p.add_argument("--roles", nargs="+", choices=("near", "far"), default=["near", "far"])
    p.add_argument("--width", type=int, default=POV_WIDTH)
    p.add_argument("--height", type=int, default=POV_HEIGHT)
    p.add_argument("--fov", type=float, default=POV_HFOV_DEG, help="Horizontal field of view in degrees")
    p.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    paths = MatchPaths.for_match(args.match_id)
    scene = Scene.load(paths.scene)
    players = Players.load(paths.players)
    for role in args.roles:
        out = paths.pov_video(role)
        count = render_pov_clip(scene, players, role, out, args.fov, args.width, args.height, log=print)
        print(f"Wrote {out} ({count} frames)")
    return 0
```

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `.venv\Scripts\python -m pytest -q`
Expected: `74 passed`.

- [ ] **Step 6: Commit**

```powershell
git add pipeline/horizon/pov.py pipeline/horizon/commands/render.py pipeline/tests/test_pov.py
git commit -m "feat(pipeline): render first-person POV clips for both players"
```

### Task 13: Export to the web app (`horizon export`) and `horizon all`

**Files:**
- Create: `pipeline/horizon/export.py`, `pipeline/horizon/commands/export.py`, `pipeline/horizon/commands/all.py`
- Test: `pipeline/tests/test_export.py`

**Interfaces:**
- Consumes: `MatchPaths`, `REPO_ROOT` (Task 1), `load_meta`, `save_meta`, `VideoInfo` (Task 3), `load_calibration`, `save_calibration`, `Calibration` (Task 4), `Identity` (Task 7), `Players`, `PlayerTrack` (Task 8), `tests/synthetic.py` (`broadcast_camera`, `make_identity`)
- Produces (`horizon.export`):
  - `PUBLIC_MATCHES = REPO_ROOT / "public" / "matches"`
  - `PLAYER_COLORS = {"near": "#3B82F6", "far": "#F97316"}`
  - `VIEWER_URL = "http://localhost:8080"`
  - `head_anchor_percent(camera, players, role, frame) -> (x%, y%)`
  - `build_tracks_json(players, camera) -> dict` (spec §6 `tracks.json`)
  - `build_manifest(match_id, info, identity, players, has_free_cam, competition="Tennis", viewer_url=VIEWER_URL) -> dict` (spec §6 `manifest.json`)
  - `update_index(public_matches) -> dict`
  - `export_match(paths, public_matches=PUBLIC_MATCHES, competition="Tennis", viewer_url=VIEWER_URL) -> Path`
- CLI:
  - `horizon export --match-id <id> [--competition Tennis] [--viewer-url http://localhost:8080] [--public <dir>]`
  - `horizon all <video> --match-id <id> [--start S] [--duration S] [--orchestrator backboard|direct|none] [--recalibrate]` runs init → calibrate (skipped if `calibration.json` exists) → track → identify → players → reconstruct → render → export

- [ ] **Step 1: Write the failing tests**

`pipeline/tests/test_export.py`:
```python
import json

import numpy as np
import pytest
from synthetic import H, W, broadcast_camera, make_identity

from horizon.calibration import Calibration, save_calibration
from horizon.export import build_manifest, build_tracks_json, export_match, head_anchor_percent
from horizon.paths import MatchPaths
from horizon.players import Players, PlayerTrack
from horizon.video import VideoInfo, save_meta


def two_frame_players() -> Players:
    def track(role, xy, bbox):
        foot = np.tile(np.array(xy, dtype=float), (2, 1))
        return PlayerTrack(role, role.title(), "", [1, None], [bbox, None], foot, 1.85, np.array([3.0, 4.0]), np.array([0.0, 0.1]))

    return Players(
        25.0,
        2,
        {
            "near": track("near", (1.0, -10.0), (600.0, 300.0, 640.0, 420.0)),
            "far": track("far", (-2.0, 10.0), (500.0, 100.0, 520.0, 150.0)),
        },
    )


def write_match(root, match_id="demo", with_pov=True) -> MatchPaths:
    paths = MatchPaths.for_match(match_id, data_root=root).ensure()
    save_meta(paths.meta, VideoInfo(W, H, 25.0, 2))
    make_identity().save(paths.identity)
    two_frame_players().save(paths.players)
    save_calibration(Calibration(camera=broadcast_camera(), keypoints={}, rms_px=0.5), paths.calibration)
    paths.source_video.write_bytes(b"main")
    if with_pov:
        paths.pov_video("near").write_bytes(b"near")
        paths.pov_video("far").write_bytes(b"far")
    return paths


def test_head_anchor_uses_box_top_and_projects_missing_frames():
    cam, players = broadcast_camera(), two_frame_players()
    assert head_anchor_percent(cam, players, "near", 0) == (pytest.approx(100 * 620 / W, abs=0.01), pytest.approx(100 * 300 / H, abs=0.01))
    uv, _ = cam.project(np.array([[1.0, -10.0, 1.85]]))
    x, y = head_anchor_percent(cam, players, "near", 1)
    assert x == pytest.approx(100 * uv[0, 0] / W, abs=0.01) and y == pytest.approx(100 * uv[0, 1] / H, abs=0.01)


def test_tracks_and_manifest_follow_the_web_contract():
    players = two_frame_players()
    tracks = build_tracks_json(players, broadcast_camera())
    assert tracks["fps"] == 25.0 and tracks["frameCount"] == 2
    assert set(tracks["players"]["near"]) == {"x", "y", "visible", "speedKmh", "distanceM"}
    assert tracks["players"]["far"]["visible"] == [1, 0]
    manifest = build_manifest("demo", VideoInfo(W, H, 25.0, 2), make_identity(), players, has_free_cam=False)
    assert set(manifest) == {
        "id", "title", "competition", "summary", "score", "fps", "frameCount", "width", "height",
        "video", "tracks", "players", "freeCam", "viewerUrl",
    }
    assert manifest["title"] == "Near Player vs Far Player"
    assert [p["id"] for p in manifest["players"]] == ["near", "far"]
    assert manifest["players"][0] == {
        "id": "near", "name": "Near Player", "description": "white shirt", "color": "#3B82F6",
        "pov": "pov_near.mp4", "statureM": 1.85,
    }
    assert manifest["freeCam"] is None


def test_export_match_copies_media_and_rebuilds_the_index(tmp_path):
    public = tmp_path / "public" / "matches"
    export_match(write_match(tmp_path / "data", "demo"), public_matches=public)
    second = write_match(tmp_path / "data", "another")
    second.free_cam_video.write_bytes(b"free")
    export_match(second, public_matches=public)
    demo = public / "demo"
    assert (demo / "main.mp4").read_bytes() == b"main"
    assert (demo / "pov_far.mp4").read_bytes() == b"far"
    assert json.loads((demo / "manifest.json").read_text())["freeCam"] is None
    assert json.loads((public / "another" / "manifest.json").read_text())["freeCam"] == "free_cam.mp4"
    assert json.loads((demo / "tracks.json").read_text())["frameCount"] == 2
    index = json.loads((public / "index.json").read_text())
    assert [m["id"] for m in index["matches"]] == ["another", "demo"]
    assert index["matches"][1] == {
        "id": "demo", "title": "Near Player vs Far Player", "near": "Near Player", "far": "Far Player",
        "competition": "Tennis", "status": "replay",
    }


def test_export_reports_missing_videos(tmp_path):
    with pytest.raises(FileNotFoundError, match="pov_near.mp4"):
        export_match(write_match(tmp_path / "data", with_pov=False), public_matches=tmp_path / "public")
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.venv\Scripts\python -m pytest tests/test_export.py -q`
Expected: `ModuleNotFoundError: No module named 'horizon.export'`.

- [ ] **Step 3: Implement `pipeline/horizon/export.py`**

```python
"""Publish a processed match to the React app: public/matches/<id>/ and public/matches/index.json (spec §6)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from horizon.calibration import load_calibration
from horizon.camera import PinholeCamera
from horizon.identify import Identity
from horizon.paths import REPO_ROOT, MatchPaths
from horizon.players import Players
from horizon.video import VideoInfo, load_meta

PUBLIC_MATCHES = REPO_ROOT / "public" / "matches"
PLAYER_COLORS = {"near": "#3B82F6", "far": "#F97316"}
VIEWER_URL = "http://localhost:8080"


def head_anchor_percent(camera: PinholeCamera, players: Players, role: str, frame: int) -> tuple[float, float]:
    """Top-centre of the player's box in % of the frame; projected from the trajectory when undetected."""
    track = players.tracks[role]
    bbox = track.bboxes[frame]
    if bbox is not None:
        x, y = (bbox[0] + bbox[2]) / 2.0, bbox[1]
    else:
        fx, fy = track.foot_xy[frame]
        uv, _ = camera.project(np.array([[fx, fy, track.stature_m]]))
        x, y = float(uv[0, 0]), float(uv[0, 1])
    return round(100.0 * x / camera.width, 2), round(100.0 * y / camera.height, 2)


def build_tracks_json(players: Players, camera: PinholeCamera) -> dict:
    result: dict = {"fps": players.fps, "frameCount": players.frame_count, "players": {}}
    for role, track in players.tracks.items():
        anchors = [head_anchor_percent(camera, players, role, frame) for frame in range(players.frame_count)]
        result["players"][role] = {
            "x": [a[0] for a in anchors],
            "y": [a[1] for a in anchors],
            "visible": [1 if v else 0 for v in track.visible],
            "speedKmh": np.round(track.speed_kmh, 1).tolist(),
            "distanceM": np.round(track.distance_m, 1).tolist(),
        }
    return result


def build_manifest(
    match_id: str,
    info: VideoInfo,
    identity: Identity,
    players: Players,
    has_free_cam: bool,
    competition: str = "Tennis",
    viewer_url: str = VIEWER_URL,
) -> dict:
    near, far = identity.players["near"], identity.players["far"]
    return {
        "id": match_id,
        "title": f"{near.name} vs {far.name}",
        "competition": competition,
        "summary": identity.summary,
        "score": identity.score,
        "fps": info.fps,
        "frameCount": info.frame_count,
        "width": info.width,
        "height": info.height,
        "video": "main.mp4",
        "tracks": "tracks.json",
        "players": [
            {
                "id": role,
                "name": identity.players[role].name,
                "description": identity.players[role].description,
                "color": PLAYER_COLORS[role],
                "pov": f"pov_{role}.mp4",
                "statureM": round(players.tracks[role].stature_m, 2),
            }
            for role in ("near", "far")
        ],
        "freeCam": "free_cam.mp4" if has_free_cam else None,
        "viewerUrl": viewer_url,
    }


def update_index(public_matches: Path) -> dict:
    matches = []
    for manifest_path in sorted(Path(public_matches).glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        matches.append(
            {
                "id": manifest["id"],
                "title": manifest["title"],
                "near": manifest["players"][0]["name"],
                "far": manifest["players"][1]["name"],
                "competition": manifest["competition"],
                "status": "replay",
            }
        )
    index = {"matches": matches}
    (Path(public_matches) / "index.json").write_text(json.dumps(index, indent=2))
    return index


def export_match(
    paths: MatchPaths, public_matches: Path = PUBLIC_MATCHES, competition: str = "Tennis", viewer_url: str = VIEWER_URL
) -> Path:
    required = [paths.source_video, paths.pov_video("near"), paths.pov_video("far")]
    missing = [p.name for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {', '.join(missing)} in {paths.root}; run the earlier pipeline steps")
    info = load_meta(paths.meta)
    identity = Identity.load(paths.identity)
    players = Players.load(paths.players)
    camera = load_calibration(paths.calibration).camera
    target = Path(public_matches) / paths.match_id
    target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(paths.source_video, target / "main.mp4")
    for role in ("near", "far"):
        shutil.copyfile(paths.pov_video(role), target / f"pov_{role}.mp4")
    has_free_cam = paths.free_cam_video.is_file()
    if has_free_cam:
        shutil.copyfile(paths.free_cam_video, target / "free_cam.mp4")
    (target / "tracks.json").write_text(json.dumps(build_tracks_json(players, camera)))
    manifest = build_manifest(paths.match_id, info, identity, players, has_free_cam, competition, viewer_url)
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2))
    update_index(public_matches)
    return target
```

- [ ] **Step 4: Implement `pipeline/horizon/commands/export.py`**

```python
"""`horizon export`: publish a processed match into public/matches for the web app."""

from __future__ import annotations

import argparse
from pathlib import Path

from horizon.export import PUBLIC_MATCHES, VIEWER_URL, export_match
from horizon.paths import MatchPaths


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("export", help="Copy videos + manifest + tracks into public/matches/<id>")
    p.add_argument("--match-id", required=True)
    p.add_argument("--competition", default="Tennis")
    p.add_argument("--viewer-url", default=VIEWER_URL, help="Where `horizon view` serves the 3D free camera")
    p.add_argument("--public", type=Path, default=PUBLIC_MATCHES, help="public/matches directory of the web app")
    p.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    target = export_match(MatchPaths.for_match(args.match_id), args.public, args.competition, args.viewer_url)
    print(f"Exported to {target}; open http://localhost:5173 after `npm run dev`")
    return 0
```

- [ ] **Step 5: Implement `pipeline/horizon/commands/all.py`**

```python
"""`horizon all`: run every pipeline step for one clip."""

from __future__ import annotations

import argparse
from pathlib import Path

from horizon.paths import MatchPaths


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("all", help="init -> calibrate -> track -> identify -> players -> reconstruct -> render -> export")
    p.add_argument("video", type=Path)
    p.add_argument("--match-id", required=True)
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("--orchestrator", choices=("backboard", "direct", "none"), default="backboard")
    p.add_argument("--recalibrate", action="store_true", help="Click the court keypoints again")
    p.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    from horizon.cli import main as cli_main

    match = ["--match-id", args.match_id]
    init = ["init", str(args.video), *match, "--start", str(args.start)]
    if args.duration is not None:
        init += ["--duration", str(args.duration)]
    steps = [init]
    if args.recalibrate or not MatchPaths.for_match(args.match_id).calibration.is_file():
        steps.append(["calibrate", *match])
    steps += [
        ["track", *match],
        ["identify", *match, "--orchestrator", args.orchestrator],
        ["players", *match],
        ["reconstruct", *match],
        ["render", *match],
        ["export", *match],
    ]
    for step in steps:
        print(f"== horizon {step[0]}")
        code = cli_main(step)
        if code != 0:
            print(f"Stopped: `horizon {step[0]}` exited with {code}")
            return code
    return 0
```

- [ ] **Step 6: Run the tests and confirm they pass**

Run: `.venv\Scripts\python -m pytest -q`
Expected: `78 passed`.

- [ ] **Step 7: Check the CLI lists every stage**

Run: `.venv\Scripts\python -m horizon --help`
Expected: the sub-command list contains `all, calibrate, doctor, export, identify, init, players, reconstruct, render, track`.

- [ ] **Step 8: Commit**

```powershell
git add pipeline/horizon/export.py pipeline/horizon/commands/export.py pipeline/horizon/commands/all.py pipeline/tests/test_export.py
git commit -m "feat(pipeline): export matches for the web app and add horizon all"
```

## Phase B — Free-camera viewer

### Task 14: Viser free-camera viewer (`horizon view`)

**Files:**
- Create: `pipeline/horizon/viewer.py`, `pipeline/horizon/commands/view.py`
- Test: `pipeline/tests/test_viewer.py`

**Interfaces:**
- Consumes:
  - `camera_pose`, `camera_from_pose`, `vertical_fov_from_horizontal`, `PinholeCamera` (Task 2)
  - `Players` (Task 8)
  - `Scene`, `ROLE_CODES` (Task 10)
  - `render_frame` (Task 11)
  - `pov_camera`, `render_clip`, `POV_WIDTH`, `POV_HEIGHT`, `POV_HFOV_DEG` (Task 12)
  - `PLAYER_COLORS` (Task 13)
  - `GroundTexture` (Task 10)
  - `MatchPaths` (Task 1)
- Produces (`horizon.viewer`):
  - constants: `FOLLOW_OPTIONS = ("off", "near eyes", "near chase", "far eyes", "far chase")`, `CHASE_DISTANCE = 2.5`, `CHASE_HEIGHT = 0.8`
  - `free_camera(wxyz, position, hfov_deg, width, height) -> PinholeCamera`
  - `look_target(camera, distance=5.0) -> np.ndarray(3)`
  - `snap_pose(players, role, frame, hfov_deg=POV_HFOV_DEG) -> (wxyz, position)`
  - `parse_follow(option) -> (role, style) | None`
  - `follow_camera(players, role, style, frame, hfov_deg, width, height) -> PinholeCamera`
  - `export_camera(option, players, frame, static_pose, hfov_deg, width, height) -> PinholeCamera`
  - `excluded_role(option) -> str | None` (only the "eyes" styles hide the viewer's own body)
  - `downsample(points, colors, max_points, seed=0) -> (points, colors)`
  - `ground_cloud(texture, spacing=0.08) -> (points, colors)`
  - `run_viewer(paths, host="0.0.0.0", port=8080) -> None` (blocking; imports viser lazily)
- CLI: `horizon view --match-id <id> [--host 0.0.0.0] [--port 8080]`. The free-cam clip is written to `data/<id>/free_cam.mp4`, and `horizon export` publishes it.

- [ ] **Step 1: Write the failing tests**

`pipeline/tests/test_viewer.py`:
```python
import numpy as np
import pytest

from horizon.ground import GroundTexture
from horizon.players import Players, PlayerTrack
from horizon.pov import pov_camera
from horizon.viewer import (
    CHASE_HEIGHT,
    downsample,
    excluded_role,
    export_camera,
    free_camera,
    ground_cloud,
    look_target,
    parse_follow,
    snap_pose,
)


def moving_players(frames: int = 5) -> Players:
    def track(role, start, step):
        foot = np.array([[start[0] + step * i, start[1]] for i in range(frames)], dtype=float)
        return PlayerTrack(role, role, "", [1] * frames, [None] * frames, foot, 1.85, np.zeros(frames), np.zeros(frames))

    return Players(25.0, frames, {"near": track("near", (0.0, -10.0), 0.2), "far": track("far", (1.0, 11.0), -0.1)})


def test_snap_pose_matches_the_player_pov():
    players = moving_players()
    wxyz, position = snap_pose(players, "near", 3, 75.0)
    cam = free_camera(wxyz, position, 75.0, 1024, 576)
    pov = pov_camera(players, "near", 3, 75.0, 1024, 576)
    assert np.allclose(cam.center, pov.center)
    target = np.array([[1.0 - 0.3, 11.0, 1.0]])
    assert np.allclose(cam.project(target)[0], pov.project(target)[0], atol=1e-6)


def test_look_target_is_straight_ahead():
    cam = pov_camera(moving_players(), "far", 0)
    assert np.allclose(look_target(cam, 5.0) - cam.center, 5.0 * cam.forward)


def test_follow_options():
    assert parse_follow("off") is None
    assert parse_follow("near chase") == ("near", "chase")
    assert excluded_role("far eyes") == "far"
    assert excluded_role("far chase") is None and excluded_role("off") is None
    with pytest.raises(ValueError):
        parse_follow("sideways")


def test_export_camera_follows_players_or_stays_static():
    players = moving_players()
    eyes = [export_camera("near eyes", players, f, None, 75.0, 320, 180).center for f in range(5)]
    assert np.allclose([c[0] for c in eyes], [0.0, 0.2, 0.4, 0.6, 0.8])
    chase = export_camera("far chase", players, 2, None, 75.0, 320, 180)
    far_eye = players.tracks["far"].eye(2)
    assert chase.center[1] > far_eye[1] + 2.0  # behind the far player (further from the net)
    assert chase.center[2] == pytest.approx(far_eye[2] + CHASE_HEIGHT)
    assert chase.forward[1] < 0  # looking back towards the near end
    static_pose = snap_pose(players, "near", 0, 60.0)
    frames = [export_camera("off", players, f, static_pose, 60.0, 320, 180).center for f in range(5)]
    assert np.allclose(frames, frames[0])


def test_downsample_and_ground_cloud():
    points = np.random.default_rng(1).normal(size=(10_000, 3)).astype(np.float32)
    colors = np.zeros((10_000, 3), np.uint8)
    small, small_colors = downsample(points, colors, 1000)
    again, _ = downsample(points, colors, 1000)
    assert small.shape == (1000, 3) and small_colors.shape == (1000, 3)
    assert np.array_equal(small, again)
    untouched, _ = downsample(points[:10], colors[:10], 1000)
    assert np.array_equal(untouched, points[:10])
    texture = GroundTexture(np.zeros((1600, 800, 3), np.uint8), -10.0, 10.0, -20.0, 20.0, (0, 0, 0))
    ground_points, ground_colors = ground_cloud(texture, spacing=0.1)
    assert len(ground_points) == 400 * 200 == len(ground_colors)
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.venv\Scripts\python -m pytest tests/test_viewer.py -q`
Expected: `ModuleNotFoundError: No module named 'horizon.viewer'`.

- [ ] **Step 3: Implement `pipeline/horizon/viewer.py`**

```python
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
        option = follow.value
        static_pose = (np.array(gizmo.wxyz), np.array(gizmo.position))
        try:
            count = render_clip(
                scene,
                lambda f: export_camera(option, players, f, static_pose, fov.value, POV_WIDTH, POV_HEIGHT),
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

    show_frame(0)
    print(f"Viewer running at http://localhost:{port} (Ctrl+C to stop)")
    while True:
        if playing.value and not busy["export"] and total > 1:
            frame_slider.value = (int(frame_slider.value) + 1) % total
        time.sleep(1.0 / max(float(fps_slider.value), 1.0))
```

- [ ] **Step 4: Implement `pipeline/horizon/commands/view.py`**

```python
"""`horizon view`: interactive 3D viewer with the free camera (Viser)."""

from __future__ import annotations

import argparse

from horizon.paths import MatchPaths


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("view", help="Open the 3D free-camera viewer (Viser)")
    p.add_argument("--match-id", required=True)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    p.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    from horizon.viewer import run_viewer

    run_viewer(MatchPaths.for_match(args.match_id), host=args.host, port=args.port)
    return 0
```

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `.venv\Scripts\python -m pytest -q`
Expected: `83 passed`.

- [ ] **Step 6: Check the command is registered**

Run: `.venv\Scripts\python -m horizon view --help`
Expected: usage text with `--match-id`, `--host` and `--port`. The interactive check of the viewer happens in Task 17, once a real clip has been processed.

- [ ] **Step 7: Commit**

```powershell
git add pipeline/horizon/viewer.py pipeline/horizon/commands/view.py pipeline/tests/test_viewer.py
git commit -m "feat(viewer): Viser free camera with snap, follow, look-through, preview and clip export"
```

## Phase C — Web app

### Task 15: Web test harness and match data layer

**Files:**
- Modify: `package.json` (scripts + dev dependencies), `vite.config.ts`
- Create: `src/test/setup.ts`, `src/test/fixtures.ts`, `src/lib/match.ts`, `src/hooks/useMatchData.ts`
- Test: `src/lib/match.test.ts`, `src/hooks/useMatchData.test.tsx`

**Interfaces:**
- Consumes: the JSON contracts in spec §6, produced by `horizon export` (Task 13)
- Produces (`src/lib/match.ts`):
  - types `PlayerRole`, `MatchSummary`, `MatchIndex`, `PlayerInfo`, `MatchManifest`, `PlayerTrackSeries`, `TracksFile`, `PlayerSnapshot`, `Size`, `Point`
  - `MATCHES_ROOT = '/matches'`
  - `assetUrl(matchId, file)`
  - `frameIndexAt(time, fps, frameCount)`
  - `playerSnapshot(tracks, role, frame)`
  - `coverTransform(video, container)`
  - `videoPercentToScreen(xPercent, yPercent, video, container)`
  - `cardPosition(anchor, card, container, gap = 12)`
  - `needsResync(povTime, mainTime, tolerance = 0.15)`
- Produces (`src/hooks/useMatchData.ts`):
  - `useMatchData(matchId) -> { manifest, tracks, error }`
  - `useMatchIndex() -> { matches, error, loading }`
- Produces (`src/test/fixtures.ts`): `sampleManifest`, `sampleTracks`, `sampleIndex`, `mockFetch(routes: Record<string, unknown>)`

- [ ] **Step 1: Install the test tooling**

Run (repo root):
```powershell
npm install
npm install -D vitest jsdom @testing-library/react @testing-library/jest-dom
```
Expected: installs without peer-dependency errors (Vitest 4 supports Vite 7; Testing Library 16 supports React 19).

- [ ] **Step 2: Add the scripts and Vitest config**

In `package.json`, set `"scripts"` to:
```json
  "scripts": {
    "dev": "vite",
    "build": "tsc -b && vite build",
    "lint": "eslint .",
    "preview": "vite preview",
    "test": "vitest run",
    "test:watch": "vitest"
  },
```

Replace `vite.config.ts` with:
```ts
/// <reference types="vitest/config" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    css: false,
  },
})
```

`src/test/setup.ts`:
```ts
import '@testing-library/jest-dom/vitest';
import { cleanup } from '@testing-library/react';
import { afterEach, vi } from 'vitest';

// jsdom does not implement media playback.
Object.defineProperty(HTMLMediaElement.prototype, 'play', {
  configurable: true,
  value: vi.fn(() => Promise.resolve()),
});
Object.defineProperty(HTMLMediaElement.prototype, 'pause', {
  configurable: true,
  value: vi.fn(),
});

// jsdom has no matchMedia; AccessibilityProvider reads prefers-contrast / prefers-reduced-motion.
if (!window.matchMedia) {
  window.matchMedia = (query: string) =>
    ({
      matches: false,
      media: query,
      onchange: null,
      addListener: () => {},
      removeListener: () => {},
      addEventListener: () => {},
      removeEventListener: () => {},
      dispatchEvent: () => false,
    }) as MediaQueryList;
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});
```

`src/test/fixtures.ts`:
```ts
import { vi } from 'vitest';
import type { MatchIndex, MatchManifest, TracksFile } from '../lib/match';

export const sampleManifest: MatchManifest = {
  id: 'demo',
  title: 'Carlos Alcaraz vs Jannik Sinner',
  competition: 'Tennis',
  summary: 'Long baseline rally won by Alcaraz.',
  score: '6-4 2-1 30-15',
  fps: 25,
  frameCount: 4,
  width: 1280,
  height: 720,
  video: 'main.mp4',
  tracks: 'tracks.json',
  players: [
    { id: 'near', name: 'Carlos Alcaraz', description: 'white shirt', color: '#3B82F6', pov: 'pov_near.mp4', statureM: 1.83 },
    { id: 'far', name: 'Jannik Sinner', description: 'navy shirt', color: '#F97316', pov: 'pov_far.mp4', statureM: 1.91 },
  ],
  freeCam: null,
  viewerUrl: 'http://localhost:8080',
};

export const sampleTracks: TracksFile = {
  fps: 25,
  frameCount: 4,
  players: {
    near: { x: [50, 51, 52, 53], y: [60, 60, 61, 61], visible: [1, 1, 0, 1], speedKmh: [10, 14, 12, 8], distanceM: [0, 0.1, 0.3, 0.4] },
    far: { x: [45, 45, 44, 44], y: [20, 20, 21, 21], visible: [1, 1, 1, 1], speedKmh: [5, 6, 7, 9], distanceM: [0, 0.1, 0.2, 0.3] },
  },
};

export const sampleIndex: MatchIndex = {
  matches: [
    { id: 'demo', title: sampleManifest.title, near: 'Carlos Alcaraz', far: 'Jannik Sinner', competition: 'Tennis', status: 'replay' },
  ],
};

/** Stub global fetch: URLs in `routes` answer 200 with the given JSON body, everything else 404. */
export function mockFetch(routes: Record<string, unknown>) {
  const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url in routes) {
      return { ok: true, status: 200, json: async () => routes[url] } as unknown as Response;
    }
    return { ok: false, status: 404, json: async () => ({}) } as unknown as Response;
  });
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}
```

- [ ] **Step 3: Write the failing tests**

`src/lib/match.test.ts`:
```ts
import { describe, expect, it } from 'vitest';
import { sampleTracks } from '../test/fixtures';
import {
  assetUrl,
  cardPosition,
  coverTransform,
  frameIndexAt,
  needsResync,
  playerSnapshot,
  videoPercentToScreen,
} from './match';

describe('frameIndexAt', () => {
  it('maps time to a clamped frame index', () => {
    expect(frameIndexAt(0, 25, 250)).toBe(0);
    expect(frameIndexAt(1, 25, 250)).toBe(25);
    expect(frameIndexAt(0.12, 25, 250)).toBe(3);
    expect(frameIndexAt(100, 25, 250)).toBe(249);
    expect(frameIndexAt(-1, 25, 250)).toBe(0);
    expect(frameIndexAt(Number.NaN, 25, 250)).toBe(0);
  });
});

describe('playerSnapshot', () => {
  it('reads one frame and tracks the top speed so far', () => {
    expect(playerSnapshot(sampleTracks, 'near', 2)).toEqual({
      x: 52, y: 61, visible: false, speedKmh: 12, distanceM: 0.3, topSpeedKmh: 14,
    });
    expect(playerSnapshot(sampleTracks, 'far', 99).x).toBe(44);
  });
});

describe('cover geometry', () => {
  it('matches object-fit: cover', () => {
    const t = coverTransform({ width: 1280, height: 720 }, { width: 1000, height: 1000 });
    expect(t.scale).toBeCloseTo(1000 / 720);
    expect(t.offsetX).toBeCloseTo((1000 - 1280 * (1000 / 720)) / 2);
    expect(t.offsetY).toBeCloseTo(0);
    const center = videoPercentToScreen(50, 50, { width: 1280, height: 720 }, { width: 1000, height: 1000 });
    expect(center.left).toBeCloseTo(500);
    expect(center.top).toBeCloseTo(500);
  });

  it('places cards above the anchor and keeps them on screen', () => {
    const container = { width: 1000, height: 600 };
    const card = { width: 180, height: 160 };
    expect(cardPosition({ left: 500, top: 400 }, card, container)).toEqual({ left: 410, top: 228 });
    expect(cardPosition({ left: 10, top: 50 }, card, container)).toEqual({ left: 8, top: 8 });
    expect(cardPosition({ left: 995, top: 400 }, card, container)).toEqual({ left: 812, top: 228 });
  });
});

describe('misc helpers', () => {
  it('builds asset urls and detects drift', () => {
    expect(assetUrl('demo', 'pov_near.mp4')).toBe('/matches/demo/pov_near.mp4');
    expect(needsResync(1.0, 1.1)).toBe(false);
    expect(needsResync(1.0, 1.3)).toBe(true);
  });
});
```

`src/hooks/useMatchData.test.tsx`:
```tsx
import { renderHook, waitFor } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { mockFetch, sampleIndex, sampleManifest, sampleTracks } from '../test/fixtures';
import { useMatchData, useMatchIndex } from './useMatchData';

describe('useMatchData', () => {
  it('loads the manifest and then its tracks file', async () => {
    const fetchMock = mockFetch({
      '/matches/demo/manifest.json': sampleManifest,
      '/matches/demo/tracks.json': sampleTracks,
    });
    const { result } = renderHook(() => useMatchData('demo'));
    await waitFor(() => expect(result.current.tracks).not.toBeNull());
    expect(result.current.manifest?.title).toBe('Carlos Alcaraz vs Jannik Sinner');
    expect(result.current.error).toBeNull();
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it('reports a missing match', async () => {
    mockFetch({});
    const { result } = renderHook(() => useMatchData('nope'));
    await waitFor(() => expect(result.current.error).toMatch(/404/));
    expect(result.current.manifest).toBeNull();
  });
});

describe('useMatchIndex', () => {
  it('lists exported matches', async () => {
    mockFetch({ '/matches/index.json': sampleIndex });
    const { result } = renderHook(() => useMatchIndex());
    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.matches.map((m) => m.id)).toEqual(['demo']);
  });
});
```

- [ ] **Step 4: Run the tests and confirm they fail**

Run: `npm test`
Expected: FAIL. The modules `./match` and `./useMatchData` cannot be resolved.

- [ ] **Step 5: Implement `src/lib/match.ts`**

```ts
/** Types for the files written by `horizon export` (spec §6) and pure helpers for the stream page. */

export type PlayerRole = 'near' | 'far';

export interface MatchSummary {
  id: string;
  title: string;
  near: string;
  far: string;
  competition: string;
  status: 'live' | 'upcoming' | 'replay';
}

export interface MatchIndex {
  matches: MatchSummary[];
}

export interface PlayerInfo {
  id: PlayerRole;
  name: string;
  description: string;
  color: string;
  pov: string;
  statureM: number;
}

export interface MatchManifest {
  id: string;
  title: string;
  competition: string;
  summary: string;
  score: string;
  fps: number;
  frameCount: number;
  width: number;
  height: number;
  video: string;
  tracks: string;
  players: PlayerInfo[];
  freeCam: string | null;
  viewerUrl: string;
}

export interface PlayerTrackSeries {
  x: number[];
  y: number[];
  visible: number[];
  speedKmh: number[];
  distanceM: number[];
}

export interface TracksFile {
  fps: number;
  frameCount: number;
  players: Record<PlayerRole, PlayerTrackSeries>;
}

export interface PlayerSnapshot {
  x: number;
  y: number;
  visible: boolean;
  speedKmh: number;
  distanceM: number;
  topSpeedKmh: number;
}

export interface Size {
  width: number;
  height: number;
}

export interface Point {
  left: number;
  top: number;
}

export const MATCHES_ROOT = '/matches';

export function assetUrl(matchId: string, file: string): string {
  return `${MATCHES_ROOT}/${encodeURIComponent(matchId)}/${file}`;
}

export function frameIndexAt(time: number, fps: number, frameCount: number): number {
  if (!Number.isFinite(time) || frameCount <= 0) return 0;
  return Math.min(frameCount - 1, Math.max(0, Math.floor(time * fps + 1e-6)));
}

export function playerSnapshot(tracks: TracksFile, role: PlayerRole, frame: number): PlayerSnapshot {
  const series = tracks.players[role];
  const i = Math.min(Math.max(frame, 0), series.x.length - 1);
  return {
    x: series.x[i],
    y: series.y[i],
    visible: series.visible[i] === 1,
    speedKmh: series.speedKmh[i],
    distanceM: series.distanceM[i],
    topSpeedKmh: Math.max(...series.speedKmh.slice(0, i + 1)),
  };
}

/** Scale and offset that CSS `object-fit: cover` applies to a video inside a container. */
export function coverTransform(video: Size, container: Size): { scale: number; offsetX: number; offsetY: number } {
  const scale = Math.max(container.width / video.width, container.height / video.height);
  return {
    scale,
    offsetX: (container.width - video.width * scale) / 2,
    offsetY: (container.height - video.height * scale) / 2,
  };
}

export function videoPercentToScreen(xPercent: number, yPercent: number, video: Size, container: Size): Point {
  const { scale, offsetX, offsetY } = coverTransform(video, container);
  return {
    left: (xPercent / 100) * video.width * scale + offsetX,
    top: (yPercent / 100) * video.height * scale + offsetY,
  };
}

function clamp(value: number, min: number, max: number): number {
  return max < min ? min : Math.min(Math.max(value, min), max);
}

/** Card centred above the anchor (head), kept 8 px inside the container. */
export function cardPosition(anchor: Point, card: Size, container: Size, gap = 12): Point {
  return {
    left: clamp(anchor.left - card.width / 2, 8, container.width - card.width - 8),
    top: clamp(anchor.top - card.height - gap, 8, container.height - card.height - 8),
  };
}

export function needsResync(povTime: number, mainTime: number, tolerance = 0.15): boolean {
  return Math.abs(povTime - mainTime) > tolerance;
}
```

- [ ] **Step 6: Implement `src/hooks/useMatchData.ts`**

```ts
import { useEffect, useState } from 'react';
import { MATCHES_ROOT, assetUrl } from '../lib/match';
import type { MatchIndex, MatchManifest, MatchSummary, TracksFile } from '../lib/match';

async function fetchJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return (await response.json()) as T;
}

interface MatchData {
  manifest: MatchManifest | null;
  tracks: TracksFile | null;
  error: string | null;
}

export function useMatchData(matchId: string | undefined): MatchData {
  const [data, setData] = useState<MatchData>({ manifest: null, tracks: null, error: null });

  useEffect(() => {
    if (!matchId) return;
    let cancelled = false;
    const load = async () => {
      try {
        const manifest = await fetchJson<MatchManifest>(assetUrl(matchId, 'manifest.json'));
        const tracks = await fetchJson<TracksFile>(assetUrl(matchId, manifest.tracks));
        if (!cancelled) setData({ manifest, tracks, error: null });
      } catch (err) {
        if (!cancelled) setData({ manifest: null, tracks: null, error: err instanceof Error ? err.message : String(err) });
      }
    };
    void load();
    return () => {
      cancelled = true;
    };
  }, [matchId]);

  return data;
}

interface IndexData {
  matches: MatchSummary[];
  error: string | null;
  loading: boolean;
}

export function useMatchIndex(): IndexData {
  const [data, setData] = useState<IndexData>({ matches: [], error: null, loading: true });

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const index = await fetchJson<MatchIndex>(`${MATCHES_ROOT}/index.json`);
        if (!cancelled) setData({ matches: index.matches, error: null, loading: false });
      } catch (err) {
        if (!cancelled) setData({ matches: [], error: err instanceof Error ? err.message : String(err), loading: false });
      }
    };
    void load();
    return () => {
      cancelled = true;
    };
  }, []);

  return data;
}
```

- [ ] **Step 7: Run the tests, the type-check and the linter**

Run: `npm test`
Expected: `8 passed` (2 test files).

Run: `npx tsc -b`
Expected: no output. (A full `npm run build` still fails at this point because of the missing soccer POV assets; Task 16 removes those imports.)

Run: `npm run lint`
Expected: no errors.

- [ ] **Step 8: Commit**

```powershell
git add package.json package-lock.json vite.config.ts src/test src/lib src/hooks/useMatchData.ts src/hooks/useMatchData.test.tsx
git commit -m "feat(web): add Vitest harness and typed match data layer"
```

### Task 16: Tennis UI components

**Files:**
- Create:
  - `src/components/tennis/tennis.css`
  - `src/components/tennis/TennisScoreOverlay.tsx`
  - `src/components/tennis/PovVideo.tsx`
  - `src/components/tennis/TennisPlayerCard.tsx`
  - `src/components/tennis/PovOverlay.tsx`
  - `src/components/tennis/FreeCamOverlay.tsx`
- Test: `src/components/tennis/tennis.test.tsx`

**Interfaces:**
- Consumes:
  - `PlayerInfo`, `PlayerSnapshot`, `needsResync` (Task 15)
  - existing classes in `src/components/player/player-glass.css` (`player-card*`, `pov-*`)
  - `sampleManifest` (Task 15)
- Produces:
  - `TennisScoreOverlay({ near, far, score, competition })`
  - `PovVideo({ src, mainTime, isPlaying, label, className? })`: keeps a POV `<video>` within 0.15 s of the broadcast clip and mirrors play/pause
  - `TennisPlayerCard({ player, povSrc, snapshot, mainTime, isPlaying, onExpand })`
  - `PovOverlay({ player, povSrc, snapshot, mainTime, isPlaying, onClose })`
  - `FreeCamOverlay({ matchId, viewerUrl, clipSrc, mainTime, isPlaying, onClose })`: "Live 3D" tab = iframe to the Viser viewer; "Exported clip" tab (only when `clipSrc`) = synced `free_cam.mp4`

- [ ] **Step 1: Write the failing tests**

`src/components/tennis/tennis.test.tsx`:
```tsx
import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import type { PlayerSnapshot } from '../../lib/match';
import { sampleManifest } from '../../test/fixtures';
import { FreeCamOverlay } from './FreeCamOverlay';
import { PovOverlay } from './PovOverlay';
import { PovVideo } from './PovVideo';
import { TennisPlayerCard } from './TennisPlayerCard';
import { TennisScoreOverlay } from './TennisScoreOverlay';

const [near, far] = sampleManifest.players;
const snapshot: PlayerSnapshot = { x: 50, y: 60, visible: true, speedKmh: 17.6, distanceM: 12.34, topSpeedKmh: 21.2 };

describe('TennisScoreOverlay', () => {
  it('shows both surnames and the score', () => {
    render(<TennisScoreOverlay near={near} far={far} score="6-4 2-1 30-15" competition="Tennis" />);
    expect(screen.getByText('ALCARAZ')).toBeInTheDocument();
    expect(screen.getByText('SINNER')).toBeInTheDocument();
    expect(screen.getByText('6-4 2-1 30-15')).toBeInTheDocument();
    expect(screen.getByRole('status')).toHaveAccessibleName(/Carlos Alcaraz versus Jannik Sinner/);
  });
});

describe('TennisPlayerCard', () => {
  it('shows the player and live speed and expands on click', () => {
    const onExpand = vi.fn();
    render(
      <TennisPlayerCard player={near} povSrc="/matches/demo/pov_near.mp4" snapshot={snapshot} mainTime={0} isPlaying={false} onExpand={onExpand} />,
    );
    expect(screen.getByText('Carlos Alcaraz')).toBeInTheDocument();
    expect(screen.getByText('NEAR')).toBeInTheDocument();
    expect(screen.getByText('18')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /Carlos Alcaraz/ }));
    expect(onExpand).toHaveBeenCalledWith(near);
  });
});

describe('PovVideo', () => {
  it('follows the broadcast clock and play state', () => {
    const { rerender } = render(<PovVideo src="/pov.mp4" mainTime={0} isPlaying={false} label="pov" />);
    const video = screen.getByLabelText('pov') as HTMLVideoElement;
    let time = 0;
    let paused = true;
    Object.defineProperty(video, 'currentTime', { configurable: true, get: () => time, set: (v: number) => { time = v; } });
    Object.defineProperty(video, 'paused', { configurable: true, get: () => paused });
    const play = vi.spyOn(video, 'play').mockImplementation(async () => { paused = false; });
    const pause = vi.spyOn(video, 'pause').mockImplementation(() => { paused = true; });

    rerender(<PovVideo src="/pov.mp4" mainTime={3} isPlaying={true} label="pov" />);
    expect(time).toBe(3);
    expect(play).toHaveBeenCalledTimes(1);

    rerender(<PovVideo src="/pov.mp4" mainTime={3.05} isPlaying={true} label="pov" />);
    expect(time).toBe(3);

    rerender(<PovVideo src="/pov.mp4" mainTime={3.1} isPlaying={false} label="pov" />);
    expect(pause).toHaveBeenCalledTimes(1);
  });
});

describe('PovOverlay', () => {
  it('shows speed, top speed and distance and closes', () => {
    const onClose = vi.fn();
    render(<PovOverlay player={far} povSrc="/matches/demo/pov_far.mp4" snapshot={snapshot} mainTime={0} isPlaying={false} onClose={onClose} />);
    expect(screen.getByRole('dialog', { name: /Jannik Sinner first-person view/ })).toBeInTheDocument();
    expect(screen.getByText('18')).toBeInTheDocument();
    expect(screen.getByText('21')).toBeInTheDocument();
    expect(screen.getByText('12.3')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Close first-person view' }));
    expect(onClose).toHaveBeenCalled();
  });
});

describe('FreeCamOverlay', () => {
  it('embeds the live Viser viewer', () => {
    render(<FreeCamOverlay matchId="demo" viewerUrl="http://localhost:8080" clipSrc={null} mainTime={0} isPlaying={false} onClose={() => {}} />);
    expect(screen.getByTitle('3D free camera viewer')).toHaveAttribute('src', 'http://localhost:8080');
    expect(screen.queryByRole('tab', { name: 'Exported clip' })).not.toBeInTheDocument();
    expect(screen.getByText('horizon view --match-id demo')).toBeInTheDocument();
  });

  it('switches to the exported clip and closes', () => {
    const onClose = vi.fn();
    render(
      <FreeCamOverlay matchId="demo" viewerUrl="http://localhost:8080" clipSrc="/matches/demo/free_cam.mp4" mainTime={0} isPlaying={false} onClose={onClose} />,
    );
    fireEvent.click(screen.getByRole('tab', { name: 'Exported clip' }));
    expect(screen.getByLabelText('Exported free camera clip')).toHaveAttribute('src', '/matches/demo/free_cam.mp4');
    expect(screen.queryByTitle('3D free camera viewer')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Close free camera' }));
    expect(onClose).toHaveBeenCalled();
  });
});
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `npm test`
Expected: FAIL. The imports under `src/components/tennis/` cannot be resolved.

- [ ] **Step 3: Write `src/components/tennis/tennis.css`**

```css
/* Tennis scoreboard, free-camera panel and small overrides (glass style from the soccer UI). */

.tennis-score {
  position: absolute;
  top: 28px;
  left: 50%;
  transform: translateX(-50%);
  z-index: 100;
}

.tennis-score-glass {
  position: absolute;
  inset: 0;
  background: rgba(0, 0, 0, 0.75);
  backdrop-filter: blur(40px);
  -webkit-backdrop-filter: blur(40px);
  border: 1px solid rgba(255, 255, 255, 0.12);
  border-radius: 12px;
}

.tennis-score-content {
  position: relative;
  display: flex;
  align-items: center;
  gap: 20px;
  padding: 14px 22px;
}

.tennis-score-player {
  display: flex;
  align-items: center;
  gap: 10px;
}

.tennis-score-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
}

.tennis-score-name {
  font-family: 'Bebas Neue', Impact, sans-serif;
  font-size: 20px;
  letter-spacing: 0.06em;
  color: rgba(255, 255, 255, 0.95);
}

.tennis-score-center {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 0 20px;
  border-left: 1px solid rgba(255, 255, 255, 0.08);
  border-right: 1px solid rgba(255, 255, 255, 0.08);
}

.tennis-score-value {
  font-family: 'Geist Mono', monospace;
  font-size: 22px;
  font-weight: 300;
  color: #fff;
  font-variant-numeric: tabular-nums;
}

.tennis-score-label {
  margin-top: 4px;
  font-family: 'Geist Mono', monospace;
  font-size: 9px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: rgba(255, 255, 255, 0.4);
}

/* Player cards are buttons for keyboard access; strip native button chrome. */
button.player-card {
  display: block;
  padding: 0;
  border: none;
  background: none;
  color: inherit;
  font: inherit;
  text-align: left;
}

button.player-card:focus-visible {
  outline: 2px solid rgba(255, 255, 255, 0.8);
  outline-offset: 2px;
}

/* Stream-page buttons next to Back / POV */
.freecam-button {
  position: absolute;
  top: 28px;
  left: 192px;
  z-index: 100;
  padding: 8px 14px;
  background: rgba(0, 0, 0, 0.75);
  backdrop-filter: blur(40px);
  -webkit-backdrop-filter: blur(40px);
  border: 1px solid rgba(255, 255, 255, 0.12);
  border-radius: 8px;
  color: rgba(255, 255, 255, 0.8);
  font-family: 'Geist Mono', monospace;
  font-size: 10px;
  font-weight: 500;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  cursor: pointer;
}

.freecam-button:hover {
  color: #fff;
  border-color: rgba(255, 255, 255, 0.25);
}

/* Free camera panel */
.freecam {
  position: fixed;
  inset: 0;
  z-index: 300;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(0, 0, 0, 0.6);
}

.freecam-panel {
  width: min(1200px, 92vw);
  height: min(760px, 86vh);
  display: flex;
  flex-direction: column;
  background: rgba(0, 0, 0, 0.85);
  border: 1px solid rgba(255, 255, 255, 0.12);
  border-radius: 12px;
  overflow: hidden;
}

.freecam-header {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 12px 16px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
  font-family: 'Geist Mono', monospace;
  font-size: 11px;
  color: rgba(255, 255, 255, 0.7);
}

.freecam-title {
  font-family: 'Bebas Neue', Impact, sans-serif;
  font-size: 18px;
  letter-spacing: 0.06em;
  color: #fff;
}

.freecam-tabs {
  display: flex;
  gap: 6px;
  flex: 1;
}

.freecam-tabs button {
  padding: 6px 10px;
  border-radius: 6px;
  border: 1px solid rgba(255, 255, 255, 0.12);
  background: transparent;
  color: rgba(255, 255, 255, 0.6);
  font: inherit;
  cursor: pointer;
}

.freecam-tabs button[aria-selected='true'] {
  background: rgba(255, 255, 255, 0.1);
  color: #fff;
}

.freecam-link {
  color: rgba(255, 255, 255, 0.7);
}

.freecam-frame,
.freecam-video {
  flex: 1;
  width: 100%;
  border: 0;
  background: #0a0a0a;
  object-fit: contain;
}

.freecam-hint {
  margin: 0;
  padding: 8px 16px;
  font-family: 'Geist Mono', monospace;
  font-size: 10px;
  color: rgba(255, 255, 255, 0.45);
}

/* Loading / error states */
.stream-status {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 16px;
  min-height: 100vh;
  font-family: 'Geist Mono', monospace;
  color: rgba(255, 255, 255, 0.7);
}

.games-empty {
  padding: 48px 24px;
  font-family: 'Geist Mono', monospace;
  font-size: 12px;
  color: rgba(255, 255, 255, 0.6);
}
```

- [ ] **Step 4: Implement the components**

`src/components/tennis/TennisScoreOverlay.tsx`:
```tsx
import type { PlayerInfo } from '../../lib/match';
import './tennis.css';

interface TennisScoreOverlayProps {
  near: PlayerInfo;
  far: PlayerInfo;
  score: string;
  competition: string;
}

function surname(name: string): string {
  const parts = name.trim().split(/\s+/);
  return (parts[parts.length - 1] || name).toUpperCase();
}

export function TennisScoreOverlay({ near, far, score, competition }: TennisScoreOverlayProps) {
  return (
    <div
      className="tennis-score"
      role="status"
      aria-label={`${near.name} versus ${far.name}${score ? `, score ${score}` : ''}`}
    >
      <div className="tennis-score-glass" aria-hidden="true" />
      <div className="tennis-score-content">
        <div className="tennis-score-player">
          <span className="tennis-score-dot" style={{ background: near.color }} aria-hidden="true" />
          <span className="tennis-score-name">{surname(near.name)}</span>
        </div>
        <div className="tennis-score-center">
          <span className="tennis-score-value">{score || 'vs'}</span>
          <span className="tennis-score-label">{competition}</span>
        </div>
        <div className="tennis-score-player">
          <span className="tennis-score-name">{surname(far.name)}</span>
          <span className="tennis-score-dot" style={{ background: far.color }} aria-hidden="true" />
        </div>
      </div>
    </div>
  );
}
```

`src/components/tennis/PovVideo.tsx`:
```tsx
import { useEffect, useRef } from 'react';
import { needsResync } from '../../lib/match';

interface PovVideoProps {
  src: string;
  mainTime: number;
  isPlaying: boolean;
  label: string;
  className?: string;
}

/** A POV clip rendered frame-aligned with the broadcast video: same fps and frame count, so time maps 1:1. */
export function PovVideo({ src, mainTime, isPlaying, label, className }: PovVideoProps) {
  const ref = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    const video = ref.current;
    if (!video) return;
    if (needsResync(video.currentTime, mainTime)) {
      video.currentTime = mainTime;
    }
    if (isPlaying && video.paused) {
      video.play().catch(() => {});
    } else if (!isPlaying && !video.paused) {
      video.pause();
    }
  }, [mainTime, isPlaying]);

  return <video ref={ref} className={className} src={src} muted playsInline preload="auto" aria-label={label} />;
}
```

`src/components/tennis/TennisPlayerCard.tsx`:
```tsx
import { useState } from 'react';
import type { CSSProperties } from 'react';
import type { PlayerInfo, PlayerSnapshot } from '../../lib/match';
import { PovVideo } from './PovVideo';
import '../player/player-glass.css';
import './tennis.css';

interface TennisPlayerCardProps {
  player: PlayerInfo;
  povSrc: string;
  snapshot: PlayerSnapshot;
  mainTime: number;
  isPlaying: boolean;
  onExpand: (player: PlayerInfo) => void;
}

export function TennisPlayerCard({ player, povSrc, snapshot, mainTime, isPlaying, onExpand }: TennisPlayerCardProps) {
  const [hovered, setHovered] = useState(false);
  const speed = Math.round(snapshot.speedKmh);

  return (
    <button
      type="button"
      className={`player-card ${hovered ? 'player-card--hover' : ''}`}
      style={{ '--player-accent': player.color } as CSSProperties}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      onClick={() => onExpand(player)}
      aria-label={`${player.name}, ${player.id} player, ${speed} kilometres per hour. Open first-person view.`}
    >
      <div className="player-card-glass" />
      <div className="player-card-border" />
      <div className="player-card-accent" />
      <div className="player-card-content">
        <div className="player-card-pov">
          <PovVideo src={povSrc} mainTime={mainTime} isPlaying={isPlaying} className="player-card-pov-video" label={`${player.name} point of view`} />
          <div className="player-card-pov-live">
            <span className="player-card-pov-dot" />
            <span>POV</span>
          </div>
        </div>
        <div className="player-card-info">
          <div className="player-card-header">
            <div className="player-card-name-section">
              <span className="player-card-name">{player.name}</span>
              <div className="player-card-meta">
                <span className="player-card-role">{player.id === 'near' ? 'NEAR' : 'FAR'}</span>
              </div>
            </div>
            <div className="player-card-stat">
              <span className="player-card-stat-value">{speed}</span>
              <span className="player-card-stat-label">KM/H</span>
            </div>
          </div>
        </div>
      </div>
    </button>
  );
}
```

`src/components/tennis/PovOverlay.tsx`:
```tsx
import type { CSSProperties } from 'react';
import type { PlayerInfo, PlayerSnapshot } from '../../lib/match';
import { PovVideo } from './PovVideo';
import '../player/player-glass.css';

interface PovOverlayProps {
  player: PlayerInfo;
  povSrc: string;
  snapshot: PlayerSnapshot;
  mainTime: number;
  isPlaying: boolean;
  onClose: () => void;
}

export function PovOverlay({ player, povSrc, snapshot, mainTime, isPlaying, onClose }: PovOverlayProps) {
  const stats = [
    { value: String(Math.round(snapshot.speedKmh)), label: 'km/h' },
    { value: String(Math.round(snapshot.topSpeedKmh)), label: 'Top km/h' },
    { value: snapshot.distanceM.toFixed(1), label: 'Metres' },
  ];

  return (
    <div className="pov-glass" role="dialog" aria-label={`${player.name} first-person view`}>
      <div className="pov-glass-container" style={{ '--player-accent': player.color } as CSSProperties}>
        <div className="pov-glass-bg" />
        <div className="pov-glass-surface" />
        <div className="pov-header">
          <div className="pov-player-info">
            <div className="pov-avatar">
              <span>{player.name.charAt(0)}</span>
            </div>
            <div className="pov-details">
              <span className="pov-player-name">{player.name}</span>
              <span className="pov-player-position">
                {player.id === 'near' ? 'Near end' : 'Far end'} · {player.statureM.toFixed(2)} m
              </span>
            </div>
          </div>
          <button className="pov-close" onClick={onClose} aria-label="Close first-person view">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
              <path d="M18 6L6 18M6 6l12 12" />
            </svg>
          </button>
        </div>
        <div className="pov-video">
          <PovVideo src={povSrc} mainTime={mainTime} isPlaying={isPlaying} className="pov-video-player" label={`${player.name} first-person video`} />
        </div>
        <div className="pov-stats">
          {stats.map((stat) => (
            <div key={stat.label} className="pov-stat">
              <span className="pov-stat-value">{stat.value}</span>
              <span className="pov-stat-label">{stat.label}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
```

`src/components/tennis/FreeCamOverlay.tsx`:
```tsx
import { useState } from 'react';
import { PovVideo } from './PovVideo';
import '../player/player-glass.css';
import './tennis.css';

interface FreeCamOverlayProps {
  matchId: string;
  viewerUrl: string;
  clipSrc: string | null;
  mainTime: number;
  isPlaying: boolean;
  onClose: () => void;
}

export function FreeCamOverlay({ matchId, viewerUrl, clipSrc, mainTime, isPlaying, onClose }: FreeCamOverlayProps) {
  const [mode, setMode] = useState<'live' | 'clip'>('live');
  const showClip = mode === 'clip' && clipSrc !== null;

  return (
    <div className="freecam" role="dialog" aria-label="Free camera">
      <div className="freecam-panel">
        <header className="freecam-header">
          <span className="freecam-title">Free camera</span>
          <div className="freecam-tabs" role="tablist">
            <button type="button" role="tab" aria-selected={!showClip} onClick={() => setMode('live')}>
              Live 3D
            </button>
            {clipSrc && (
              <button type="button" role="tab" aria-selected={showClip} onClick={() => setMode('clip')}>
                Exported clip
              </button>
            )}
          </div>
          <a className="freecam-link" href={viewerUrl} target="_blank" rel="noreferrer">
            Open in new tab
          </a>
          <button className="pov-close" onClick={onClose} aria-label="Close free camera">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
              <path d="M18 6L6 18M6 6l12 12" />
            </svg>
          </button>
        </header>
        {showClip && clipSrc ? (
          <PovVideo src={clipSrc} mainTime={mainTime} isPlaying={isPlaying} className="freecam-video" label="Exported free camera clip" />
        ) : (
          <>
            <iframe className="freecam-frame" src={viewerUrl} title="3D free camera viewer" allow="fullscreen" />
            <p className="freecam-hint">
              Blank? Start the 3D viewer first: <code>horizon view --match-id {matchId}</code>
            </p>
          </>
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 5: Run the tests and the linter**

Run: `npm test`
Expected: `14 passed` (3 test files).

Run: `npm run lint`
Expected: no errors.

- [ ] **Step 6: Commit**

```powershell
git add src/components/tennis
git commit -m "feat(web): add tennis scoreboard, POV cards, POV overlay and free-camera panel"
```

### Task 17: Tennis pages and soccer clean-up

**Files:**
- Modify: `src/pages/GamesPage.tsx` (rewrite), `src/pages/StreamPage.tsx` (rewrite), `src/pages/ProcessingPage.tsx`, `index.html`
- Delete:
  - `src/components/stream/PenaltyScoreOverlay.tsx`, `src/components/stream/penalty-glass.css`
  - `src/components/stream/GoalConfetti.tsx`, `src/components/stream/goal-confetti.css`
  - `src/hooks/usePlayerTracking.ts`
  - `src/components/player/PlayerCard.tsx`, `src/components/player/POVVideoOverlay.tsx`, `src/components/player/player-glass.css.bak`
  - `public/bruno_tracks.json`
  - `src/assets/man_utd.png`, `src/assets/arsenal.png`, `src/assets/*.mov`
- Test: `src/pages/pages.test.tsx`

**Interfaces:**
- Consumes:
  - `useMatchData`, `useMatchIndex`, `assetUrl`, `frameIndexAt`, `playerSnapshot`, `videoPercentToScreen`, `cardPosition`, and the fixtures (Task 15)
  - all components from Task 16
  - existing `StreamContainer`, `VideoControls`, `GameCard`, `AccessibilityToggle`, `AccessibilityProvider`
- Produces:
  - route `/`: match list from `public/matches/index.json`
  - route `/processing/:gameId`: short transition
  - route `/stream/:gameId`: tennis stream with the scoreboard, two POV cards pinned above the players, the POV overlay, and FREE CAM (button accessible name "Open free camera")

- [ ] **Step 1: Write the failing page tests**

`src/pages/pages.test.tsx`:
```tsx
import { fireEvent, render, screen } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { describe, expect, it } from 'vitest';
import { AccessibilityProvider } from '../contexts/AccessibilityContext';
import { mockFetch, sampleIndex, sampleManifest, sampleTracks } from '../test/fixtures';
import { GamesPage } from './GamesPage';
import { StreamPage } from './StreamPage';

function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <AccessibilityProvider>
        <Routes>
          <Route path="/" element={<GamesPage />} />
          <Route path="/stream/:gameId" element={<StreamPage />} />
        </Routes>
      </AccessibilityProvider>
    </MemoryRouter>,
  );
}

describe('GamesPage', () => {
  it('lists exported matches', async () => {
    mockFetch({ '/matches/index.json': sampleIndex });
    renderAt('/');
    expect(await screen.findByRole('listitem', { name: /Carlos Alcaraz versus Jannik Sinner/ })).toBeInTheDocument();
  });

  it('explains how to create a match when none exist', async () => {
    mockFetch({});
    renderAt('/');
    expect(await screen.findByText(/No processed matches yet/)).toBeInTheDocument();
  });
});

describe('StreamPage', () => {
  it('shows the scoreboard, both POV cards, the POV overlay and the free camera', async () => {
    mockFetch({ '/matches/demo/manifest.json': sampleManifest, '/matches/demo/tracks.json': sampleTracks });
    renderAt('/stream/demo');
    expect(await screen.findByText('ALCARAZ')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Carlos Alcaraz, near player/ })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /Jannik Sinner, far player/ }));
    expect(screen.getByRole('dialog', { name: /Jannik Sinner first-person view/ })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Open free camera' }));
    expect(screen.getByTitle('3D free camera viewer')).toHaveAttribute('src', 'http://localhost:8080');
  });

  it('reports unknown matches', async () => {
    mockFetch({});
    renderAt('/stream/missing');
    expect(await screen.findByText(/Match not found/)).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `npm test`
Expected: FAIL. The GamesPage tests find no match list (the page still renders hard-coded soccer fixtures), and the StreamPage tests fail to import the missing `goalie_pov.mov`.

- [ ] **Step 3: Rewrite `src/pages/GamesPage.tsx`**

```tsx
import { GameCard, type Game } from '../components/games/GameCard';
import { AccessibilityToggle } from '../components/shared/AccessibilityToggle';
import { useMatchIndex } from '../hooks/useMatchData';
import type { MatchSummary } from '../lib/match';
import '../components/games/games-glass.css';
import '../components/tennis/tennis.css';

function initials(name: string): string {
  return name
    .split(/\s+/)
    .map((part) => part[0] ?? '')
    .join('')
    .slice(0, 3)
    .toUpperCase();
}

function toGame(match: MatchSummary): Game {
  return {
    id: match.id,
    homeTeam: { name: match.near, shortName: initials(match.near), color: '#3B82F6' },
    awayTeam: { name: match.far, shortName: initials(match.far), color: '#F97316' },
    competition: match.competition,
    competitionShort: match.competition.toUpperCase(),
    time: '',
    status: match.status,
  };
}

export function GamesPage() {
  const { matches, loading } = useMatchIndex();

  return (
    <div className="games-page">
      <a href="#main-content" className="skip-link">
        Skip to main content
      </a>

      <header className="games-header" role="banner">
        <div className="games-brand">
          <span className="games-logo" aria-hidden="true">ph</span>
          <h1 className="games-brand-title">Project Horizon · Tennis</h1>
        </div>
        <p className="games-tagline">See the rally through<br />the eyes of those who play it</p>
      </header>

      <main id="main-content" role="main">
        <h2 className="sr-only">Available Matches</h2>
        {loading ? (
          <p className="games-empty" role="status">Loading matches…</p>
        ) : matches.length === 0 ? (
          <p className="games-empty" role="status">
            No processed matches yet. In <code>pipeline\</code> run{' '}
            <code>.venv\Scripts\python -m horizon all path\to\clip.mp4 --match-id demo</code>.
          </p>
        ) : (
          <div className="games-grid" role="list" aria-label="Tennis matches">
            {matches.map((match) => (
              <GameCard key={match.id} game={toGame(match)} />
            ))}
          </div>
        )}
      </main>

      <AccessibilityToggle />
    </div>
  );
}
```

- [ ] **Step 4: Update `src/pages/ProcessingPage.tsx`**

Replace the `statusMessages` array and the `DURATION` constant with:
```tsx
const statusMessages = [
  'Loading broadcast clip...',
  'Calibrating court geometry...',
  'Estimating monocular depth...',
  'Building the 3D point cloud...',
  'Placing player cameras...',
  'Stream ready',
];

const DURATION = 6000; // artifacts are precomputed by the pipeline; this is only a transition
```

- [ ] **Step 5: Rewrite `src/pages/StreamPage.tsx`**

```tsx
import { useCallback, useEffect, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { AccessibilityToggle } from '../components/shared/AccessibilityToggle';
import { StreamContainer } from '../components/stream/StreamContainer';
import type { StreamContainerRef } from '../components/stream/StreamContainer';
import { VideoControls } from '../components/stream/VideoControls';
import { FreeCamOverlay } from '../components/tennis/FreeCamOverlay';
import { PovOverlay } from '../components/tennis/PovOverlay';
import { TennisPlayerCard } from '../components/tennis/TennisPlayerCard';
import { TennisScoreOverlay } from '../components/tennis/TennisScoreOverlay';
import { useMatchData } from '../hooks/useMatchData';
import { assetUrl, cardPosition, frameIndexAt, playerSnapshot, videoPercentToScreen } from '../lib/match';
import type { PlayerRole } from '../lib/match';
import '../App.css';
import '../components/tennis/tennis.css';

const CARD_SIZE = { width: 180, height: 160 };

function BackButton({ onClick }: { onClick: () => void }) {
  return (
    <button className="home-button" onClick={onClick} aria-label="Return to home">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M15 18l-6-6 6-6" />
      </svg>
      <span>Back</span>
    </button>
  );
}

export function StreamPage() {
  const navigate = useNavigate();
  const { gameId } = useParams<{ gameId: string }>();
  const { manifest, tracks, error } = useMatchData(gameId);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [showCards, setShowCards] = useState(true);
  const [expandedId, setExpandedId] = useState<PlayerRole | null>(null);
  const [freeCamOpen, setFreeCamOpen] = useState(false);
  const [viewport, setViewport] = useState(() => ({ width: window.innerWidth, height: window.innerHeight }));
  const streamRef = useRef<StreamContainerRef>(null);

  useEffect(() => {
    const onResize = () => setViewport({ width: window.innerWidth, height: window.innerHeight });
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  // `timeupdate` fires only ~4x per second; poll the video every animation frame while playing.
  useEffect(() => {
    if (!isPlaying) return;
    let raf = 0;
    const tick = () => {
      setCurrentTime(streamRef.current?.getCurrentTime() ?? 0);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [isPlaying]);

  const handleTimeUpdate = useCallback((time: number, dur: number) => {
    setCurrentTime(time);
    setDuration(dur);
  }, []);

  const handleStateChange = useCallback((playing: boolean) => {
    setIsPlaying(playing);
  }, []);

  if (error) {
    return (
      <div className="app">
        <BackButton onClick={() => navigate('/')} />
        <div className="stream-status" role="alert">
          <p>Match not found.</p>
          <p>Export it with <code>horizon export --match-id {gameId}</code> ({error})</p>
        </div>
      </div>
    );
  }

  if (!manifest || !tracks) {
    return (
      <div className="app">
        <div className="stream-status" role="status">Loading match…</div>
      </div>
    );
  }

  const frame = frameIndexAt(currentTime, manifest.fps, manifest.frameCount);
  const videoSize = { width: manifest.width, height: manifest.height };
  const [near, far] = manifest.players;
  const expanded = manifest.players.find((p) => p.id === expandedId) ?? null;

  return (
    <div className="app">
      <StreamContainer
        ref={streamRef}
        videoSrc={assetUrl(manifest.id, manifest.video)}
        onTimeUpdate={handleTimeUpdate}
        onStateChange={handleStateChange}
      >
        <BackButton onClick={() => navigate('/')} />

        <button
          className="toggle-pov-button"
          onClick={() => setShowCards(!showCards)}
          aria-label={showCards ? 'Hide POV cards' : 'Show POV cards'}
          aria-pressed={showCards}
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
            <circle cx="12" cy="12" r="3" />
          </svg>
          <span>POV</span>
        </button>

        <button className="freecam-button" onClick={() => setFreeCamOpen(true)} aria-label="Open free camera">
          Free cam
        </button>

        <TennisScoreOverlay near={near} far={far} score={manifest.score} competition={manifest.competition} />

        {showCards &&
          manifest.players.map((player, index) => {
            const snapshot = playerSnapshot(tracks, player.id, frame);
            const anchor = videoPercentToScreen(snapshot.x, snapshot.y, videoSize, viewport);
            const position = cardPosition(anchor, CARD_SIZE, viewport);
            return (
              <div
                key={player.id}
                className="player-card-wrapper"
                style={{ left: position.left, top: position.top, animationDelay: `${index * 100}ms`, transition: 'left 0.1s linear, top 0.1s linear' }}
              >
                <div style={{ opacity: snapshot.visible ? 1 : 0.5 }}>
                  <TennisPlayerCard
                    player={player}
                    povSrc={assetUrl(manifest.id, player.pov)}
                    snapshot={snapshot}
                    mainTime={currentTime}
                    isPlaying={isPlaying}
                    onExpand={(p) => setExpandedId(p.id)}
                  />
                </div>
              </div>
            );
          })}

        <VideoControls
          currentTime={currentTime}
          duration={duration}
          isPlaying={isPlaying}
          onPlayPause={() => (isPlaying ? streamRef.current?.pause() : streamRef.current?.play())}
          onSeek={(time) => streamRef.current?.seekTo(time)}
          onFullscreen={() => document.documentElement.requestFullscreen?.()}
        />
      </StreamContainer>

      {expanded && (
        <PovOverlay
          player={expanded}
          povSrc={assetUrl(manifest.id, expanded.pov)}
          snapshot={playerSnapshot(tracks, expanded.id, frame)}
          mainTime={currentTime}
          isPlaying={isPlaying}
          onClose={() => setExpandedId(null)}
        />
      )}

      {freeCamOpen && (
        <FreeCamOverlay
          matchId={manifest.id}
          viewerUrl={manifest.viewerUrl}
          clipSrc={manifest.freeCam ? assetUrl(manifest.id, manifest.freeCam) : null}
          mainTime={currentTime}
          isPlaying={isPlaying}
          onClose={() => setFreeCamOpen(false)}
        />
      )}

      <AccessibilityToggle />
    </div>
  );
}
```

- [ ] **Step 6: Remove the soccer-only files and update the page metadata**

```powershell
git rm src/components/stream/PenaltyScoreOverlay.tsx src/components/stream/penalty-glass.css src/components/stream/GoalConfetti.tsx src/components/stream/goal-confetti.css src/hooks/usePlayerTracking.ts src/components/player/PlayerCard.tsx src/components/player/POVVideoOverlay.tsx src/components/player/player-glass.css.bak public/bruno_tracks.json src/assets/man_utd.png src/assets/arsenal.png
Remove-Item src\assets\*.mov
```
In `index.html`, replace the description and title lines with:
```html
    <meta name="description" content="Project Horizon Tennis - see the rally through the eyes of both players, or place your own camera anywhere on court." />
```
```html
    <title>Project Horizon · Tennis</title>
```

- [ ] **Step 7: Run the tests, the build and the linter**

Run: `npm test`
Expected: `18 passed` (4 test files).

Run: `npm run build`
Expected: `tsc -b` succeeds and Vite prints `✓ built in …`.

Run: `npm run lint`
Expected: no errors.

- [ ] **Step 8: Commit**

```powershell
git add -A src public index.html
git commit -m "feat(web): tennis match list and stream page; remove soccer-only code and assets"
```

## Phase D — Integration

### Task 18: End-to-end run on a real clip, README, legacy removal

**Files:**
- Delete: `backend/` (superseded by `pipeline/horizon/tracking.py`), `preprocessing/` (superseded by `pipeline/horizon/llm/*` and `identify.py`)
- Rewrite: `README.md`
- Create (not committed): `pipeline/.env`, `pipeline/data/demo/*`, `public/matches/demo/*`

**Interfaces:**
- Consumes: every CLI command from Tasks 1–14 and the web app from Tasks 15–17
- Produces: a verified demo match and the project README

- [ ] **Step 1: Remove the legacy soccer pipeline**

```powershell
git rm -r backend preprocessing
```
Then search for leftovers with the Grep tool (pattern `preprocessing|backend/|bruno`, path `src`).
Expected: no matches.

- [ ] **Step 2: Rewrite `README.md`**

````markdown
# Project Horizon · Tennis

See the rally through the eyes of both players, or drop your own camera anywhere on court.

A broadcast tennis clip goes through an offline Python pipeline (`pipeline/`):
- It calibrates the court from clicked keypoints.
- It tracks every person (YOLO26-seg + ByteTrack).
- It identifies the two players with TwelveLabs Pegasus + Gemini, orchestrated by Backboard.
- It lifts the scene into 3D with a monocular depth model (Depth Anything V2) aligned to the court geometry.
- It renders first-person POV videos for both players.

A Viser app (`horizon view`) gives you a free-placement 3D camera. The React app (`src/`) plays everything back with POV cards pinned above the players.

Design: `docs/superpowers/specs/2026-09-20-tennis-horizon-design.md`

## Requirements

- Windows 10/11 with an NVIDIA GPU (developed on an RTX 3060 Ti, 8 GB). A CPU works but is slow.
- Node 24 + npm, Python 3.11, [uv](https://docs.astral.sh/uv/), Git
- Keys:
  - AWS with Bedrock model access to **TwelveLabs Pegasus 1.2** (us-east-1)
  - OpenRouter
  - Backboard. Optional: use `--orchestrator direct` to skip Backboard, or `none` to skip all AI services.

## Setup

```powershell
# web app
npm install

# pipeline
cd pipeline
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv\Scripts\python.exe -e ".[ml,viewer,dev]"
Copy-Item .env.example .env   # then fill in the keys
.venv\Scripts\python -m horizon doctor
```

## Process a clip

Use one continuous shot from the main broadcast camera: 3–20 s, both players visible, no cuts or zoom.

```powershell
cd pipeline
.venv\Scripts\python -m horizon all C:\path\to\match.mp4 --match-id demo --start 12 --duration 10
```

A window opens on the first frame. Click each keypoint named at the top; the red dot in the mini-court shows where it is.
- `S` skips a keypoint you can't see.
- `U` undoes the last click.
- `Enter` solves once you have at least 4 points.

Check `data\demo\calibration_preview.jpg`: the red court lines must sit on the painted lines.

Individual steps: `init`, `calibrate`, `track`, `identify`, `players`, `reconstruct`, `render`, `export`. Run `python -m horizon <step> --help` for options.

## Explore

```powershell
# terminal 1: 3D free camera (http://localhost:8080)
cd pipeline
.venv\Scripts\python -m horizon view --match-id demo

# terminal 2: web app (http://localhost:5173)
npm run dev
```

In the viewer:
- Drag the white gizmo to place the free camera, or use **Snap to near/far player**.
- **Follow** attaches the camera to a player: `eyes` is first person, `chase` is behind and above.
- **Look through free camera** shows its view live.
- **Export free-cam clip** writes `free_cam.mp4`. Then run `horizon export --match-id demo` to publish it to the web app's FREE CAM panel.

## Tests

```powershell
cd pipeline; .venv\Scripts\python -m pytest -q; cd ..
npm test
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `yolo26s-seg.pt` not found | `horizon track --match-id demo --model yolo11s-seg.pt` |
| OpenRouter 404 / model not found | Set `GEMINI_MODEL_ID` in `pipeline\.env` to a current id from https://openrouter.ai/google |
| Pegasus `AccessDeniedException` | Enable TwelveLabs Pegasus in the Bedrock console (Model access) for `AWS_REGION` |
| Pegasus rejects the clip (>25 MB) | Re-run `horizon init` with a shorter `--duration` or `--max-height 540` |
| Backboard errors | `horizon identify --match-id demo --orchestrator direct` |
| Wrong player chosen | Read the warnings in `identity.json`; re-run `identify` with `--frame N`, choosing a frame where both players are clearly visible |
| Calibration RMS > 4 px | `horizon calibrate --match-id demo` and click more keypoints (service-line T's help) |
| FREE CAM panel blank | Start `horizon view --match-id demo`; the panel embeds http://localhost:8080 |
| Sparse point cloud | `horizon reconstruct --match-id demo --stride 1 --depth-model depth-anything/Depth-Anything-V2-Base-hf` |

## Known limitations

A single camera cannot see everything.
- The far player's POV looks back towards the broadcast camera, where nothing was filmed, so that area is sky or fill colour.
- Players are 2.5-D, like billboards.

See spec §7.
````

- [ ] **Step 3: Configure the keys and check the environment**

Copy `pipeline\.env.example` to `pipeline\.env` and fill in `AWS_REGION`, `TWELVELABS_MODEL_ID`, `OPENROUTER_API_KEY`, `GEMINI_MODEL_ID` and `BACKBOARD_API_KEY`. AWS credentials come from `aws configure` or environment variables.

Run: `.venv\Scripts\python -m horizon doctor` (from `pipeline\`)
Expected:
- `torch`, `transformers`, `ultralytics`, `viser` and `boto3` show `installed`
- `cuda: True (NVIDIA GeForce RTX 3060 Ti)`
- every secret shows `set`

- [ ] **Step 4: Run the pipeline on a real clip and inspect each artifact**

Run: `.venv\Scripts\python -m horizon all C:\path\to\match.mp4 --match-id demo --duration 10` (click the keypoints when the window opens)

Expected, step by step:
1. `init`: `source.mp4` at 1280x720 (or smaller) and under 25 MB.
2. `calibrate`: RMS ≤ 4 px, and the camera position printed a few metres high behind the near baseline (y < −12, z > 3). In `calibration_preview.jpg` the lines overlay the court.
3. `track`: a few dozen track ids (players, ball kids, umpire, crowd).
4. `identify`: `near: <name> -> track N (gemini)` and `far: … (gemini)`. A `(heuristic)` source is acceptable but read the warnings.
5. `players`: statures between 1.6 and 2.1 m, detections in most frames, plausible distance and top speed (10–30 km/h).
6. `reconstruct`: over 50k background points and several thousand player points per frame. In `ground.png` a top-down court is visible with the far baseline at the top.
7. `render`: `pov_near.mp4` shows the court and the far player centred ahead. `pov_far.mp4` looks back at the near player. Both have exactly as many frames as `source.mp4`.
8. `export`: `public\matches\demo\` contains `main.mp4`, `pov_near.mp4`, `pov_far.mp4`, `manifest.json` and `tracks.json`, and `public\matches\index.json` lists `demo`.

If a step fails, fix its cause and re-run only that step (`horizon <step> --match-id demo`).

- [ ] **Step 5: Verify the free-camera viewer**

Run: `.venv\Scripts\python -m horizon view --match-id demo` and open http://localhost:8080.
Check each item:
- [ ] The court is textured under the point cloud. The stands and net appear as points, and the players move while `Playing` is on.
- [ ] A grey frustum sits at the broadcast camera. Blue and orange frustums ride on the players' heads and point at each other.
- [ ] **Broadcast camera** reproduces the original framing. **Near/Far player's eyes** match the POV videos.
- [ ] Dragging the white gizmo moves the free-camera frustum. **Look through free camera** shows that viewpoint.
- [ ] **Snap to near player** jumps the gizmo to the near player's eyes. `Follow = far chase` keeps it behind the far player while playing.
- [ ] **Render preview** shows an image in the panel.
- [ ] **Export free-cam clip** reports `saved free_cam.mp4 (N frames)`. Then `horizon export --match-id demo` sets `"freeCam": "free_cam.mp4"` in the manifest.

- [ ] **Step 6: Verify the web app**

Keep `horizon view` running. From the repo root, run `npm run dev` and open http://localhost:5173.
Check each item:
- [ ] The match list shows `<near> v <far>` with REPLAY. Clicking it goes through the 6 s transition to the stream page.
- [ ] The scoreboard shows both surnames and the Pegasus score (or `vs`).
- [ ] Two POV cards float above the players' heads and follow them, including after a window resize. Their previews stay in sync when you seek with the controls.
- [ ] Clicking a card opens the POV panel with live km/h, top km/h and metres.
- [ ] **Free cam** opens the panel with the live Viser viewer. The **Exported clip** tab plays `free_cam.mp4` in sync.

- [ ] **Step 7: Run the automated suites one final time**

```powershell
Set-Location pipeline; .venv\Scripts\python -m pytest -q; Set-Location ..
npm test
npm run build
npm run lint
```
Expected: `83 passed`, `18 passed`, a successful build, and no lint errors.

- [ ] **Step 8: Commit**

```powershell
git add -A README.md backend preprocessing
git commit -m "docs: tennis README; remove superseded soccer pipeline"
```

