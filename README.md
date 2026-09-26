<h1 align="center">Bagel</h1>

<p align="center"><em>A new way to view professional tennis.</em></p>

<!-- <p align="center"><a href="DEMO_URL">Watch the demo</a></p> -->

---

## What this is

Bagel turns a single broadcast tennis clip into something you can watch from *inside* the point: follow a rally through either player's eyes, or put a free camera anywhere on court.

## How it works

An offline Python pipeline (`pipeline/`) processes one broadcast clip:

1. **Court calibration**: you click a few court keypoints; the pipeline solves the broadcast camera.
2. **Tracking**: YOLO26-seg + ByteTrack follow every person in frame.
3. **Player identification**: TwelveLabs Pegasus (Amazon Bedrock) and Gemini (OpenRouter), orchestrated by Backboard, pick out the two players.
4. **3D reconstruction**: Depth Anything V2 lifts the scene into 3D, aligned to the court geometry.
5. **Rendering**: first-person POV videos are rendered for both players.

Two apps show the results:

- **3D viewer** (`horizon view`, built on Viser): a free-placement camera at `http://localhost:8080`.
- **Web app** (`src/`, React 19 + Vite): plays the match back with POV cards pinned above the players and embeds the 3D viewer in its Free camera panel.

Design doc: [`docs/superpowers/specs/2026-09-20-tennis-horizon-design.md`](docs/superpowers/specs/2026-09-20-tennis-horizon-design.md)

## Tech stack

| Layer | Tools |
|---|---|
| Web app | React 19, TypeScript, Vite 7, React Router 7, Vitest |
| Computer vision | YOLO26-seg, ByteTrack, OpenCV, Depth Anything V2 (PyTorch + Transformers) |
| AI / video understanding | TwelveLabs Pegasus 1.2 (Amazon Bedrock), Gemini (OpenRouter), Backboard |
| 3D viewer | Viser |
| Tooling | npm, uv, ESLint, pytest |

## Requirements

**System**

- Windows 10/11
- NVIDIA GPU recommended (developed on an RTX 3060 Ti, 8 GB). CPU works but is slow.

**Software**

- [Node.js 24](https://nodejs.org/) + npm
- [Python 3.11](https://www.python.org/) (3.11 or 3.12 supported)
- [uv](https://docs.astral.sh/uv/)
- Git

**API keys** (only needed for player identification)

| Service | Used for | Env var |
|---|---|---|
| AWS Bedrock, with model access to **TwelveLabs Pegasus 1.2** in `us-east-1` | Video understanding | AWS credentials via `aws configure`, plus `AWS_REGION` |
| [OpenRouter](https://openrouter.ai/) | Gemini | `OPENROUTER_API_KEY` |
| Backboard *(optional)* | Orchestration | `BACKBOARD_API_KEY` |

No keys? Use `--orchestrator direct` to skip Backboard, or `--orchestrator none` to skip all AI services.

## Setup

### 1. Web app

```powershell
npm install
```

### 2. Pipeline

```powershell
cd pipeline
uv venv --python 3.11 .venv

# PyTorch with CUDA 12.8 (drop --index-url for a CPU-only install)
uv pip install --python .venv\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Pipeline + ML models + 3D viewer + test tools
uv pip install --python .venv\Scripts\python.exe -e ".[ml,viewer,dev]"
```

### 3. Environment

```powershell
Copy-Item .env.example .env
```

Fill in the keys in `pipeline\.env`, then check everything is wired up:

```powershell
.venv\Scripts\python -m horizon doctor
```

## Usage

### 1. Process a clip

Use one continuous shot from the main broadcast camera: **3–20 s, both players visible, no cuts or zoom.**

```powershell
cd pipeline
.venv\Scripts\python -m horizon all C:\path\to\match.mp4 --match-id demo --start 12 --duration 10
```

A calibration window opens on the first frame. Click each keypoint named at the top; the red dot on the mini-court shows where it is.

| Key | Action |
|---|---|
| `S` | Skip a keypoint you can't see |
| `U` | Undo the last click |
| `Enter` | Solve (needs at least 4 points) |

Then open `data\demo\calibration_preview.jpg` and check that the red lines sit on the painted court lines.

Re-running with the same `--match-id` reuses the saved `calibration.json`. Pass `--recalibrate` when the clip comes from a different camera, angle or match.

To run steps individually: `init`, `calibrate`, `track`, `identify`, `players`, `reconstruct`, `render`, `export`. See `python -m horizon <step> --help`.

### 2. Watch it

Run both apps side by side, one per terminal:

```powershell
# Terminal 1: 3D viewer -> http://localhost:8080
cd pipeline
.venv\Scripts\python -m horizon view --match-id demo
```

```powershell
# Terminal 2: web app -> http://localhost:5173
npm run dev
```

The web app's **Free camera** panel embeds the 3D viewer, so it stays blank until terminal 1 is running.

**In the 3D viewer**

- Drag the white gizmo to place the camera, or use **Snap to near/far player**.
- **Follow** attaches the camera to a player: `eyes` is first person, `chase` is behind and above.
- **Look through free camera** shows its view live.
- **Export clip** writes `free_cam.mp4`. Run `horizon export --match-id demo` to publish it to the web app.

## Scripts

| Command | What it does |
|---|---|
| `npm run dev` | Start the web app dev server |
| `npm run build` | Type-check and build for production |
| `npm run lint` | Lint the web app |
| `npm test` | Run web app tests (Vitest) |
| `cd pipeline; .venv\Scripts\python -m pytest -q` | Run pipeline tests |

## Troubleshooting

| Symptom | Fix |
|---|---|
| Free camera panel is blank | Start `horizon view --match-id <id>` in a second terminal |
| `yolo26s-seg.pt` not found | `horizon track --match-id demo --model yolo11s-seg.pt` |
| OpenRouter 404 / model not found | Set `GEMINI_MODEL_ID` in `pipeline\.env` to a current id from [openrouter.ai/google](https://openrouter.ai/google) |
| Pegasus `AccessDeniedException` | Enable TwelveLabs Pegasus under Bedrock → Model access for your `AWS_REGION` |
| Pegasus rejects the clip (>25 MB) | Re-run `horizon init` with a shorter `--duration` or `--max-height 540` |
| Backboard errors | `horizon identify --match-id demo --orchestrator direct` |
| Wrong player chosen | Check warnings in `identity.json`; re-run `identify --frame N` on a frame where both players are clearly visible |
| Calibration RMS > 4 px | `horizon calibrate --match-id demo` and click more keypoints (service-line T's help) |
| Sparse point cloud | `horizon reconstruct --match-id demo --stride 1 --depth-model depth-anything/Depth-Anything-V2-Base-hf` |

## Known limitations

A single camera can't see everything:

- The far player's POV looks back toward the broadcast camera, where nothing was filmed, so that area is filled with sky or a flat colour.
- Players are 2.5D, like billboards.

See §7 of the design doc.

## Accessibility

High-contrast mode, reduced motion, full keyboard navigation and ARIA live announcements. The app respects the OS-level `prefers-reduced-motion` and `prefers-contrast` settings. See [`accessibility.md`](accessibility.md).
