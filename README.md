<h1 align="center">Bagel</h1>

<p align="center"><em>A new way to view professional tennis.</em></p>

<!-- <p align="center"><a href="DEMO_URL">Watch the demo</a></p> -->

---

## What this is

Bagel turns a single broadcast tennis clip into something you can watch from *inside* the point: follow a rally through either player's eyes, or put a free camera anywhere on court.

The repo ships with a processed sample match (`ao`), so you can try everything without a GPU or API keys.

Works on **macOS** and **Windows**. Commands are the same on both unless a step shows separate versions. Use Terminal on macOS and PowerShell on Windows.

## Required vs optional

**To get Bagel running, you only need the Quick start below.** Everything else is optional.

| | What | Needed for |
|---|---|---|
| ✅ **Required** | [Quick start](#quick-start-required) steps 1–3: install tools, install dependencies, run the viewer and the web app | Watching the sample match: POV cards, player views and the 3D free camera |
| ⬜ Optional | [Exporting a free-camera clip](#using-the-3d-viewer) | Saving a video of your own camera path |
| ⬜ Optional | [Process your own clip](#process-your-own-clip-optional) | Turning a new broadcast video into a match. Needs PyTorch, and API keys for automatic player identification |
| ⬜ Optional | [Scripts](#scripts) (build, lint, tests) | Developing Bagel itself |

## Quick start (required)

About 5 minutes. **Every step in this section is required.**

### 1. Install the tools (required)

You need [Git](https://git-scm.com/downloads), [Node.js 24](https://nodejs.org/) and [uv](https://docs.astral.sh/uv/). uv is a Python package manager that downloads Python for you.

**macOS**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close and reopen your terminal afterwards so `uv` is found.

### 2. Get the code and install dependencies (required)

Run this once, from wherever you keep projects:

```bash
git clone https://github.com/raiyan37/bagel.git
cd bagel
npm install
cd pipeline
uv venv --python 3.11 .venv
uv pip install -e ".[viewer]"
cd ..
```

### 3. Run it (required)

You need **two terminals**, both opened in the `bagel` folder, and both must stay running.

**Terminal 1: 3D viewer (required for the Free camera)**

```bash
cd pipeline
uv run --no-sync horizon view --match-id ao
```

Wait for `Viewer running at http://localhost:8080`, then leave it running.

**Terminal 2: web app (required)**

```bash
npm run dev
```

Open **http://localhost:5173** and pick the match. The **Free camera** button shows the 3D viewer inside the app; it stays black if terminal 1 isn't running.

Press `Ctrl+C` in each terminal to stop.

**That's it: Bagel is fully working.** Everything below is optional.

> **Why `uv run --no-sync`?** It runs the pipeline's own Python environment (`pipeline/.venv`) without you having to activate it, and `--no-sync` stops uv from changing what's installed. Run all `horizon` commands from the `pipeline` folder.

## Using the 3D viewer

- **Views** jumps to the broadcast camera or either player's eyes.
- Drag the white gizmo to place the free camera, or use **Snap to near/far player**.
- **Follow** attaches the camera to a player: `eyes` is first person, `chase` is behind and above.
- **Look through free camera** shows its view live.
- **Export clip** *(optional)* writes `free_cam.mp4`. To show it in the web app's Free camera panel, run this from the `pipeline` folder:

```bash
uv run --no-sync horizon export --match-id ao
```

## Process your own clip (optional)

**Skip this section unless you want to use your own video.** The sample match already works without it.

An NVIDIA GPU is strongly recommended; on a Mac, processing runs on the CPU and is slow, but the results are the same.

| Step | Status |
|---|---|
| Install PyTorch + ML packages | ✅ Required for processing |
| API keys | ⬜ Optional: without them, add `--orchestrator none`; players are picked by court position and named "Near player" / "Far player" |
| `horizon doctor` | ⬜ Optional: checks your setup |
| Run the pipeline | ✅ Required for processing |

### 1. Full install (required for processing)

From the `pipeline` folder, install PyTorch for your machine first:

**macOS, or Windows without an NVIDIA GPU**

```bash
uv pip install torch torchvision
```

**Windows with an NVIDIA GPU** (CUDA 12.8)

```powershell
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Then, on either OS:

```bash
uv pip install -e ".[ml,viewer,dev]"
```

### 2. API keys (optional)

Skip this if you don't have keys; use `--orchestrator none` in step 3 instead. Keys are only used for automatic player identification:

| Service | Used for | Setting in `pipeline/.env` |
|---|---|---|
| AWS Bedrock, with model access to **TwelveLabs Pegasus 1.2** in `us-east-1` | Video understanding | AWS credentials via `aws configure`, plus `AWS_REGION` |
| [OpenRouter](https://openrouter.ai/) | Gemini | `OPENROUTER_API_KEY` |
| Backboard *(optional even with keys)* | Orchestration | `BACKBOARD_API_KEY` |

With AWS and OpenRouter but no Backboard key, use `--orchestrator direct` in step 3.

Create your settings file:

**macOS**

```bash
cp .env.example .env
```

**Windows**

```powershell
Copy-Item .env.example .env
```

Fill in the keys in `pipeline/.env`. Optionally, check the setup:

```bash
uv run --no-sync horizon doctor
```

### 3. Run the pipeline (required for processing)

Use one continuous shot from the main broadcast camera: **3–20 s, both players visible, no cuts or zoom.** Pick any match id (lowercase letters, digits, `-`); `myclip` is used below. If you skipped step 2, add `--orchestrator none` to the end of the command.

**macOS**

```bash
uv run --no-sync horizon all ~/Movies/match.mp4 --match-id myclip --start 12 --duration 10
```

**Windows**

```powershell
uv run --no-sync horizon all C:\path\to\match.mp4 --match-id myclip --start 12 --duration 10
```

A calibration window opens on the first frame. Click each keypoint named at the top; the red dot on the mini-court shows where it is.

| Key | Action |
|---|---|
| `S` | Skip a keypoint you can't see |
| `U` | Undo the last click |
| `Enter` | Solve (needs at least 4 points) |

Check `pipeline/data/myclip/calibration_preview.jpg`: the red lines should sit on the painted court lines.

When it finishes, the match appears in the web app. View it in 3D with the same id:

```bash
uv run --no-sync horizon view --match-id myclip
```

Re-running with the same id reuses the saved `calibration.json`. Pass `--recalibrate` if the clip comes from a different camera, angle or match.

*(Optional)* To run steps one at a time instead of `all`: `init`, `calibrate`, `track`, `identify`, `players`, `ball`, `reconstruct`, `render`, `export`. See `uv run --no-sync horizon <step> --help`.

## How it works

The offline Python pipeline (`pipeline/`):

1. **Court calibration**: solves the broadcast camera from clicked court keypoints.
2. **Tracking**: YOLO26-seg + ByteTrack follow every person in frame.
3. **Player identification**: TwelveLabs Pegasus and Gemini, orchestrated by Backboard, pick out the two players.
4. **3D reconstruction**: Depth Anything V2 lifts the scene into 3D, aligned to the court geometry.
5. **Rendering**: first-person POV videos for both players.

The **3D viewer** (`horizon view`, built on Viser) serves a free camera at `http://localhost:8080`. The **web app** (`src/`, React + Vite) plays the match back with POV cards pinned above the players and embeds the viewer.

Design doc: [`docs/superpowers/specs/2026-09-20-tennis-horizon-design.md`](docs/superpowers/specs/2026-09-20-tennis-horizon-design.md)

## Tech stack

| Layer | Tools |
|---|---|
| Web app | React 19, TypeScript, Vite 7, React Router 7, Vitest |
| Computer vision | YOLO26-seg, ByteTrack, OpenCV, Depth Anything V2 (PyTorch + Transformers) |
| AI / video understanding | TwelveLabs Pegasus 1.2 (Amazon Bedrock), Gemini (OpenRouter), Backboard |
| 3D viewer | Viser |
| Tooling | npm, uv, ESLint, pytest |

## Scripts

Only `npm run dev` is needed to use Bagel, and it's already covered in the Quick start. The rest are for development.

| Command | Status | Run from | What it does |
|---|---|---|---|
| `npm run dev` | ✅ Required | `bagel` | Start the web app |
| `npm run build` | ⬜ Optional | `bagel` | Type-check and build for production |
| `npm run lint` | ⬜ Optional | `bagel` | Lint the web app |
| `npm test` | ⬜ Optional | `bagel` | Web app tests |
| `uv run --no-sync pytest -q` | ⬜ Optional | `pipeline` | Pipeline tests (needs the full install) |

## Troubleshooting

| Symptom | Fix |
|---|---|
| `No such file or directory: .../data/<id>/scene.npz` | That match hasn't been processed. Use `--match-id ao` for the sample, or run the pipeline for your id first |
| `No module named horizon` or `horizon` not found | You're not in the `pipeline` folder, or step 2 of the quick start hasn't been run |
| `uv: command not found` / `uv` is not recognized | Reopen the terminal after installing uv |
| Free camera panel is black | Start the 3D viewer (terminal 1) and wait for `Viewer running` |
| Port 8080 already in use | Another viewer is running; close it, or add `--port 8081` |
| macOS: calibration window doesn't appear | Check behind other windows or in the Dock; it opens as a separate Python window |
| `yolo26s-seg.pt` not found | `uv run --no-sync horizon track --match-id myclip --model yolo11s-seg.pt` |
| OpenRouter 404 / model not found | Set `GEMINI_MODEL_ID` in `pipeline/.env` to a current id from [openrouter.ai/google](https://openrouter.ai/google) |
| Pegasus `AccessDeniedException` | Enable TwelveLabs Pegasus under Bedrock → Model access for your `AWS_REGION` |
| Pegasus rejects the clip (>25 MB) | Re-run `horizon init` with a shorter `--duration` or `--max-height 540` |
| Backboard errors | `uv run --no-sync horizon identify --match-id myclip --orchestrator direct` |
| Wrong player chosen | Check warnings in `identity.json`; re-run `identify --frame N` on a frame where both players are clearly visible |
| Calibration RMS > 4 px | `uv run --no-sync horizon calibrate --match-id myclip` and click more keypoints (service-line T's help) |
| Sparse point cloud | `uv run --no-sync horizon reconstruct --match-id myclip --stride 1 --depth-model depth-anything/Depth-Anything-V2-Base-hf` |

## Known limitations

A single camera can't see everything:

- The far player's POV looks back toward the broadcast camera, where nothing was filmed, so that area is filled with sky or a flat colour.
- Players are 2.5D, like billboards.

See §7 of the design doc.

## Accessibility

High-contrast mode, reduced motion, full keyboard navigation and ARIA live announcements. The app respects the OS-level `prefers-reduced-motion` and `prefers-contrast` settings. See [`accessibility.md`](accessibility.md).
