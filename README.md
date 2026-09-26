# bagel

Watch the point from inside it: follow a rally from either player's eyes, or put the camera anywhere on court.

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

A match id that already has a `calibration.json` reuses it instead of asking for the clicks again, and prints which file it reused. Pass `--recalibrate` whenever the clip comes from a different camera, angle or match.

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
