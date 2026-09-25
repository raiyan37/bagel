# Project Horizon: Tennis Spin-off — Design

**Date:** 2026-09-20
**Status:** Draft for review
**Implements into:** a new sibling project `C:\Users\rhaqu\Desktop\ProjectHorizon-Tennis` (a copy of this repo)

## 1. What exists today (soccer demo)

| Area | Files | What it really does |
|---|---|---|
| Web UI | `src/` (React 19 + Vite 7 + react-router 7) | `GamesPage` (hard-coded fixtures) → `ProcessingPage` (fake 30 s progress bar) → `StreamPage` (plays `src/assets/bruno.mov`, overlays penalty scoreboard, confetti at t=8 s, POV cards). |
| Player positions | `backend/track_video.py` → `public/bruno_tracks.json` → `src/hooks/usePlayerTracking.ts` | YOLOv8n + ByteTrack person boxes per frame. The hook maps ByteTrack IDs to people through a hand-written `TRACK_TO_PLAYER` table and turns box centres into % coordinates for card placement. |
| POV videos | `src/assets/*_pov.mov` | Pre-rendered clips made outside this repo. `PlayerCard`/`POVVideoOverlay` sync them with hand-tuned `POV_CONFIG` offsets. `goalie_pov.mov`, `goalie_upper.mov`, `ref_pov.mov` are imported but missing. |
| AI analysis | `preprocessing/script.py`, `preprocessing/sponsor.py` | TwelveLabs Pegasus (via AWS Bedrock) describes the "main character" and the clip. Gemini (via OpenRouter) returns an (x, y) for that person in frame 0. Backboard wraps both as assistant tools. The output (`full_stats.json`) is **not consumed** by the UI. |
| 3D | none | No depth model, point cloud or Viser code exists in the repo. The POV clips were produced by an external, uncommitted workflow. |

Evidence that shapes the new design:
* `full_stats.json` contains `{"x": 164, "y": 721}` for an 854×480 frame. Gemini answered in its native 0–1000 normalised space, not pixels. The new code asks for `box_2d` in 0–1000 and converts.
* `extract_description()` in `script.py` returns the raw JSON after checking only the first key (indentation bug). The new Pegasus client uses Bedrock structured output (`responseFormat.jsonSchema`) and parses `message`.
* ByteTrack IDs fragment (the goalkeeper is IDs 21 and 103). The new code re-associates tracks geometrically on the court.

## 2. Goals

1. Ingest a single broadcast tennis clip and reconstruct it in 3D with a **monocular depth model** (Depth Anything V2).
2. Produce **first-person POV videos for both players**, frame-aligned with the broadcast clip. The near player looks at the far player and vice versa.
3. Provide a **free-placement camera**: a draggable virtual camera in the 3D scene that can snap to or follow either player. It can be looked through live and exported as a clip.
4. Use **TwelveLabs Pegasus + Gemini, orchestrated through Backboard**, to identify who each player is and where they are. Geometry (court calibration + tracking + depth) makes that placement metrically accurate.
5. A tennis web experience that reuses the Horizon look: match list, scoreboard, two POV cards pinned above the players, an expanded POV panel, and a FREE CAM view.

## 3. Non-goals (v1)

* Live streaming. The pipeline is offline, per clip.
* Clips with cuts, replays, zooms or moving cameras. The broadcast camera must be static for the whole clip.
* Doubles matches, ball tracking, automatic court-line detection, true novel-view synthesis of unseen surfaces (Gaussian splatting / diffusion inpainting).

## 4. Input constraints

* One continuous shot from the main broadcast camera behind a baseline. Both players visible. 3–20 s.
* Any resolution. `horizon init` trims, downscales to ≤720 p and re-encodes to H.264.
* The encoded clip must be ≤25 MB, the Bedrock limit for inline Pegasus video. 20 s of 720p at CRF 20 is ~5–10 MB.

## 5. Architecture

```
            ┌───────────────────────────── pipeline/ (Python 3.11) ─────────────────────────────┐
 raw clip → │ init → calibrate → track → identify → players → reconstruct → render → export    │ → public/matches/<id>/
            │  H.264   court     YOLO26   Pegasus+   court     Depth Any-   POV       manifest │
            │  meta    clicks→   seg +    Gemini via trajecto- thing V2 +   videos    + tracks │
            │          camera    ByteTrack Backboard  ries      point cloud  (splat)          │
            └────────────────────────────────────────────┬──────────────────────────────────────┘
                                                         │ data/<id>/scene.npz, players.json, calibration.json
                                   horizon view <id>  ←──┘   (Viser server :8080: point cloud, POV frustums,
                                                             draggable free camera, snap/follow, clip export)
 React app (Vite :5173): /  →  /processing/:id  →  /stream/:id   (reads public/matches/<id>/manifest.json,
                         tracks.json, main.mp4, pov_near.mp4, pov_far.mp4, free_cam.mp4; FREE CAM = iframe to Viser)
```

### 5.1 Coordinate conventions

* **World:** metres, origin at the centre of the court on the ground. +X runs across the court (to the right as seen from the broadcast camera), +Y runs towards the far baseline, +Z is up. This matches Viser's default +Z-up world.
* **Cameras:** OpenCV/COLMAP convention (+X right, +Y down, +Z forward), `x_cam = R · x_world + t`. Viser uses the same camera convention, so a camera's scene-node pose is `(wxyz(Rᵀ), C = −Rᵀt)`.
* **Pixels:** OpenCV convention. Pixel centres are at integer coordinates and the principal point is at `((w−1)/2, (h−1)/2)`.
* **Player roles:** `near` (court half y<0, closest to the camera) and `far` (y>0). The UI colours are near `#3B82F6` and far `#F97316`.

### 5.2 Pipeline stages and artifacts (`pipeline/data/<match-id>/`)

| Stage | Command | Output | Method |
|---|---|---|---|
| init | `horizon init <video> --match-id <id> [--start s] [--duration s]` | `source.mp4`, `meta.json` | OpenCV decode → resize → imageio-ffmpeg libx264, yuv420p, faststart |
| calibrate | `horizon calibrate --match-id <id>` | `calibration.json`, `calibration_preview.jpg` | Click ≥4 of 14 named court keypoints. Planar PnP (IPPE) with a 1-D focal-length search; principal point fixed, no distortion. |
| track | `horizon track --match-id <id>` | `detections.json` | Ultralytics `yolo26s-seg.pt` + ByteTrack, person class only. Boxes + simplified mask polygons. |
| identify | `horizon identify --match-id <id> [--orchestrator backboard\|direct\|none]` | `identity.json` | Pegasus (names, clothing, score, summary) → Gemini (`box_2d` per player on frame 0) → IoU match to tracks, validated against the court half. Backboard runs the two tools via tool calling. Geometric fallback. |
| players | `horizon players --match-id <id>` | `players.json` | Per-frame association on the court plane, gap interpolation, smoothing, stature from box top (single-view metrology), speed and distance |
| reconstruct | `horizon reconstruct --match-id <id>` | `scene.npz`, `ground.png` | Masked-median clean plate. Depth Anything V2 disparity aligned to the calibrated ground plane, ground/non-ground split, ground orthophoto texture, static background points, per-frame player points anchored at the calibrated foot depth. |
| render | `horizon render --match-id <id>` | `pov_near.mp4`, `pov_far.mp4` | Analytic textured ground plane + z-buffered adaptive point splatting from each player's eye towards the opponent |
| export | `horizon export --match-id <id>` | `public/matches/<id>/…`, `public/matches/index.json` | Copies media, writes the manifest, per-frame card anchors and stats |
| view | `horizon view --match-id <id>` | Viser at `http://localhost:8080` | Free camera: transform-controls gizmo + frustum, snap/follow player, look-through, preview render, export `free_cam.mp4` |

### 5.3 Algorithms (summary)

* **Calibration.** For a candidate focal length f, `solvePnPGeneric(IPPE)` gives up to two planar poses. Keep the one with the camera above the ground and the lowest RMS. Search f over `geomspace(0.3w, 8w, 240)`, then refine with golden-section. Accept if RMS ≤ 4 px.
* **Ground intersection.** Rays are scaled so camera-z = 1. Then `depth = −C_z / d_z`, and that value is directly the camera-z depth of the hit point.
* **Stature.** For foot (X, Y) and box top row v, solve `v = (a + b·z)/(c + d·z)` for z using rows 2 and 3 of `P = K[R|t]`. Stature is the median over the clip. Eye height is 0.94 × stature.
* **Depth alignment.** Depth Anything V2 (relative) outputs affine-invariant disparity. Fit `s·d + b ≈ 1/z_plane` on court pixels (court rectangle + 2–3 m run-off, minus the net quad and players) by least squares, trimming the worst 20 % and refitting. Require s > 0. Depth is `1 / max(s·d + b, 1/max_depth)`.
* **Ground vs. non-ground.** A clean-plate pixel is ground if its ray hits z=0 inside the texture extent and `|z_aligned − z_plane| ≤ 12 % · z_plane`.
* **Ground texture.** A 2.5 cm/texel orthophoto over X∈[−10, 10], Y∈[−20, 20], sampled from the clean plate by projecting texel centres. Texels whose source pixel is not ground get the median surround colour.
* **Players.** Pixels are rasterised from the mask polygon. Aligned depth is shifted so the median depth of the bottom 10 % of mask rows equals the calibrated foot depth, then clamped to foot depth ±0.6 m.
* **Rendering.** Pass 1 casts every output pixel onto z=0 and samples the texture; above the horizon it draws a sky gradient. Pass 2 projects points, splats each over `ceil(2·r·f/z)` px (clamped to 1…7), where r is the point's world footprint, and depth-tests against the ground. The nearest point wins via sort + `np.unique`. The viewer's own points are excluded.
* **POV camera.** Eye = (smoothed foot XY, eye height). Target = (opponent foot XY, 1.0 m). Horizontal FOV 75°, 1024×576, near plane 0.3 m.

### 5.4 Role of each AI service

| Service | Input | Output used |
|---|---|---|
| TwelveLabs Pegasus 1.2 (Bedrock `us.twelvelabs.pegasus-1-2-v1:0`) | whole clip (base64 ≤25 MB) | Player names (if on screen or spoken), clothing descriptions, scoreboard text, point summary. Structured JSON via `responseFormat.jsonSchema`. |
| Gemini (OpenRouter, default `google/gemini-3.5-flash`, env `GEMINI_MODEL_ID`) | frame 0 JPEG + both descriptions | `box_2d` per player → which ByteTrack ID is which player |
| Backboard (`app.backboard.io/api`) | assistant "Horizon Tennis Analyst" with tools `analyze_tennis_clip` and `locate_players` | Orchestrates the two tools, keeps memory across clips. Tool outputs are captured locally. The assistant's prose is not trusted. |

The LLM answers are advisory. The chosen tracks must sit on the expected court half. Otherwise the pipeline logs a warning and falls back to the geometric heuristic: the most persistent track inside each half.

## 6. Data contracts

### `public/matches/index.json`
```json
{ "matches": [ { "id": "demo", "title": "Near Player vs Far Player", "near": "Near Player", "far": "Far Player",
                 "competition": "Tennis", "status": "replay" } ] }
```

### `public/matches/<id>/manifest.json`
```json
{
  "id": "demo", "title": "A vs B", "competition": "Tennis", "summary": "…", "score": "6-4 3-2 30-15",
  "fps": 25.0, "frameCount": 250, "width": 1280, "height": 720,
  "video": "main.mp4", "tracks": "tracks.json",
  "players": [
    { "id": "near", "name": "A", "description": "…", "color": "#3B82F6", "pov": "pov_near.mp4", "statureM": 1.85 },
    { "id": "far",  "name": "B", "description": "…", "color": "#F97316", "pov": "pov_far.mp4",  "statureM": 1.88 }
  ],
  "freeCam": "free_cam.mp4",
  "viewerUrl": "http://localhost:8080"
}
```
`freeCam` is `null` when no free-camera clip was exported.

### `public/matches/<id>/tracks.json` (columnar, one entry per frame)
```json
{ "fps": 25.0, "frameCount": 250,
  "players": { "near": { "x": [..], "y": [..], "visible": [1,0,..], "speedKmh": [..], "distanceM": [..] },
               "far":  { … } } }
```
`x`/`y` are the head anchor (box top-centre) in % of the video frame. The UI maps them through the `object-fit: cover` transform.

## 7. Known limitations

* Single view: surfaces the broadcast camera never saw are missing. The far player's POV looks back towards the camera, so the area behind the near baseline is sky/fill colour, and players are 2.5-D "billboards". Their front sides show the colours of their backs.
* Ground texture resolution degrades with distance from the broadcast camera.
* Stature/eye height is one number per player. Jumps and crouches do not move the POV camera.
* Players must not change ends within the clip.

## 8. Testing strategy

* Python: pytest, no GPU and no network. Synthetic cameras, synthetic scenes, fake depth estimators, and fake HTTP/Bedrock clients cover the math and parsing. The ML-backed steps (`track`, `reconstruct` with the real model) get a manual smoke run in the final task.
* Web: Vitest + Testing Library (jsdom) for pure helpers (frame lookup, cover transform, sync decisions) and the new components, driven by a fixture manifest and tracks file.
