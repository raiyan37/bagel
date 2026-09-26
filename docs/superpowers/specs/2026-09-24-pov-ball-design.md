# POV Ball: Design

**Date:** 2026-09-24
**Status:** Draft for review
**Extends:** `docs/superpowers/specs/2026-09-20-tennis-horizon-design.md` (base spec). Ball tracking is listed
there as a v1 non-goal (§3); this document lifts that one restriction and leaves every other part of the base
spec in force.
**Implements into:** `C:\Users\rhaqu\Desktop\ProjectHorizon-Tennis`

## 1. The ask

The player POV clips (`pov_near.mp4`, `pov_far.mp4`) currently show the court, the background and the opposing
player, but no ball. A tennis POV without a ball reads as a still scene. This adds the moving ball to both POV
clips, at the right place in 3D, so it grows as it approaches, shrinks as it leaves, and is hidden when the
ground or the opponent is in front of it.

## 2. Why this is solvable

The broadcast camera is static and calibrated, so every detected ball pixel gives one world-space ray. A ray
fixes two of the ball's three degrees of freedom; the depth along it is unknown. Neither of the pipeline's
existing depth sources helps: Depth Anything V2 cannot resolve a 3-pixel object, and the ball is not on the
ground plane, so `ground_intersection` is wrong for it by metres.

Gravity supplies the missing constraint. Between one contact (racket or bounce) and the next, the ball is a
projectile:

```
X(t) = A + V·t + ½·G·t²        G = (0, 0, −9.81) m/s²
```

Six unknowns, `A` and `V`. Each observed frame contributes two collinearity equations. Three observations make
the system determined; a real flight gives 10–30, so it is heavily overdetermined and noise averages out.

The useful part is that `X(t)` is **affine** in `(A, V)`. Writing the projection of `X(t)` with `P = K[R|t]`,
`M = P[:, :3]`, `c = P[:, 3]` and the known term `k_i(t) = M_i·(½G t²) + c_i`:

```
u·(M₂·X + c₂) = (M₀·X + c₀)
  ⟹  (u·M₂ − M₀)·A  +  t·(u·M₂ − M₀)·V  =  k₀(t) − u·k₂(t)
```

and the same for `v` with rows 1 and 2. Both are **linear** in the six unknowns. The whole trajectory fit is one
`lstsq` call — no optimiser, no initial guess, no iteration count to tune, and the result is deterministic and
exactly testable against a synthetic trajectory.

Clearing the denominator minimises algebraic rather than reprojection error, which over-weights far
observations. Two reweighting passes with `w = 1/z` from the previous estimate recover an approximate
reprojection-error solution at negligible cost.

## 3. Approach

```
source.mp4 ──┐
detections   ├──▶ detect ──▶ link ──▶ split ──▶ fit ──▶ ball.json ──▶ render ──▶ pov_near.mp4
players.json │    blobs     tracklet  flights  lstsq   (T,3) world              pov_far.mp4
calibration ─┘    per frame  in 2D    at        per
                                      residual  flight
                                      spikes
```

1. **Detect** (2D). Difference each frame against a masked-median clean plate — the same `median_plate` the
   reconstruct stage already uses — and keep small connected components that are also yellow-green
   (`G − B` well above zero). The Australian Open court is blue, so `G − B` separates a yellow ball from the
   court, the lines and the players by a wide margin. Player pixels are excluded with a dilated mask, which
   also removes the ball at the moment of contact; the trajectory fit interpolates through that gap.
2. **Link** (2D). Grow tracklets greedily with a constant-velocity prediction, tolerating short gaps, and keep
   the longest. One clip is one rally, so one tracklet is the ball.
3. **Split.** A single parabola cannot span a bounce or a racket hit. Fit the whole tracklet, and where the
   reprojection residual is too large, cut at the worst frame and recurse. Bounces and hits fall out as the cut
   points; no explicit bounce detector is needed.
4. **Fit.** Per segment, solve the linear system above, then reject segments that are not physically possible
   (below the ground, above 15 m, outside the court box, faster than 90 m/s). A rejected segment leaves those
   frames without a ball rather than drawing a wrong one.
5. **Render.** Expand the per-frame ball centre into a small sphere of points and hand them to the existing
   splat renderer alongside the background and player points. Occlusion, perspective size and the z-buffer all
   come for free from `render_view`.

## 4. Non-goals

* Aerodynamic drag and the Magnus effect. Gravity alone is accurate to a few centimetres over a single flight
  at these clip lengths, which is far below the size of the rendered ball.
* Motion blur on the rendered ball.
* The ball in the Viser free camera, in `free_cam.mp4`, in the web app overlays, or reprojected onto the
  broadcast video as a debug artifact. The renderer is shared, so the free camera is a small follow-on if it is
  ever wanted, but it is out of scope here.
* Ball speed, bounce-location or in/out statistics in the UI.
* Multi-ball clips, ball kids' balls, and clips containing more than one rally.

## 5. Constraints

* Inherits every global constraint of the base spec: Windows PowerShell 5.1, Python `>=3.11,<3.13` run as
  `.venv\Scripts\python` from `pipeline\`, no GPU/network/API keys in automated tests, OpenCV camera
  convention, world frame in metres with +Z up.
* No new runtime dependency. Detection uses OpenCV and NumPy, both already required.
* The existing artifacts (`scene.npz`, `players.json`, `detections.json`) keep their current format. The ball
  is a new, separate artifact, and rendering treats it as optional so a match without `ball.json` renders
  exactly as it does today.
* Ball radius is the ITF value, 0.0335 m.

## 6. Data contract

### `pipeline/data/<id>/ball.json`
```json
{
  "fps": 25.0,
  "frame_count": 250,
  "reprojection_rms_px": 1.83,
  "segments": [{ "start": 12, "end": 41 }, { "start": 44, "end": 70 }],
  "xyz": [null, null, [1.204, -3.417, 1.108], "…one entry per frame…"]
}
```
`xyz[i]` is the ball's world position in metres on frame `i`, or `null` when no segment covers that frame.
`segments` are inclusive frame ranges, each one flight between contacts. `reprojection_rms_px` is over all
observations used by the accepted segments, and is the number to look at when judging whether a run worked.

## 7. Known limitations

* Recall drops when the ball crosses a yellow-green background region (a sponsor board, some clay surrounds).
  The `G − B` gate is tuned for a blue hard court.
* The ball is hidden for frames with no accepted segment, including the contact frames swallowed by the player
  mask dilation. It pops in and out rather than stopping.
* A serve toss is nearly vertical and short, so it is often rejected as too few observations.
* Very fast frames blur the ball into a long streak whose centroid lags the true position by up to half a
  frame of travel; this biases the fit slightly along the direction of flight.
* One rally per clip. A clip with two rallies keeps only the longer one.
