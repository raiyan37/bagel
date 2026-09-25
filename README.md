<div align="center">
  <img
    src="src/assets/project-horizon-logo.svg"
    width="200"
    alt="Project Horizon"
  />

  <h1>Project Horizon</h1>
  <em>You can be anyone.</em>
  <br /><br />

  <a href="https://youtu.be/Wcb6SSw7LFY">Watch the demo</a>
</div>

## What this is

Project Horizon is a proof-of-concept for **first-person sports viewing**: taking a normal broadcast clip and giving fans a player's-eye view of the action, on top of the usual scoreboard/highlight experience.

The demo clip is a Premier League penalty shootout (Bruno Fernandes vs. David Raya). The web app plays the broadcast footage with a live scoreboard and goal celebration, while overlaying computer-vision-tracked player cards that expand into synced point-of-view video for each player on the pitch.

## How it works

**Frontend** — React 19 + TypeScript + Vite, with React Router for the flow between a match list, a "processing" screen, and the live stream view.

**Player tracking** — A Python backend (`backend/track_video.py`) runs YOLOv8 object detection with ByteTrack multi-object tracking over the broadcast video, frame-index labels each player, and exports normalized bounding-box coordinates as JSON. The frontend (`usePlayerTracking` hook) reads that track data and positions floating player cards over the video in real time as it plays.

**AI scene understanding** — A separate preprocessing pipeline (`preprocessing/`) uses TwelveLabs' Pegasus video-understanding model (via Amazon Bedrock) to describe the clip and identify the main subject, and Gemini to localize that person in frame — early experiments toward automating what's currently hand-tuned player/track mapping.

**Accessibility** — first-class support rather than an afterthought: a high-contrast mode, a reduced-motion mode, full keyboard navigation with visible focus states, and ARIA live-region announcements for in-game events (e.g. goals), all respecting the user's OS-level `prefers-reduced-motion` / `prefers-contrast` settings and persisting choices in `localStorage`. See [`accessibility.md`](accessibility.md).

## Tech stack

| Layer | Tools |
|---|---|
| Frontend | React 19, TypeScript, Vite 7, React Router 7 |
| Computer vision | YOLOv8, ByteTrack, OpenCV, Ultralytics |
| AI / video understanding | TwelveLabs Pegasus (Amazon Bedrock), Google Gemini |
| Tooling | ESLint, npm |

## Status

This is an active prototype exploring how far a broadcast-only pipeline (no dedicated multi-camera rig) can go toward immersive, player-perspective sports content. The soccer demo above is functional end-to-end; a follow-on design (see [`docs/superpowers/`](docs/superpowers)) extends the same idea to tennis with full 3D scene reconstruction (monocular depth estimation, court calibration, and a free-roaming virtual camera).

## Running locally

```bash
npm install
npm run dev
```
