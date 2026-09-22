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
