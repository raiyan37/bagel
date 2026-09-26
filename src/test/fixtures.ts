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
