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
