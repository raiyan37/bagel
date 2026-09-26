import { useEffect, useMemo, useState } from 'react';
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

/** Stored state tagged with the match it was loaded for, so a stale match is never served. */
interface LoadedMatch extends MatchData {
  id: string | undefined;
}

const NOTHING_LOADED: MatchData = { manifest: null, tracks: null, error: null };

export function useMatchData(matchId: string | undefined): MatchData {
  const [loaded, setLoaded] = useState<LoadedMatch>({ id: undefined, ...NOTHING_LOADED });

  useEffect(() => {
    if (!matchId) return;
    let cancelled = false;
    const load = async () => {
      try {
        const manifest = await fetchJson<MatchManifest>(assetUrl(matchId, 'manifest.json'));
        const tracks = await fetchJson<TracksFile>(assetUrl(matchId, manifest.tracks));
        if (!cancelled) setLoaded({ id: matchId, manifest, tracks, error: null });
      } catch (err) {
        if (!cancelled) {
          setLoaded({ id: matchId, manifest: null, tracks: null, error: err instanceof Error ? err.message : String(err) });
        }
      }
    };
    void load();
    return () => {
      cancelled = true;
    };
  }, [matchId]);

  // Derived at render time: until the effect has loaded *this* id, report nothing rather than the
  // previous match. Memoised so the identity stays stable between renders, as it did before.
  return useMemo(
    () =>
      loaded.id === matchId
        ? { manifest: loaded.manifest, tracks: loaded.tracks, error: loaded.error }
        : NOTHING_LOADED,
    [loaded, matchId],
  );
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
