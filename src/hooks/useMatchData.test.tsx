import { renderHook, waitFor } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { mockFetch, sampleIndex, sampleManifest, sampleTracks } from '../test/fixtures';
import { useMatchData, useMatchIndex } from './useMatchData';

describe('useMatchData', () => {
  it('loads the manifest and then its tracks file', async () => {
    const fetchMock = mockFetch({
      '/matches/demo/manifest.json': sampleManifest,
      '/matches/demo/tracks.json': sampleTracks,
    });
    const { result } = renderHook(() => useMatchData('demo'));
    await waitFor(() => expect(result.current.tracks).not.toBeNull());
    expect(result.current.manifest?.title).toBe('Carlos Alcaraz vs Jannik Sinner');
    expect(result.current.error).toBeNull();
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it('reports a missing match', async () => {
    mockFetch({});
    const { result } = renderHook(() => useMatchData('nope'));
    await waitFor(() => expect(result.current.error).toMatch(/404/));
    expect(result.current.manifest).toBeNull();
  });

  it('does not serve the previous match after the id changes', async () => {
    mockFetch({
      '/matches/demo/manifest.json': sampleManifest,
      '/matches/demo/tracks.json': sampleTracks,
    });
    const { result, rerender } = renderHook(({ id }) => useMatchData(id), { initialProps: { id: 'demo' } });
    await waitFor(() => expect(result.current.manifest).not.toBeNull());
    rerender({ id: 'other' });
    expect(result.current.manifest).toBeNull();
    expect(result.current.tracks).toBeNull();
    expect(result.current.error).toBeNull();
  });
});

describe('useMatchIndex', () => {
  it('lists exported matches', async () => {
    mockFetch({ '/matches/index.json': sampleIndex });
    const { result } = renderHook(() => useMatchIndex());
    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.matches.map((m) => m.id)).toEqual(['demo']);
  });
});
