import { fireEvent, render, screen } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { describe, expect, it } from 'vitest';
import { AccessibilityProvider } from '../contexts/AccessibilityContext';
import { mockFetch, sampleIndex, sampleManifest, sampleTracks } from '../test/fixtures';
import { GamesPage } from './GamesPage';
import { StreamPage } from './StreamPage';

function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <AccessibilityProvider>
        <Routes>
          <Route path="/" element={<GamesPage />} />
          <Route path="/stream/:gameId" element={<StreamPage />} />
        </Routes>
      </AccessibilityProvider>
    </MemoryRouter>,
  );
}

describe('GamesPage', () => {
  it('lists exported matches', async () => {
    mockFetch({ '/matches/index.json': sampleIndex });
    renderAt('/');
    expect(await screen.findByRole('listitem', { name: /Carlos Alcaraz versus Jannik Sinner/ })).toBeInTheDocument();
  });

  it('explains how to create a match when none exist', async () => {
    mockFetch({ '/matches/index.json': { matches: [] } });
    renderAt('/');
    expect(await screen.findByText(/No processed matches yet/)).toBeInTheDocument();
  });

  it('reports why the match list could not be read', async () => {
    mockFetch({});
    renderAt('/');
    expect(await screen.findByRole('alert')).toHaveTextContent(/HTTP 404/);
  });
});

describe('StreamPage', () => {
  it('shows the scoreboard, both POV cards, the POV overlay and the free camera', async () => {
    mockFetch({ '/matches/demo/manifest.json': sampleManifest, '/matches/demo/tracks.json': sampleTracks });
    renderAt('/stream/demo');
    expect(await screen.findByText('ALCARAZ')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Carlos Alcaraz, near player/ })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /Jannik Sinner, far player/ }));
    expect(screen.getByRole('dialog', { name: /Jannik Sinner first-person view/ })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Open free camera' }));
    expect(screen.getByTitle('3D free camera viewer')).toHaveAttribute('src', 'http://localhost:8080');
  });

  it('reports unknown matches', async () => {
    mockFetch({});
    renderAt('/stream/missing');
    expect(await screen.findByText(/Match not found/)).toBeInTheDocument();
  });
});
