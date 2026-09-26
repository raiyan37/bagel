import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import type { PlayerSnapshot } from '../../lib/match';
import { sampleManifest } from '../../test/fixtures';
import { FreeCamOverlay } from './FreeCamOverlay';
import { PovOverlay } from './PovOverlay';
import { PovVideo } from './PovVideo';
import { TennisPlayerCard } from './TennisPlayerCard';

const [near, far] = sampleManifest.players;
const snapshot: PlayerSnapshot = { x: 50, y: 60, visible: true, speedKmh: 17.6, distanceM: 12.34, topSpeedKmh: 21.2 };

describe('TennisPlayerCard', () => {
  it('shows the player and live speed and expands on click', () => {
    const onExpand = vi.fn();
    render(
      <TennisPlayerCard player={near} povSrc="/matches/demo/pov_near.mp4" snapshot={snapshot} mainTime={0} isPlaying={false} onExpand={onExpand} />,
    );
    expect(screen.getByText('Carlos Alcaraz')).toBeInTheDocument();
    expect(screen.getByText('near')).toBeInTheDocument();
    expect(screen.getByText('18')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /Carlos Alcaraz/ }));
    expect(onExpand).toHaveBeenCalledWith(near);
  });
});

describe('PovVideo', () => {
  it('follows the broadcast clock and play state', () => {
    const { rerender } = render(<PovVideo src="/pov.mp4" mainTime={0} isPlaying={false} label="pov" />);
    const video = screen.getByLabelText('pov') as HTMLVideoElement;
    let time = 0;
    let paused = true;
    Object.defineProperty(video, 'currentTime', { configurable: true, get: () => time, set: (v: number) => { time = v; } });
    Object.defineProperty(video, 'paused', { configurable: true, get: () => paused });
    const play = vi.spyOn(video, 'play').mockImplementation(async () => { paused = false; });
    const pause = vi.spyOn(video, 'pause').mockImplementation(() => { paused = true; });

    rerender(<PovVideo src="/pov.mp4" mainTime={3} isPlaying={true} label="pov" />);
    expect(time).toBe(3);
    expect(play).toHaveBeenCalledTimes(1);

    rerender(<PovVideo src="/pov.mp4" mainTime={3.05} isPlaying={true} label="pov" />);
    expect(time).toBe(3);

    rerender(<PovVideo src="/pov.mp4" mainTime={3.1} isPlaying={false} label="pov" />);
    expect(pause).toHaveBeenCalledTimes(1);
  });

  it('re-syncs when the clip source changes while paused', () => {
    const { rerender } = render(<PovVideo src="/a.mp4" mainTime={0} isPlaying={false} label="pov" />);
    const video = screen.getByLabelText('pov') as HTMLVideoElement;
    let time = 0;
    const paused = true;
    Object.defineProperty(video, 'currentTime', { configurable: true, get: () => time, set: (v: number) => { time = v; } });
    Object.defineProperty(video, 'paused', { configurable: true, get: () => paused });

    rerender(<PovVideo src="/a.mp4" mainTime={5} isPlaying={false} label="pov" />);
    expect(time).toBe(5);

    // Swapping the source resets the media element to frame 0; the effect has to seek it back.
    time = 0;
    rerender(<PovVideo src="/b.mp4" mainTime={5} isPlaying={false} label="pov" />);
    expect(time).toBe(5);
  });
});

describe('PovOverlay', () => {
  it('shows speed, top speed and distance and closes', () => {
    const onClose = vi.fn();
    render(<PovOverlay player={far} povSrc="/matches/demo/pov_far.mp4" snapshot={snapshot} mainTime={0} isPlaying={false} onClose={onClose} />);
    expect(screen.getByRole('dialog', { name: /Jannik Sinner first-person view/ })).toBeInTheDocument();
    expect(screen.getByText('18')).toBeInTheDocument();
    expect(screen.getByText('21')).toBeInTheDocument();
    expect(screen.getByText('12.3')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Close first-person view' }));
    expect(onClose).toHaveBeenCalled();
  });
});

describe('FreeCamOverlay', () => {
  it('embeds the live Viser viewer', () => {
    render(<FreeCamOverlay matchId="demo" viewerUrl="http://localhost:8080" clipSrc={null} mainTime={0} isPlaying={false} onClose={() => {}} />);
    expect(screen.getByTitle('3D free camera viewer')).toHaveAttribute('src', 'http://localhost:8080');
    expect(screen.queryByRole('tab', { name: 'Exported clip' })).not.toBeInTheDocument();
    expect(screen.getByText('horizon view --match-id demo')).toBeInTheDocument();
  });

  it('switches to the exported clip and closes', () => {
    const onClose = vi.fn();
    render(
      <FreeCamOverlay matchId="demo" viewerUrl="http://localhost:8080" clipSrc="/matches/demo/free_cam.mp4" mainTime={0} isPlaying={false} onClose={onClose} />,
    );
    fireEvent.click(screen.getByRole('tab', { name: 'Exported clip' }));
    expect(screen.getByLabelText('Exported free camera clip')).toHaveAttribute('src', '/matches/demo/free_cam.mp4');
    expect(screen.queryByTitle('3D free camera viewer')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Close free camera' }));
    expect(onClose).toHaveBeenCalled();
  });
});
