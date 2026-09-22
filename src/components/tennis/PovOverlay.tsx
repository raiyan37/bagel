import type { CSSProperties } from 'react';
import type { PlayerInfo, PlayerSnapshot } from '../../lib/match';
import { PovVideo } from './PovVideo';
import '../player/player-glass.css';

interface PovOverlayProps {
  player: PlayerInfo;
  povSrc: string;
  snapshot: PlayerSnapshot;
  mainTime: number;
  isPlaying: boolean;
  onClose: () => void;
}

export function PovOverlay({ player, povSrc, snapshot, mainTime, isPlaying, onClose }: PovOverlayProps) {
  const stats = [
    { value: String(Math.round(snapshot.speedKmh)), label: 'km/h' },
    { value: String(Math.round(snapshot.topSpeedKmh)), label: 'Top km/h' },
    { value: snapshot.distanceM.toFixed(1), label: 'Metres' },
  ];

  return (
    <div className="pov-glass" role="dialog" aria-label={`${player.name} first-person view`}>
      <div className="pov-glass-container" style={{ '--player-accent': player.color } as CSSProperties}>
        <div className="pov-glass-bg" />
        <div className="pov-glass-surface" />
        <div className="pov-header">
          <div className="pov-player-info">
            <div className="pov-avatar">
              <span>{player.name.charAt(0)}</span>
            </div>
            <div className="pov-details">
              <span className="pov-player-name">{player.name}</span>
              <span className="pov-player-position">
                {player.id === 'near' ? 'Near end' : 'Far end'} · {player.statureM.toFixed(2)} m
              </span>
            </div>
          </div>
          <button className="pov-close" onClick={onClose} aria-label="Close first-person view">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
              <path d="M18 6L6 18M6 6l12 12" />
            </svg>
          </button>
        </div>
        <div className="pov-video">
          <PovVideo src={povSrc} mainTime={mainTime} isPlaying={isPlaying} className="pov-video-player" label={`${player.name} first-person video`} />
        </div>
        <div className="pov-stats">
          {stats.map((stat) => (
            <div key={stat.label} className="pov-stat">
              <span className="pov-stat-value">{stat.value}</span>
              <span className="pov-stat-label">{stat.label}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
