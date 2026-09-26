import type { CSSProperties } from 'react';
import type { PlayerInfo, PlayerSnapshot } from '../../lib/match';
import { PovVideo } from './PovVideo';
import '../player/player.css';

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
    { value: String(Math.round(snapshot.speedKmh)), label: 'km/h now' },
    { value: String(Math.round(snapshot.topSpeedKmh)), label: 'km/h top' },
    { value: snapshot.distanceM.toFixed(1), label: 'metres run' },
  ];

  return (
    <div className="pov" role="dialog" aria-label={`${player.name} first-person view`}>
      <div className="pov-panel" style={{ '--accent': player.color } as CSSProperties}>
        <header className="pov-head">
          <div>
            <h2 className="pov-name">{player.name}</h2>
            <p className="pov-end">
              {player.id === 'near' ? 'Near end' : 'Far end'}, standing {player.statureM.toFixed(2)} m
            </p>
          </div>
          <button className="pov-close" onClick={onClose} aria-label="Close first-person view">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
              <path d="M18 6L6 18M6 6l12 12" />
            </svg>
          </button>
        </header>

        <PovVideo
          src={povSrc}
          mainTime={mainTime}
          isPlaying={isPlaying}
          className="pov-video"
          label={`${player.name} first-person video`}
        />

        <dl className="pov-stats">
          {stats.map((stat) => (
            <div key={stat.label} className="pov-stat">
              <dd className="pov-stat-value">{stat.value}</dd>
              <dt className="pov-stat-label">{stat.label}</dt>
            </div>
          ))}
        </dl>
      </div>
    </div>
  );
}
