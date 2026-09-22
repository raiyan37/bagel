import { useState } from 'react';
import type { CSSProperties } from 'react';
import type { PlayerInfo, PlayerSnapshot } from '../../lib/match';
import { PovVideo } from './PovVideo';
import '../player/player-glass.css';
import './tennis.css';

interface TennisPlayerCardProps {
  player: PlayerInfo;
  povSrc: string;
  snapshot: PlayerSnapshot;
  mainTime: number;
  isPlaying: boolean;
  onExpand: (player: PlayerInfo) => void;
}

export function TennisPlayerCard({ player, povSrc, snapshot, mainTime, isPlaying, onExpand }: TennisPlayerCardProps) {
  const [hovered, setHovered] = useState(false);
  const speed = Math.round(snapshot.speedKmh);

  return (
    <button
      type="button"
      className={`player-card ${hovered ? 'player-card--hover' : ''}`}
      style={{ '--player-accent': player.color } as CSSProperties}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      onClick={() => onExpand(player)}
      aria-label={`${player.name}, ${player.id} player, ${speed} kilometres per hour. Open first-person view.`}
    >
      <div className="player-card-glass" />
      <div className="player-card-border" />
      <div className="player-card-accent" />
      <div className="player-card-content">
        <div className="player-card-pov">
          <PovVideo src={povSrc} mainTime={mainTime} isPlaying={isPlaying} className="player-card-pov-video" label={`${player.name} point of view`} />
          <div className="player-card-pov-live">
            <span className="player-card-pov-dot" />
            <span>POV</span>
          </div>
        </div>
        <div className="player-card-info">
          <div className="player-card-header">
            <div className="player-card-name-section">
              <span className="player-card-name">{player.name}</span>
              <div className="player-card-meta">
                <span className="player-card-role">{player.id === 'near' ? 'NEAR' : 'FAR'}</span>
              </div>
            </div>
            <div className="player-card-stat">
              <span className="player-card-stat-value">{speed}</span>
              <span className="player-card-stat-label">KM/H</span>
            </div>
          </div>
        </div>
      </div>
    </button>
  );
}
