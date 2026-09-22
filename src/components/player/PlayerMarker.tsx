import type { Player } from '../../types';
import './styles.css';

interface PlayerMarkerProps {
  player: Player;
  isActive?: boolean;
  onClick?: () => void;
}

export function PlayerMarker({ player, isActive = false, onClick }: PlayerMarkerProps) {
  return (
    <div
      className={`player-marker ${isActive ? 'player-marker-active' : ''}`}
      style={{
        '--team-color': player.teamColor,
        left: `${player.fieldPosition.x}%`,
        top: `${player.fieldPosition.y}%`,
      } as React.CSSProperties}
      onClick={onClick}
    >
      <div className="marker-pulse" />
      <div className="marker-dot">
        <span className="marker-number">{player.number}</span>
      </div>
      <div className="marker-line" />
    </div>
  );
}
