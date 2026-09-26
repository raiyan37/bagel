import type { CSSProperties } from 'react';
import type { PlayerInfo, PlayerSnapshot } from '../../lib/match';
import { PovVideo } from './PovVideo';
import '../player/player.css';

interface TennisPlayerCardProps {
  player: PlayerInfo;
  povSrc: string;
  snapshot: PlayerSnapshot;
  mainTime: number;
  isPlaying: boolean;
  onExpand: (player: PlayerInfo) => void;
}

export function TennisPlayerCard({ player, povSrc, snapshot, mainTime, isPlaying, onExpand }: TennisPlayerCardProps) {
  const speed = Math.round(snapshot.speedKmh);
  const end = player.id === 'near' ? 'near' : 'far';

  return (
    <button
      type="button"
      className="pcard"
      style={{ '--accent': player.color } as CSSProperties}
      onClick={() => onExpand(player)}
      aria-label={`${player.name}, ${player.id} player, ${speed} kilometres per hour. Open first-person view.`}
    >
      <span className="pcard-view">
        <PovVideo
          src={povSrc}
          mainTime={mainTime}
          isPlaying={isPlaying}
          className="pcard-video"
          label={`${player.name} point of view`}
        />
      </span>
      <span className="pcard-foot">
        <span className="pcard-name">{player.name}</span>
        <span className="pcard-end">{end}</span>
        <span className="pcard-speed">
          <span className="pcard-speed-value">{speed}</span>
          <span className="pcard-speed-unit">km/h</span>
        </span>
      </span>
    </button>
  );
}
