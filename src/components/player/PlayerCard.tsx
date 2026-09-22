import { useState, useRef, useEffect } from 'react';
import type { Player } from '../../types';
import brunoPov from '../../assets/bruno_pov.mov';
import assRefPov from '../../assets/ass_ref_pov.mov';
import goaliePov from '../../assets/goalie_pov.mov';
import goalieUpper from '../../assets/goalie_upper.mov';
import refPov from '../../assets/ref_pov.mov';
import './player-glass.css';

interface PlayerCardProps {
  player: Player;
  onExpand?: (player: Player) => void;
  mainVideoTime?: number;
  isMainPlaying?: boolean;
}

// POV video sync config
const POV_CONFIG = {
  'Penalty Taker': { startTime: 6.5, duration: 1, endTime: 7.5 },
  'Assistant Referee': { startTime: 6, duration: 2, endTime: 8 },
  'Goalkeeper': { switchTime: 5, startTime: 6, duration: 1.7, endTime: 7.7 },
  'Referee': { startTime: 5.5, duration: 10, endTime: 15.5 },
};

export function PlayerCard({ player, onExpand, mainVideoTime = 0, isMainPlaying = false }: PlayerCardProps) {
  const [isHovered, setIsHovered] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);

  const isPenaltyTaker = player.position === 'Penalty Taker';
  const isAssistantRef = player.position === 'Assistant Referee';
  const isGoalkeeper = player.position === 'Goalkeeper';
  const isReferee = player.position === 'Referee';
  const hasPov = isPenaltyTaker || isAssistantRef || isGoalkeeper || isReferee;
  const povSrc = isPenaltyTaker ? brunoPov : isAssistantRef ? assRefPov : isGoalkeeper ? goaliePov : isReferee ? refPov : null;

  // For goalkeeper, show looping upper video before switch time
  const config = POV_CONFIG[player.position as keyof typeof POV_CONFIG];
  const switchTime = config && 'switchTime' in config ? config.switchTime : config?.startTime;
  const showGoalieLoop = isGoalkeeper && switchTime !== undefined && mainVideoTime < switchTime;

  // Sync POV preview with main video time
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !hasPov) return;

    const config = POV_CONFIG[player.position as keyof typeof POV_CONFIG];
    if (!config) return;

    const { startTime, duration, endTime } = config;

    if (mainVideoTime >= startTime && mainVideoTime < endTime) {
      const povTime = mainVideoTime - startTime;
      const clampedTime = Math.min(Math.max(povTime, 0), duration);

      if (Math.abs(video.currentTime - clampedTime) > 0.1) {
        video.currentTime = clampedTime;
      }

      if (isMainPlaying && video.paused) {
        video.play().catch(() => {});
      } else if (!isMainPlaying && !video.paused) {
        video.pause();
      }
    } else if (mainVideoTime < startTime) {
      video.currentTime = 0;
      video.pause();
    } else {
      video.currentTime = duration;
      video.pause();
    }
  }, [mainVideoTime, isMainPlaying, hasPov, player.position]);

  const getRoleLabel = () => {
    if (player.position === 'Goalkeeper') return 'GK';
    if (player.position === 'Penalty Taker') return 'PK';
    if (player.position === 'Referee') return 'REF';
    if (player.position === 'Assistant Referee') return 'AR';
    return player.position.slice(0, 3).toUpperCase();
  };

  const getStatValue = () => {
    if (player.position === 'Goalkeeper') return { label: 'SAVE %', value: `${player.stats.passes}%` };
    if (player.position === 'Penalty Taker') return { label: 'CONV %', value: `${player.stats.passes}%` };
    if (player.position === 'Referee') return { label: 'CALLS', value: `${player.stats.sprints}` };
    if (player.position === 'Assistant Referee') return { label: 'MATCHES', value: `${player.stats.sprints}` };
    return { label: 'RATING', value: '7.2' };
  };

  const stat = getStatValue();

  return (
    <div
      className={`player-card ${isHovered ? 'player-card--hover' : ''}`}
      style={{ '--player-accent': player.teamColor } as React.CSSProperties}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      onClick={() => onExpand?.(player)}
    >
      {/* Glass layers */}
      <div className="player-card-glass" />
      <div className="player-card-border" />
      <div className="player-card-accent" />

      {/* Content */}
      <div className="player-card-content">
        {/* POV Video Preview */}
        <div className="player-card-pov">
          {hasPov && povSrc ? (
            <>
              {isGoalkeeper && (
                <video
                  src={goalieUpper}
                  muted
                  playsInline
                  autoPlay
                  loop
                  className="player-card-pov-video"
                  style={{ display: showGoalieLoop ? 'block' : 'none' }}
                />
              )}
              <video
                ref={videoRef}
                src={povSrc}
                muted
                playsInline
                className={`player-card-pov-video ${isAssistantRef ? 'player-card-pov-video--crop-top' : ''}`}
                style={isGoalkeeper ? { display: showGoalieLoop ? 'none' : 'block' } : undefined}
              />
            </>
          ) : (
            <div className="player-card-pov-bg">
              <div className="player-card-pov-grid" />
              <span className="player-card-pov-text">POV</span>
            </div>
          )}
          <div className="player-card-pov-live">
            <span className="player-card-pov-dot" />
            <span>LIVE</span>
          </div>
        </div>

        {/* Player Info */}
        <div className="player-card-info">
          <div className="player-card-header">
            <div className="player-card-name-section">
              <span className="player-card-name">{player.name}</span>
              <div className="player-card-meta">
                {player.number > 0 && (
                  <span className="player-card-number">#{player.number}</span>
                )}
                <span className="player-card-role">{getRoleLabel()}</span>
              </div>
            </div>
            <div className="player-card-stat">
              <span className="player-card-stat-value">{stat.value}</span>
              <span className="player-card-stat-label">{stat.label}</span>
            </div>
          </div>
        </div>

        {/* Expand indicator */}
        <div className="player-card-expand">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polyline points="15 3 21 3 21 9" />
            <polyline points="9 21 3 21 3 15" />
            <line x1="21" y1="3" x2="14" y2="10" />
            <line x1="3" y1="21" x2="10" y2="14" />
          </svg>
        </div>
      </div>
    </div>
  );
}
