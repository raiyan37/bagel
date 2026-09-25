import { useRef, useEffect } from 'react';
import type { Player } from '../../types';
import brunoPov from '../../assets/bruno_pov.mov';
import assRefPov from '../../assets/ass_ref_pov.mov';
import goaliePov from '../../assets/goalie_pov.mov';
import goalieUpper from '../../assets/goalie_upper.mov';
import refPov from '../../assets/ref_pov.mov';
import './player-glass.css';

interface POVVideoOverlayProps {
  player: Player;
  onClose: () => void;
  mainVideoTime: number;
  isMainPlaying: boolean;
}

// POV video sync config
const POV_CONFIG = {
  'Penalty Taker': { startTime: 6.5, duration: 1, endTime: 7.5 },
  'Assistant Referee': { startTime: 6, duration: 2, endTime: 8 },
  'Goalkeeper': { switchTime: 5, startTime: 6, duration: 1.7, endTime: 7.7 },
  'Referee': { startTime: 5.5, duration: 10, endTime: 15.5 },
};

export function POVVideoOverlay({ player, onClose, mainVideoTime, isMainPlaying }: POVVideoOverlayProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const getStats = () => {
    if (player.position === 'Goalkeeper') {
      return [
        { value: '1', label: 'Saves' },
        { value: '87%', label: 'Save Rate' },
        { value: '2.1', label: 'Distance' },
      ];
    }
    if (player.position === 'Penalty Taker') {
      return [
        { value: `${player.stats.passes}%`, label: 'Conv Rate' },
        { value: '118', label: 'km/h' },
        { value: '0.9', label: 'xG' },
      ];
    }
    if (player.position === 'Referee') {
      return [
        { value: `${player.stats.sprints}`, label: 'Calls' },
        { value: '2.1', label: 'KM Run' },
        { value: '0', label: 'Cards' },
      ];
    }
    if (player.position === 'Assistant Referee') {
      return [
        { value: `${player.stats.sprints}`, label: 'Flags' },
        { value: '1.2', label: 'KM Run' },
        { value: '156', label: 'Matches' },
      ];
    }
    return [
      { value: '7.2', label: 'Rating' },
      { value: `${player.stats.passes}`, label: 'Passes' },
      { value: `${player.stats.distance}`, label: 'KM' },
    ];
  };

  const stats = getStats();

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

  // Sync POV video with main video time
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !hasPov) return;

    const config = POV_CONFIG[player.position as keyof typeof POV_CONFIG];
    if (!config) return;

    const { startTime, duration, endTime } = config;

    if (mainVideoTime >= startTime && mainVideoTime < endTime) {
      // During the POV window, play synced
      const povTime = mainVideoTime - startTime;
      const clampedTime = Math.min(Math.max(povTime, 0), duration);

      // Only seek if significantly out of sync
      if (Math.abs(video.currentTime - clampedTime) > 0.1) {
        video.currentTime = clampedTime;
      }

      if (isMainPlaying && video.paused) {
        video.play().catch(() => {});
      } else if (!isMainPlaying && !video.paused) {
        video.pause();
      }
    } else if (mainVideoTime < startTime) {
      // Before POV window - show first frame
      video.currentTime = 0;
      video.pause();
    } else {
      // After POV window - show last frame
      video.currentTime = duration;
      video.pause();
    }
  }, [mainVideoTime, isMainPlaying, hasPov, player.position]);

  return (
    <div className="pov-glass">
      <div className="pov-glass-container" style={{ '--player-accent': player.teamColor } as React.CSSProperties}>
        {/* Glass layers */}
        <div className="pov-glass-bg" />
        <div className="pov-glass-surface" />
        <div className="pov-glass-glow" />

        {/* Header */}
        <div className="pov-header">
          <div className="pov-player-info">
            <div className="pov-avatar">
              <span>{player.name.charAt(0)}</span>
            </div>
            <div className="pov-details">
              <span className="pov-player-name">{player.name}</span>
              <span className="pov-player-position">{player.position}</span>
            </div>
          </div>

          <div className="pov-live-badge">
            <span className="pov-live-dot" />
            <span className="pov-live-text">LIVE</span>
          </div>

          <button className="pov-close" onClick={onClose} aria-label="Close">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M18 6L6 18M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Video */}
        <div className="pov-video">
          {hasPov && povSrc ? (
            <>
              {isGoalkeeper && (
                <video
                  src={goalieUpper}
                  muted
                  playsInline
                  autoPlay
                  loop
                  className="pov-video-player"
                  style={{ display: showGoalieLoop ? 'block' : 'none' }}
                />
              )}
              <video
                ref={videoRef}
                src={povSrc}
                muted
                playsInline
                className={`pov-video-player ${isAssistantRef ? 'pov-video-player--crop-top' : ''}`}
                style={isGoalkeeper ? { display: showGoalieLoop ? 'none' : 'block' } : undefined}
              />
            </>
          ) : (
            <div className="pov-video-placeholder">
              <div className="pov-video-grid" />
              <span className="pov-video-text">POV Camera Feed</span>
            </div>
          )}
        </div>

        {/* Stats */}
        <div className="pov-stats">
          {stats.map((stat, index) => (
            <div key={index} className="pov-stat">
              <span className="pov-stat-value">{stat.value}</span>
              <span className="pov-stat-label">{stat.label}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
