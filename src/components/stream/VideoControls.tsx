import { useState } from 'react';
import './controls-glass.css';

interface VideoControlsProps {
  currentTime: number;
  duration: number;
  isPlaying: boolean;
  onPlayPause: () => void;
  onSeek: (time: number) => void;
  onFullscreen?: () => void;
}

const formatTime = (seconds: number): string => {
  if (!seconds || isNaN(seconds)) return '0:00';
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
};

export function VideoControls({
  currentTime,
  duration,
  isPlaying,
  onPlayPause,
  onSeek,
  onFullscreen,
}: VideoControlsProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const progress = duration > 0 ? (currentTime / duration) * 100 : 0;

  const handleProgressClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const percentage = clickX / rect.width;
    onSeek(percentage * duration);
  };

  return (
    <div className={`controls-glass ${isExpanded ? 'controls-glass--expanded' : 'controls-glass--collapsed'}`}>
      <div className="controls-glass-bg" />
      <div className="controls-glass-surface" />

      <div className="controls-content">
        {/* Toggle button (collapsed state) */}
        {!isExpanded && (
          <button
            className="controls-toggle"
            onClick={() => setIsExpanded(true)}
            aria-label="Show controls"
          >
            <svg viewBox="0 0 24 24" fill="currentColor">
              <rect x="4" y="6" width="16" height="2" rx="1" />
              <rect x="4" y="11" width="16" height="2" rx="1" />
              <rect x="4" y="16" width="16" height="2" rx="1" />
            </svg>
          </button>
        )}

        {/* Expanded controls */}
        {isExpanded && (
          <>
            {/* Play/Pause */}
            <button
              className="controls-btn controls-btn--play"
              onClick={onPlayPause}
              aria-label={isPlaying ? 'Pause' : 'Play'}
            >
              <div className="controls-btn-bg" />
              {isPlaying ? (
                <svg viewBox="0 0 24 24" fill="currentColor">
                  <rect x="6" y="4" width="4" height="16" rx="1" />
                  <rect x="14" y="4" width="4" height="16" rx="1" />
                </svg>
              ) : (
                <svg viewBox="0 0 24 24" fill="currentColor">
                  <path d="M6 4.5l14 7.5-14 7.5V4.5z" />
                </svg>
              )}
            </button>

            {/* Progress bar */}
            <div className="controls-progress-section">
              <span className="controls-time">{formatTime(currentTime)}</span>

              <div className="controls-progress" onClick={handleProgressClick}>
                <div className="controls-progress-track">
                  <div
                    className="controls-progress-fill"
                    style={{ width: `${progress}%` }}
                  />
                  <div
                    className="controls-progress-handle"
                    style={{ left: `${progress}%` }}
                  />
                </div>
              </div>

              <span className="controls-time controls-time--dim">{formatTime(duration)}</span>
            </div>

            {/* Fullscreen */}
            <button
              className="controls-btn controls-btn--action"
              onClick={onFullscreen}
              aria-label="Fullscreen"
            >
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <polyline points="15 3 21 3 21 9" />
                <polyline points="9 21 3 21 3 15" />
                <line x1="21" y1="3" x2="14" y2="10" />
                <line x1="3" y1="21" x2="10" y2="14" />
              </svg>
            </button>

            {/* Close button */}
            <button
              className="controls-btn controls-btn--close"
              onClick={() => setIsExpanded(false)}
              aria-label="Hide controls"
            >
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                <path d="M18 6L6 18M6 6l12 12" />
              </svg>
            </button>
          </>
        )}
      </div>
    </div>
  );
}
