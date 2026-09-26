import { useState } from 'react';
import { PovVideo } from './PovVideo';
import '../player/player.css';
import './tennis.css';

interface FreeCamOverlayProps {
  matchId: string;
  viewerUrl: string;
  clipSrc: string | null;
  mainTime: number;
  isPlaying: boolean;
  onClose: () => void;
}

export function FreeCamOverlay({ matchId, viewerUrl, clipSrc, mainTime, isPlaying, onClose }: FreeCamOverlayProps) {
  const [mode, setMode] = useState<'live' | 'clip'>('live');
  const showClip = mode === 'clip' && clipSrc !== null;

  return (
    <div className="freecam" role="dialog" aria-label="Free camera">
      <div className="freecam-panel">
        <header className="freecam-header">
          <span className="freecam-title">Free camera</span>
          <div className="freecam-tabs" role="tablist">
            <button type="button" role="tab" aria-selected={!showClip} onClick={() => setMode('live')}>
              Live 3D
            </button>
            {clipSrc && (
              <button type="button" role="tab" aria-selected={showClip} onClick={() => setMode('clip')}>
                Exported clip
              </button>
            )}
          </div>
          <a className="freecam-link" href={viewerUrl} target="_blank" rel="noreferrer">
            Open in new tab
          </a>
          <button className="pov-close" onClick={onClose} aria-label="Close free camera">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
              <path d="M18 6L6 18M6 6l12 12" />
            </svg>
          </button>
        </header>
        {showClip && clipSrc ? (
          <PovVideo src={clipSrc} mainTime={mainTime} isPlaying={isPlaying} className="freecam-video" label="Exported free camera clip" />
        ) : (
          <>
            <iframe className="freecam-frame" src={viewerUrl} title="3D free camera viewer" allow="fullscreen" />
            <p className="freecam-hint">
              Blank? Start the 3D viewer first: <code>horizon view --match-id {matchId}</code>
            </p>
          </>
        )}
      </div>
    </div>
  );
}
