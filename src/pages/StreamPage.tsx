import { useCallback, useEffect, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { AccessibilityToggle } from '../components/shared/AccessibilityToggle';
import { StreamContainer } from '../components/stream/StreamContainer';
import type { StreamContainerRef } from '../components/stream/StreamContainer';
import { VideoControls } from '../components/stream/VideoControls';
import { FreeCamOverlay } from '../components/tennis/FreeCamOverlay';
import { PovOverlay } from '../components/tennis/PovOverlay';
import { TennisPlayerCard } from '../components/tennis/TennisPlayerCard';
import { TennisScoreOverlay } from '../components/tennis/TennisScoreOverlay';
import { useMatchData } from '../hooks/useMatchData';
import { assetUrl, cardPosition, frameIndexAt, playerSnapshot, videoPercentToScreen } from '../lib/match';
import type { PlayerRole } from '../lib/match';
import '../App.css';
import '../components/tennis/tennis.css';

const CARD_SIZE = { width: 180, height: 160 };

function BackButton({ onClick }: { onClick: () => void }) {
  return (
    <button className="home-button" onClick={onClick} aria-label="Return to home">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M15 18l-6-6 6-6" />
      </svg>
      <span>Back</span>
    </button>
  );
}

export function StreamPage() {
  const navigate = useNavigate();
  const { gameId } = useParams<{ gameId: string }>();
  const { manifest, tracks, error } = useMatchData(gameId);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [showCards, setShowCards] = useState(true);
  const [expandedId, setExpandedId] = useState<PlayerRole | null>(null);
  const [freeCamOpen, setFreeCamOpen] = useState(false);
  const [viewport, setViewport] = useState(() => ({ width: window.innerWidth, height: window.innerHeight }));
  const streamRef = useRef<StreamContainerRef>(null);

  useEffect(() => {
    const onResize = () => setViewport({ width: window.innerWidth, height: window.innerHeight });
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  // `timeupdate` fires only ~4x per second; poll the video every animation frame while playing.
  useEffect(() => {
    if (!isPlaying) return;
    let raf = 0;
    const tick = () => {
      setCurrentTime(streamRef.current?.getCurrentTime() ?? 0);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [isPlaying]);

  const handleTimeUpdate = useCallback((time: number, dur: number) => {
    setCurrentTime(time);
    setDuration(dur);
  }, []);

  const handleStateChange = useCallback((playing: boolean) => {
    setIsPlaying(playing);
  }, []);

  if (error) {
    return (
      <div className="app">
        <BackButton onClick={() => navigate('/')} />
        <div className="stream-status" role="alert">
          <p>Match not found.</p>
          <p>Export it with <code>horizon export --match-id {gameId}</code> ({error})</p>
        </div>
      </div>
    );
  }

  if (!manifest || !tracks) {
    return (
      <div className="app">
        <div className="stream-status" role="status">Loading match…</div>
      </div>
    );
  }

  const frame = frameIndexAt(currentTime, manifest.fps, manifest.frameCount);
  const videoSize = { width: manifest.width, height: manifest.height };
  const [near, far] = manifest.players;
  const expanded = manifest.players.find((p) => p.id === expandedId) ?? null;

  return (
    <div className="app">
      <StreamContainer
        ref={streamRef}
        videoSrc={assetUrl(manifest.id, manifest.video)}
        onTimeUpdate={handleTimeUpdate}
        onStateChange={handleStateChange}
      >
        <BackButton onClick={() => navigate('/')} />

        <button
          className="toggle-pov-button"
          onClick={() => setShowCards(!showCards)}
          aria-label={showCards ? 'Hide POV cards' : 'Show POV cards'}
          aria-pressed={showCards}
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
            <circle cx="12" cy="12" r="3" />
          </svg>
          <span>POV</span>
        </button>

        <button className="freecam-button" onClick={() => setFreeCamOpen(true)} aria-label="Open free camera">
          Free cam
        </button>

        <TennisScoreOverlay near={near} far={far} score={manifest.score} competition={manifest.competition} />

        {showCards &&
          manifest.players.map((player, index) => {
            const snapshot = playerSnapshot(tracks, player.id, frame);
            const anchor = videoPercentToScreen(snapshot.x, snapshot.y, videoSize, viewport);
            const position = cardPosition(anchor, CARD_SIZE, viewport);
            return (
              <div
                key={player.id}
                className="player-card-wrapper"
                style={{ left: position.left, top: position.top, animationDelay: `${index * 100}ms`, transition: 'left 0.1s linear, top 0.1s linear' }}
              >
                <div style={{ opacity: snapshot.visible ? 1 : 0.5 }}>
                  <TennisPlayerCard
                    player={player}
                    povSrc={assetUrl(manifest.id, player.pov)}
                    snapshot={snapshot}
                    mainTime={currentTime}
                    isPlaying={isPlaying}
                    onExpand={(p) => setExpandedId(p.id)}
                  />
                </div>
              </div>
            );
          })}

        <VideoControls
          currentTime={currentTime}
          duration={duration}
          isPlaying={isPlaying}
          onPlayPause={() => (isPlaying ? streamRef.current?.pause() : streamRef.current?.play())}
          onSeek={(time) => streamRef.current?.seekTo(time)}
          onFullscreen={() => document.documentElement.requestFullscreen?.()}
        />
      </StreamContainer>

      {expanded && (
        <PovOverlay
          player={expanded}
          povSrc={assetUrl(manifest.id, expanded.pov)}
          snapshot={playerSnapshot(tracks, expanded.id, frame)}
          mainTime={currentTime}
          isPlaying={isPlaying}
          onClose={() => setExpandedId(null)}
        />
      )}

      {freeCamOpen && (
        <FreeCamOverlay
          matchId={manifest.id}
          viewerUrl={manifest.viewerUrl}
          clipSrc={manifest.freeCam ? assetUrl(manifest.id, manifest.freeCam) : null}
          mainTime={currentTime}
          isPlaying={isPlaying}
          onClose={() => setFreeCamOpen(false)}
        />
      )}

      <AccessibilityToggle />
    </div>
  );
}
