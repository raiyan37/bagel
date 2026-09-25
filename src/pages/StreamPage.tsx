import { useState, useRef, useCallback, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { StreamContainer } from '../components/stream/StreamContainer';
import type { StreamContainerRef } from '../components/stream/StreamContainer';
import { PenaltyScoreOverlay } from '../components/stream/PenaltyScoreOverlay';
import { LiveBadge } from '../components/stream/LiveBadge';
import { VideoControls } from '../components/stream/VideoControls';
import { PlayerCard } from '../components/player/PlayerCard';
import { POVVideoOverlay } from '../components/player/POVVideoOverlay';
import { usePlayerTracking } from '../hooks/usePlayerTracking';
import { GoalConfetti } from '../components/stream/GoalConfetti';
import { AccessibilityToggle } from '../components/shared/AccessibilityToggle';
import { useAccessibility } from '../contexts/AccessibilityContext';
import type { Player } from '../types';
import brunoVideo from '../assets/bruno.mov';
import '../App.css';

// Penalty shootout data - FA Cup 2024/25: Man United vs Arsenal
const initialPenaltyData = {
  homeTeam: {
    name: 'Manchester United',
    shortName: 'MUN',
    flag: '',
    color: '#DA291C',
  },
  awayTeam: {
    name: 'Arsenal',
    shortName: 'ARS',
    flag: '',
    color: '#EF0107',
  },
  homePenalties: {
    scored: 0,
    taken: 0,
    saved: 0,
  },
  awayPenalties: {
    scored: 0,
    taken: 0,
    saved: 0,
  },
  currentRound: 1,
  isHomeTurn: true,
};

// Goal event timing
const GOAL_TIME = 8; // Bruno scores at 8 seconds

// Penalty-specific players - FA Cup 2024/25: Bruno vs Raya (First penalty)
// Stats will be updated dynamically based on goal state
const getPlayers = (scored: boolean): Player[] => [
  {
    id: 'goalkeeper',
    name: 'David Raya',
    number: 22,
    position: 'Goalkeeper',
    team: 'away',
    teamColor: '#F5A623',
    avatar: '',
    stats: {
      passes: 24, // Career PK save %
      passAccuracy: 0,
      shots: 0,
      shotsOnTarget: 0,
      tackles: 0,
      distance: 0,
      speed: 0,
      sprints: 0,
    },
    fieldPosition: { x: 15, y: 50 },
  },
  {
    id: 'penalty-taker',
    name: 'Bruno Fernandes',
    number: 8,
    position: 'Penalty Taker',
    team: 'home',
    teamColor: '#3B82F6',
    avatar: '',
    stats: {
      passes: scored ? 93 : 92, // Career conversion % (ticks up after scoring)
      passAccuracy: 0,
      shots: 0,
      shotsOnTarget: 0,
      tackles: 0,
      distance: 0,
      speed: 0,
      sprints: 0,
    },
    fieldPosition: { x: 30, y: 50 },
  },
  {
    id: 'referee',
    name: 'Anthony Taylor',
    number: 0,
    position: 'Referee',
    team: 'home',
    teamColor: '#F97316',
    avatar: '',
    stats: {
      passes: 0,
      passAccuracy: 0,
      shots: 0,
      shotsOnTarget: 0,
      tackles: 0,
      distance: 0,
      speed: 0,
      sprints: scored ? 47 : 46, // Calls made this match (increments on goal signal)
    },
    fieldPosition: { x: 50, y: 30 },
  },
  {
    id: 'assistant-referee',
    name: 'G. Mayfield',
    number: 0,
    position: 'Assistant Referee',
    team: 'home',
    teamColor: '#F97316',
    avatar: '',
    stats: {
      passes: 0,
      passAccuracy: 0,
      shots: 0,
      shotsOnTarget: 0,
      tackles: 0,
      distance: 0,
      speed: 0,
      sprints: 156, // Career matches
    },
    fieldPosition: { x: 95, y: 70 },
  },
];

// Initial card positions on screen (fallback when tracking unavailable)
const initialCardPositions: Record<string, { x: number; y: number }> = {
  'referee': { x: 600, y: 200 },
  'penalty-taker': { x: 100, y: 300 },
  'goalkeeper': { x: 1000, y: 150 },
  'assistant-referee': { x: 1000, y: 350 },
};

// Card dimensions for offset calculations
const CARD_WIDTH = 180;
const CARD_HEIGHT = 160;

export function StreamPage() {
  const navigate = useNavigate();
  const { announce } = useAccessibility();
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [selectedPlayer, setSelectedPlayer] = useState<Player | null>(null);
  const [containerSize, setContainerSize] = useState({ width: 0, height: 0 });
  const [showPOVCards, setShowPOVCards] = useState(false);
  const [penaltyData, setPenaltyData] = useState(initialPenaltyData);
  const [goalScored, setGoalScored] = useState(false);
  const [showConfetti, setShowConfetti] = useState(false);

  const streamRef = useRef<StreamContainerRef>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Get tracking positions synced with video time
  const { positions: trackingPositions } = usePlayerTracking(currentTime);

  // Update container size on mount and resize
  useEffect(() => {
    const updateSize = () => {
      if (containerRef.current) {
        setContainerSize({
          width: containerRef.current.clientWidth,
          height: containerRef.current.clientHeight,
        });
      }
    };

    updateSize();
    window.addEventListener('resize', updateSize);
    return () => window.removeEventListener('resize', updateSize);
  }, []);

  // Detect goal scored at 8 seconds
  useEffect(() => {
    if (currentTime >= GOAL_TIME && !goalScored) {
      setGoalScored(true);
      setShowConfetti(true);
      setPenaltyData(prev => ({
        ...prev,
        homePenalties: {
          scored: 1,
          taken: 1,
          saved: 0,
        },
        isHomeTurn: false,
      }));
      // Announce goal for screen readers
      announce('Goal! Bruno Fernandes scores for Manchester United!', 'assertive');
      // Auto-hide confetti after animation
      setTimeout(() => setShowConfetti(false), 4000);
    }
    // Reset if video is rewound before goal time
    if (currentTime < GOAL_TIME && goalScored) {
      setGoalScored(false);
      setShowConfetti(false);
      setPenaltyData(initialPenaltyData);
    }
  }, [currentTime, goalScored, announce]);

  // Calculate card positions from tracking data
  const getCardPosition = (playerId: string) => {
    const tracking = trackingPositions[playerId];

    if (tracking?.visible && containerSize.width > 0) {
      // Convert percentage to pixels, position card above player's head
      const VERTICAL_OFFSET = 95; // Extra offset to clear the player's head
      return {
        x: (tracking.x / 100) * containerSize.width - CARD_WIDTH / 2,
        y: (tracking.y / 100) * containerSize.height - CARD_HEIGHT - VERTICAL_OFFSET,
      };
    }

    // Fallback to initial positions if tracking not available
    return initialCardPositions[playerId] || { x: 0, y: 0 };
  };

  const handleTimeUpdate = useCallback((time: number, dur: number) => {
    setCurrentTime(time);
    setDuration(dur);
  }, []);

  const handleStateChange = useCallback((playing: boolean) => {
    setIsPlaying(playing);
  }, []);

  const handlePlayPause = () => {
    if (isPlaying) {
      streamRef.current?.pause();
    } else {
      streamRef.current?.play();
    }
  };

  const handleSeek = (time: number) => {
    streamRef.current?.seekTo(time);
  };

  const handleFullscreen = () => {
    document.documentElement.requestFullscreen?.();
  };

  const handlePlayerExpand = (player: Player) => {
    setSelectedPlayer(player);
  };

  const handleCloseOverlay = () => {
    setSelectedPlayer(null);
  };

  return (
    <div className="app" ref={containerRef}>
      <StreamContainer
        ref={streamRef}
        videoSrc={brunoVideo}
        onTimeUpdate={handleTimeUpdate}
        onStateChange={handleStateChange}
      >
        {/* Home Button */}
        <button
          className="home-button"
          onClick={() => navigate('/')}
          aria-label="Return to home"
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M15 18l-6-6 6-6" />
          </svg>
          <span>Back</span>
        </button>

        {/* Toggle POV Cards Button */}
        <button
          className="toggle-pov-button"
          onClick={() => setShowPOVCards(!showPOVCards)}
          aria-label={showPOVCards ? 'Hide POV cards' : 'Show POV cards'}
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
            {showPOVCards ? (
              <><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" /><circle cx="12" cy="12" r="3" /></>
            ) : (
              <><path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24" /><line x1="1" y1="1" x2="23" y2="23" /></>
            )}
          </svg>
          <span>{showPOVCards ? 'POV' : 'POV'}</span>
        </button>

        <PenaltyScoreOverlay
          homeTeam={penaltyData.homeTeam}
          awayTeam={penaltyData.awayTeam}
          homePenalties={penaltyData.homePenalties}
          awayPenalties={penaltyData.awayPenalties}
          currentRound={penaltyData.currentRound}
          isHomeTurn={penaltyData.isHomeTurn}
          justScored={goalScored ? 'home' : null}
        />

        <LiveBadge />

        {/* Goal Confetti */}
        <GoalConfetti active={showConfetti} />

        {/* POV Player Cards - positioned by tracking */}
        {showPOVCards && getPlayers(goalScored).map((player, index) => {
          const pos = getCardPosition(player.id);
          const isVisible = trackingPositions[player.id]?.visible !== false;

          return (
            <div
              key={player.id}
              className="player-card-wrapper"
              style={{
                left: pos.x,
                top: pos.y,
                animationDelay: `${index * 100}ms`,
                opacity: isVisible ? 1 : 0.3,
                transition: 'left 0.15s linear, top 0.15s linear',
              }}
            >
              <PlayerCard
                player={player}
                onExpand={handlePlayerExpand}
                mainVideoTime={currentTime}
                isMainPlaying={isPlaying}
              />
            </div>
          );
        })}

        <VideoControls
          currentTime={currentTime}
          duration={duration}
          isPlaying={isPlaying}
          onPlayPause={handlePlayPause}
          onSeek={handleSeek}
          onFullscreen={handleFullscreen}
        />
      </StreamContainer>

      {/* POV Video Overlay */}
      {selectedPlayer && (
        <POVVideoOverlay
          player={selectedPlayer}
          onClose={handleCloseOverlay}
          mainVideoTime={currentTime}
          isMainPlaying={isPlaying}
        />
      )}

      <AccessibilityToggle />
    </div>
  );
}
