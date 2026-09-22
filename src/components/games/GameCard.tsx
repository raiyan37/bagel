import { useNavigate } from 'react-router-dom';
import './games-glass.css';

export interface Game {
  id: string;
  homeTeam: {
    name: string;
    shortName: string;
    color: string;
  };
  awayTeam: {
    name: string;
    shortName: string;
    color: string;
  };
  competition: string;
  competitionShort: string;
  time: string;
  status: 'live' | 'upcoming' | 'replay';
}

interface GameCardProps {
  game: Game;
}

export function GameCard({ game }: GameCardProps) {
  const navigate = useNavigate();

  const handleClick = () => {
    navigate(`/processing/${game.id}`);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      handleClick();
    }
  };

  const statusLabel = game.status === 'live'
    ? 'Live now'
    : game.status === 'replay'
      ? 'Replay available'
      : `Starts at ${game.time}`;

  return (
    <article
      className="game-card"
      onClick={handleClick}
      onKeyDown={handleKeyDown}
      role="listitem"
      tabIndex={0}
      aria-label={`${game.homeTeam.name} versus ${game.awayTeam.name}, ${game.competition}. ${statusLabel}. Press Enter to watch.`}
    >
      <div className="game-card-content">
        {/* Teams - text only */}
        <div className="game-card-teams">
          <div className="game-card-team">
            <span className="game-card-team-name">{game.homeTeam.name}</span>
          </div>

          <div className="game-card-vs" aria-hidden="true">
            <span>v</span>
          </div>

          <div className="game-card-team">
            <span className="game-card-team-name">{game.awayTeam.name}</span>
          </div>
        </div>

        {/* Meta row */}
        <div className="game-card-footer">
          {game.status === 'live' ? (
            <div className="game-card-live" aria-label="Live now">
              <span className="game-card-live-dot" aria-hidden="true" />
              <span className="game-card-live-text">LIVE</span>
            </div>
          ) : game.status === 'replay' ? (
            <div className="game-card-replay">
              <span>REPLAY</span>
            </div>
          ) : (
            <div className="game-card-time">
              <span>{game.time}</span>
            </div>
          )}

          <span className="game-card-sep" aria-hidden="true" />
          <span className="game-card-comp-text">{game.competition}</span>
        </div>
      </div>

      {/* Arrow indicator */}
      <div className="game-card-arrow" aria-hidden="true">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
          <path d="M5 12h14M12 5l7 7-7 7" />
        </svg>
      </div>
    </article>
  );
}
