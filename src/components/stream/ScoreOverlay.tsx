import type { Match } from '../../types';
import './styles.css';

interface ScoreOverlayProps {
  match: Match;
}

export function ScoreOverlay({ match }: ScoreOverlayProps) {
  return (
    <div className="score-overlay glass glass-noise animate-fade-in-scale">
      <div className="score-teams">
        <div className="score-team">
          <span className="team-flag">{match.homeTeam.flag}</span>
          <span className="team-name">{match.homeTeam.shortName}</span>
        </div>
        <div className="score-value">
          <span className="score-number">{match.homeScore}</span>
          <span className="score-separator">-</span>
          <span className="score-number">{match.awayScore}</span>
        </div>
        <div className="score-team">
          <span className="team-name">{match.awayTeam.shortName}</span>
          <span className="team-flag">{match.awayTeam.flag}</span>
        </div>
      </div>
      <div className="score-time glass-dark">
        <span className="time-value">{match.time}</span>
      </div>
    </div>
  );
}
